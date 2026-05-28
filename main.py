"""
main.py - 布局算法的 I/O、测试、验证入口

功能:
  1. 单用例/批量运行算法，保存输出
  2. --validate: 验证 expected_output 和算法输出的所有约束
  3. --expected-only: 只验证 expected_output
  4. --strategies: 指定初始化策略
  5. 测试用例目录自动发现

用法:
  python main.py input.json [--time 115]                     # 单用例运行
  python main.py case1.json case2.json [--time 115]          # 批量运行
  python main.py -d ./test_cases [--time 30]                 # 目录批量
  python main.py -d ./test_cases --validate [--time 30]      # 目录 + 约束验证
  python main.py -d ./test_cases --validate --expected-only  # 只验证预期输出
  python main.py --validate case01                           # 按名称过滤
  python main.py -t [--time 30]                              # 内置测试用例
  python main.py --list-strategies                           # 列出可用策略
  python main.py --list-cases                                # 列出测试用例

初始化策略选项:
  --strategies s1,s2,s3    指定策略序列（逗号分隔）
  --list-strategies        列出所有可用策略

可用策略:
  network_aware      - 基于网络连接的BFS布局
  compact_grid       - 紧凑网格布局
  random_tight       - 随机紧密布局
  clustered          - 聚类布局
  quadratic_placement - 二次规划布局（推荐）
"""
import json
import sys
import os
import glob
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyArrowPatch
import matplotlib.font_manager as fm
import numpy as np

# 配置中文字体 - 直接使用字体文件路径
_ZH_FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
if os.path.exists(_ZH_FONT_PATH):
    fm.fontManager.addfont(_ZH_FONT_PATH)
    _zh_font_prop = fm.FontProperties(fname=_ZH_FONT_PATH)
    plt.rcParams['font.sans-serif'] = [_zh_font_prop.get_name()]
else:
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from route import (Problem, ConstraintSystem, SimulatedAnnealing,
                   solve, compute_cost, compute_hpwl, compute_area,
                   compute_overlap_penalty)


# ============================================================
# 输入合法性检查
# ============================================================

def _check_id(box_id: int, n: int, context: str):
    """检查矩形 ID 是否在有效范围内"""
    if box_id < 1 or box_id > n:
        raise ValueError(f"{context}中引用的 box {box_id} 超出范围 [1, {n}]")


def _check_symmetry_groups(data: Dict[str, Any], widths: List[float], 
                           heights: List[float], n: int, eps: float):
    """检查对称组约束的合法性"""
    for axis in ["symmetry_x", "symmetry_y"]:
        for group in data.get(axis, []):
            # 检查 symmetry_pair
            for pair in group.get("symmetry_pair", []):
                if len(pair) != 2:
                    raise ValueError(
                        f"{axis} 的 symmetry_pair 必须是二元组，得到 {pair}"
                    )
                i, j = pair
                _check_id(i, n, f"{axis} symmetry_pair")
                _check_id(j, n, f"{axis} symmetry_pair")
                
                # 对称的两个矩形尺寸必须相同
                if abs(widths[i-1] - widths[j-1]) > eps or abs(heights[i-1] - heights[j-1]) > eps:
                    raise ValueError(
                        f"{axis} 对称组尺寸矛盾：box {i} 尺寸为 [{widths[i-1]}, {heights[i-1]}]，"
                        f"box {j} 尺寸为 [{widths[j-1]}, {heights[j-1]}]。"
                        f"对称约束要求两个矩形尺寸必须相同。"
                    )
            
            # 检查 self_symmetry
            for s in group.get("self_symmetry", []):
                _check_id(s, n, f"{axis} self_symmetry")


def _check_repeat_groups(data: Dict[str, Any], widths: List[float], 
                         heights: List[float], n: int, eps: float):
    """检查重复组约束的合法性"""
    repeat_groups = [rg["groups"] for rg in data.get("repeat_groups", [])]
    
    for rg in repeat_groups:
        if len(rg) < 2:
            continue
        
        ref_group = rg[0]
        ref_len = len(ref_group)
        
        # 检查所有引用 ID
        for box_id in ref_group:
            _check_id(box_id, n, "重复组")
        
        for g_idx in range(1, len(rg)):
            group = rg[g_idx]
            
            # 检查每个 group 的矩形个数必须相同
            if len(group) != ref_len:
                raise ValueError(
                    f"重复组结构错误：参考组有 {ref_len} 个矩形 {ref_group}，"
                    f"但组 {g_idx} 有 {len(group)} 个矩形 {group}，个数不一致"
                )
            
            # 检查引用 ID
            for box_id in group:
                _check_id(box_id, n, "重复组")
            
            # 检查对应位置的矩形尺寸必须相同
            for ref_box_id, box_id in zip(ref_group, group):
                ref_w, ref_h = widths[ref_box_id - 1], heights[ref_box_id - 1]
                box_w, box_h = widths[box_id - 1], heights[box_id - 1]
                if abs(ref_w - box_w) > eps or abs(ref_h - box_h) > eps:
                    raise ValueError(
                        f"重复组尺寸矛盾：参考组的 box {ref_box_id} 尺寸为 [{ref_w}, {ref_h}]，"
                        f"但组 {g_idx} 的 box {box_id} 尺寸为 [{box_w}, {box_h}]。"
                        f"重复组要求对应位置的 box 尺寸必须相同，此问题无解。"
                    )


def _check_alignment_constraints(data: Dict[str, Any], n: int):
    """检查对齐约束的 ID 合法性"""
    align = data.get("align", {})
    for direction, groups in align.items():
        for group in groups:
            for box_id in group:
                _check_id(box_id, n, f"align.{direction}")


def _check_network_constraints(data: Dict[str, Any], n: int):
    """检查网络约束的 ID 合法性"""
    for net_id, net in enumerate(data.get("nets", [])):
        for box_id in net:
            _check_id(box_id, n, f"nets[{net_id}]")


def validate_input(data: Dict[str, Any], eps: float = 1e-6):
    """
    检查输入数据的合法性和可解性。
    
    检查项:
    1. 矩形 ID 范围：所有约束引用的 ID 必须在 [1, n] 范围内
    2. 对称组：
       - symmetry_pair 中两个矩形尺寸必须相同
       - self_symmetry 矩形必须存在
    3. 重复组：
       - 每个 group 的矩形个数必须相同
       - 对应位置的矩形尺寸必须相同
    
    Raises:
        ValueError: 输入数据存在根本矛盾，无解
    """
    widths = [s[0] for s in data["box_size"]]
    heights = [s[1] for s in data["box_size"]]
    n = len(widths)
    
    _check_symmetry_groups(data, widths, heights, n, eps)
    _check_repeat_groups(data, widths, heights, n, eps)
    _check_alignment_constraints(data, n)
    _check_network_constraints(data, n)


# ============================================================
# 可视化
# ============================================================

# 颜色方案
_COLORS = {
    'sym_x': '#FF6B6B',      # 红
    'sym_y': '#4ECDC4',      # 青
    'align': '#45B7D1',      # 蓝
    'repeat': '#96CEB4',     # 绿
    'default': '#D4A574',    # 棕
    'net': '#FF8C42',        # 橙
    'self_sym': '#FFEAA7',   # 黄
}


def _get_box_colors(data: Dict[str, Any]) -> Dict[int, str]:
    """根据约束类型为每个box分配颜色"""
    n = len(data["box_size"])
    colors = {i: _COLORS['default'] for i in range(n)}

    # 对称组着色（优先级高）
    for group in data.get("symmetry_x", []):
        for pair in group.get("symmetry_pair", []):
            colors[pair[0] - 1] = _COLORS['sym_x']
            colors[pair[1] - 1] = _COLORS['sym_x']
        for s in group.get("self_symmetry", []):
            colors[s - 1] = _COLORS['self_sym']

    for group in data.get("symmetry_y", []):
        for pair in group.get("symmetry_pair", []):
            colors[pair[0] - 1] = _COLORS['sym_y']
            colors[pair[1] - 1] = _COLORS['sym_y']
        for s in group.get("self_symmetry", []):
            colors[s - 1] = _COLORS['self_sym']

    # 重复组着色
    for rg in data.get("repeat_groups", []):
        for group in rg.get("groups", []):
            for box_id in group:
                colors[box_id - 1] = _COLORS['repeat']

    return colors


def _draw_boxes(ax, data: Dict[str, Any], positions: List[List[float]],
                title: str, show_ids: bool = True):
    """在ax上绘制布局"""
    widths = [s[0] for s in data["box_size"]]
    heights = [s[1] for s in data["box_size"]]
    n = len(widths)
    colors = _get_box_colors(data)

    for i in range(n):
        x, y = positions[i][0], positions[i][1]
        w, h = widths[i], heights[i]
        color = colors[i]

        rect = patches.Rectangle((x, y), w, h,
                                  linewidth=1.5, edgecolor='#333',
                                  facecolor=color, alpha=0.7)
        ax.add_patch(rect)

        if show_ids:
            cx = x + w / 2
            cy = y + h / 2
            fontsize = max(6, min(10, min(w, h) * 0.8))
            ax.text(cx, cy, str(i + 1),
                    ha='center', va='center',
                    fontsize=fontsize, fontweight='bold', color='#333')

    # 计算边界
    if positions:
        min_x = min(p[0] for p in positions)
        min_y = min(p[1] for p in positions)
        max_x = max(positions[i][0] + widths[i] for i in range(n))
        max_y = max(positions[i][1] + heights[i] for i in range(n))
        pad_x = (max_x - min_x) * 0.1 + 2
        pad_y = (max_y - min_y) * 0.1 + 2
        ax.set_xlim(min_x - pad_x, max_x + pad_x)
        ax.set_ylim(min_y - pad_y, max_y + pad_y)

    ax.set_aspect('equal')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')



# ---- 约束组提取 ----

def _extract_constraint_groups(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    将输入约束拆分为独立的组，每组一个 dict:
    {
      "type": "sym_x" | "sym_y" | "repeat" | "align" | "net",
      "label": str,        # 子图标题
      "color": str,        # 主色
      "box_ids": [int],    # 1-based box IDs
      "sub_groups": {...}, # 内部子结构
    }
    """
    groups = []

    # X 对称
    for gi, g in enumerate(data.get("symmetry_x", [])):
        pairs = g.get("symmetry_pair", [])
        selfs = g.get("self_symmetry", [])
        ids = set()
        for p in pairs:
            ids.update(p)
        ids.update(selfs)
        groups.append({
            "type": "sym_x", "label": f"X对称组{gi}",
            "color": _COLORS['sym_x'],
            "box_ids": sorted(ids),
            "sub_groups": {"pairs": pairs, "selfs": selfs},
        })

    # Y 对称
    for gi, g in enumerate(data.get("symmetry_y", [])):
        pairs = g.get("symmetry_pair", [])
        selfs = g.get("self_symmetry", [])
        ids = set()
        for p in pairs:
            ids.update(p)
        ids.update(selfs)
        groups.append({
            "type": "sym_y", "label": f"Y对称组{gi}",
            "color": _COLORS['sym_y'],
            "box_ids": sorted(ids),
            "sub_groups": {"pairs": pairs, "selfs": selfs},
        })

    # 重复组
    for ri, rg in enumerate(data.get("repeat_groups", [])):
        all_ids = []
        for grp in rg.get("groups", []):
            all_ids.extend(grp)
        groups.append({
            "type": "repeat", "label": f"重复组{ri}",
            "color": _COLORS['repeat'],
            "box_ids": sorted(set(all_ids)),
            "sub_groups": {"groups": rg.get("groups", [])},
        })

    # 对齐
    align = data.get("align", {})
    for direction in ["left", "right", "top", "bottom"]:
        for ai, grp in enumerate(align.get(direction, [])):
            if not grp:
                continue
            groups.append({
                "type": "align", "label": f"对齐-{direction}[{ai}]",
                "color": _COLORS['align'],
                "box_ids": sorted(grp),
                "sub_groups": {"direction": direction, "group": grp},
            })

    # 网络
    nets = data.get("nets", [])
    for ni, net in enumerate(nets):
        groups.append({
            "type": "net", "label": f"Net{ni}",
            "color": _COLORS['net'],
            "box_ids": sorted(net),
            "sub_groups": {"net": net},
        })

    return groups


# ---- 单个约束组子图绘制 ----

def _draw_constraint_group_panel(ax, group: Dict[str, Any],
                                  widths: List[float], heights: List[float]):
    """在单个 ax 上绘制一个约束组：box 缩略图 + 关系连线"""
    box_ids = group["box_ids"]  # 1-based
    color = group["color"]
    gtype = group["type"]
    n = len(box_ids)
    if n == 0:
        ax.axis('off')
        return

    # 布局：紧凑网格
    cols = max(2, int(np.ceil(np.sqrt(n))))
    cell_w = 28
    cell_h = 22
    positions = {}
    for idx, bid in enumerate(box_ids):
        row = idx // cols
        col = idx % cols
        cx = col * cell_w + cell_w / 2
        cy = -(row * cell_h) - cell_h / 2
        positions[bid] = (cx, cy)

    # 绘制 box 缩略块
    max_dim = max(max(widths), max(heights)) if widths else 1
    scale = 12.0 / max_dim if max_dim > 0 else 1.0
    for bid in box_ids:
        cx, cy = positions[bid]
        w = widths[bid - 1] * scale
        h = heights[bid - 1] * scale
        rect = patches.FancyBboxPatch(
            (cx - w/2, cy - h/2), w, h,
            boxstyle="round,pad=0.5",
            linewidth=1.5, edgecolor='#444',
            facecolor=color, alpha=0.75)
        ax.add_patch(rect)
        ax.text(cx, cy, str(bid), ha='center', va='center',
                fontsize=8, fontweight='bold', color='white')
        ax.text(cx, cy - h/2 - 1.5, f"{widths[bid-1]:.0f}×{heights[bid-1]:.0f}",
                ha='center', va='top', fontsize=5.5, color='#666')

    sub = group["sub_groups"]

    if gtype in ("sym_x", "sym_y"):
        pairs = sub.get("pairs", [])
        selfs = sub.get("selfs", [])
        axis_label = "X轴" if gtype == "sym_x" else "Y轴"
        for pair in pairs:
            if pair[0] in positions and pair[1] in positions:
                x1, y1 = positions[pair[0]]
                x2, y2 = positions[pair[1]]
                rad = 0.3 if gtype == "sym_x" else -0.3
                ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                            arrowprops=dict(arrowstyle='<->', color=color,
                                            lw=1.8, connectionstyle=f'arc3,rad={rad}'))
                mx, my = (x1+x2)/2, (y1+y2)/2
                offset = 2.5 if gtype == "sym_x" else -2.5
                ax.text(mx, my + offset, '⇔', fontsize=8, color=color,
                        ha='center', fontweight='bold')
        for s in selfs:
            if s in positions:
                cx, cy = positions[s]
                ax.plot([cx, cx], [cy + 6, cy - 6], color=color, lw=2,
                        linestyle='--', alpha=0.7)
                ax.text(cx + 3, cy + 6, f"自{axis_label}", fontsize=5.5,
                        color=color, fontweight='bold')

    elif gtype == "repeat":
        grp_lists = sub.get("groups", [])
        group_colors = ['#E17055', '#00B894', '#6C5CE7', '#FDCB6E']
        for gi, grp in enumerate(grp_lists):
            gc = group_colors[gi % len(group_colors)]
            grp_positions = [positions[bid] for bid in grp if bid in positions]
            if grp_positions:
                min_x = min(p[0] for p in grp_positions) - cell_w * 0.45
                max_x = max(p[0] for p in grp_positions) + cell_w * 0.45
                min_y = min(p[1] for p in grp_positions) - cell_h * 0.45
                max_y = max(p[1] for p in grp_positions) + cell_h * 0.45
                rect = patches.Rectangle(
                    (min_x, min_y), max_x - min_x, max_y - min_y,
                    linewidth=2, edgecolor=gc, facecolor=gc, alpha=0.08,
                    linestyle='--')
                ax.add_patch(rect)
                ax.text(min_x + 1, max_y - 0.5, f"G{gi}", fontsize=6,
                        color=gc, fontweight='bold')
            for k in range(len(grp) - 1):
                if grp[k] in positions and grp[k+1] in positions:
                    x1, y1 = positions[grp[k]]
                    x2, y2 = positions[grp[k+1]]
                    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                                arrowprops=dict(arrowstyle='-', color=gc,
                                                lw=1.5, connectionstyle='arc3,rad=0'))
        if len(grp_lists) >= 2:
            for gi in range(1, len(grp_lists)):
                ref = grp_lists[0]
                grp = grp_lists[gi]
                for k in range(min(len(ref), len(grp))):
                    if ref[k] in positions and grp[k] in positions:
                        x1, y1 = positions[ref[k]]
                        x2, y2 = positions[grp[k]]
                        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                                    arrowprops=dict(arrowstyle='->', color='#999',
                                                    lw=1, linestyle='dotted',
                                                    connectionstyle='arc3,rad=0.2'))

    elif gtype == "align":
        direction = sub.get("direction", "")
        grp = sub.get("group", [])
        grp_positions = [positions[bid] for bid in grp if bid in positions]
        if len(grp_positions) >= 2:
            if direction == "left":
                line_x = min(p[0] for p in grp_positions) - 5
                ys = [p[1] for p in grp_positions]
                ax.plot([line_x, line_x], [min(ys) - 5, max(ys) + 5],
                        color=color, lw=2.5, alpha=0.8)
                # 箭头：竖线 + 右向短横
                for yy in [min(ys) - 5, max(ys) + 5]:
                    ax.annotate('', xy=(line_x + 3, yy), xytext=(line_x, yy),
                                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
                ax.text(line_x - 1, max(ys) + 8, 'left', fontsize=6,
                        color=color, ha='center', fontweight='bold')
            elif direction == "right":
                line_x = max(p[0] for p in grp_positions) + 5
                ys = [p[1] for p in grp_positions]
                ax.plot([line_x, line_x], [min(ys) - 5, max(ys) + 5],
                        color=color, lw=2.5, alpha=0.8)
                for yy in [min(ys) - 5, max(ys) + 5]:
                    ax.annotate('', xy=(line_x - 3, yy), xytext=(line_x, yy),
                                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
                ax.text(line_x + 1, max(ys) + 8, 'right', fontsize=6,
                        color=color, ha='center', fontweight='bold')
            elif direction == "top":
                line_y = max(p[1] for p in grp_positions) + 5
                xs = [p[0] for p in grp_positions]
                ax.plot([min(xs) - 5, max(xs) + 5], [line_y, line_y],
                        color=color, lw=2.5, alpha=0.8)
                for xx in [min(xs) - 5, max(xs) + 5]:
                    ax.annotate('', xy=(xx, line_y - 3), xytext=(xx, line_y),
                                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
                ax.text(max(xs) + 8, line_y, 'top', fontsize=6,
                        color=color, ha='left', va='center', fontweight='bold')
            elif direction == "bottom":
                line_y = min(p[1] for p in grp_positions) - 5
                xs = [p[0] for p in grp_positions]
                ax.plot([min(xs) - 5, max(xs) + 5], [line_y, line_y],
                        color=color, lw=2.5, alpha=0.8)
                for xx in [min(xs) - 5, max(xs) + 5]:
                    ax.annotate('', xy=(xx, line_y + 3), xytext=(xx, line_y),
                                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
                ax.text(max(xs) + 8, line_y, 'bottom', fontsize=6,
                        color=color, ha='left', va='center', fontweight='bold')
        for k in range(len(grp) - 1):
            if grp[k] in positions and grp[k+1] in positions:
                x1, y1 = positions[grp[k]]
                x2, y2 = positions[grp[k+1]]
                ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                            arrowprops=dict(arrowstyle='-', color=color,
                                            lw=1.2, connectionstyle='arc3,rad=0.15',
                                            alpha=0.5))

    elif gtype == "net":
        net = sub.get("net", [])
        if len(net) >= 2 and net[0] in positions:
            cx, cy = positions[net[0]]
            for k in range(1, len(net)):
                if net[k] in positions:
                    x2, y2 = positions[net[k]]
                    ax.annotate('', xy=(x2, y2), xytext=(cx, cy),
                                arrowprops=dict(arrowstyle='->', color=color,
                                                lw=1.5, connectionstyle='arc3,rad=0.15',
                                                alpha=0.7))

    # 设置边界
    all_x = [p[0] for p in positions.values()]
    all_y = [p[1] for p in positions.values()]
    pad = 8
    ax.set_xlim(min(all_x) - cell_w/2 - pad, max(all_x) + cell_w/2 + pad)
    ax.set_ylim(min(all_y) - cell_h/2 - pad, max(all_y) + cell_h/2 + pad)
    ax.set_aspect('equal')
    ax.set_title(group["label"], fontsize=9, fontweight='bold',
                 color=color, pad=3)
    ax.axis('off')


def _draw_input_constraints_overview(data: Dict[str, Any], ax):
    """绘制输入约束总览图（所有 box + 颜色编码）"""
    n = len(data["box_size"])
    widths = [s[0] for s in data["box_size"]]
    heights = [s[1] for s in data["box_size"]]
    colors = _get_box_colors(data)

    cols = max(3, int(np.ceil(np.sqrt(n))))
    positions = {}
    max_w = max(widths) * 1.5
    max_h = max(heights) * 1.5
    for i in range(n):
        row = i // cols
        col = i % cols
        positions[i] = [col * max_w, -row * max_h]

    for i in range(n):
        x, y = positions[i][0], positions[i][1]
        w, h = widths[i] * 0.6, heights[i] * 0.6
        color = colors[i]
        rect = patches.Rectangle((x, y), w, h,
                                  linewidth=1.5, edgecolor='#333',
                                  facecolor=color, alpha=0.7)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, f"{i+1}\n{w:.0f}×{h:.0f}",
                ha='center', va='center', fontsize=7, fontweight='bold')

    for group in data.get("symmetry_x", []):
        for pair in group.get("symmetry_pair", []):
            i, j = pair[0] - 1, pair[1] - 1
            x1, y1 = positions[i][0] + widths[i]*0.3, positions[i][1] + heights[i]*0.3
            x2, y2 = positions[j][0] + widths[j]*0.3, positions[j][1] + heights[j]*0.3
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle='<->', color=_COLORS['sym_x'],
                                       lw=1.5, connectionstyle='arc3,rad=0.3'))

    for group in data.get("symmetry_y", []):
        for pair in group.get("symmetry_pair", []):
            i, j = pair[0] - 1, pair[1] - 1
            x1, y1 = positions[i][0] + widths[i]*0.3, positions[i][1] + heights[i]*0.3
            x2, y2 = positions[j][0] + widths[j]*0.3, positions[j][1] + heights[j]*0.3
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle='<->', color=_COLORS['sym_y'],
                                       lw=1.5, connectionstyle='arc3,rad=-0.3'))

    legend_items = [
        patches.Patch(color=_COLORS['sym_x'], alpha=0.7, label='X对称'),
        patches.Patch(color=_COLORS['sym_y'], alpha=0.7, label='Y对称'),
        patches.Patch(color=_COLORS['self_sym'], alpha=0.7, label='自对称'),
        patches.Patch(color=_COLORS['repeat'], alpha=0.7, label='重复组'),
        patches.Patch(color=_COLORS['align'], alpha=0.7, label='对齐'),
        patches.Patch(color=_COLORS['net'], alpha=0.7, label='网络'),
        patches.Patch(color=_COLORS['default'], alpha=0.7, label='无约束'),
    ]
    ax.legend(handles=legend_items, loc='upper right', fontsize=7)

    all_x = [p[0] for p in positions.values()]
    all_y = [p[1] for p in positions.values()]
    ax.set_xlim(min(all_x) - max_w, max(all_x) + max_w * 2)
    ax.set_ylim(min(all_y) - max_h, max(all_y) + max_h)
    ax.set_aspect('equal')
    ax.set_title('全局总览', fontsize=11, fontweight='bold')
    ax.axis('off')


def visualize_case(data: Dict[str, Any], algo_result: Dict[str, Any] = None,
                   case_name: str = "", save_path: str = None):
    """
    可视化用例：约束分组面板 + 预期布局 + 算法布局

    布局策略（嵌套 GridSpec）：
      - 上半区：约束分组面板（每个约束组独立子图，box 可重复出现）
      - 下半区：全局总览 + 预期布局 + 算法布局（并排大面板）

    Args:
        data: 测试用例数据 (含 input 和 expected_output)
        algo_result: 算法输出结果 (可选)
        case_name: 用例名称
        save_path: 保存路径 (可选，不指定则显示)
    """
    from matplotlib.gridspec import GridSpec

    input_data = data.get("input", data)
    expected = data.get("expected_output", {})
    desc = data.get("description", case_name)

    widths = [s[0] for s in input_data["box_size"]]
    heights = [s[1] for s in input_data["box_size"]]

    # 提取约束组
    groups = _extract_constraint_groups(input_data)
    n_groups = len(groups)

    has_expected = expected and expected.get("box_position")
    has_algo = algo_result and algo_result.get("box_position")

    # --- 布局计算 ---
    # 约束面板列数（最多6列）
    max_cols = 6
    if n_groups <= max_cols:
        top_cols = max(n_groups, 1)
        top_rows = 1
    else:
        top_cols = max_cols
        top_rows = int(np.ceil(n_groups / max_cols))

    # 底部面板数量
    n_bottom = 1 + (1 if has_expected else 0) + (1 if has_algo else 0)

    # 总列数 = max(约束列数, 底部列数)
    total_cols = max(top_cols, n_bottom)

    # 高度比：约束区占 40%，底部占 60%
    fig = plt.figure(figsize=(4.5 * total_cols, 4.5 * top_rows + 8))
    
    # 主 GridSpec：2行（约束区 + 底部区）
    gs_main = GridSpec(2, 1, figure=fig, height_ratios=[top_rows * 4, 8], hspace=0.3)
    
    # 上半区：约束面板子 GridSpec
    gs_top = gs_main[0].subgridspec(top_rows, top_cols, hspace=0.3, wspace=0.25)
    
    # 下半区：底部子 GridSpec
    gs_bottom = gs_main[1].subgridspec(1, n_bottom, wspace=0.2)

    fig.suptitle(desc, fontsize=16, fontweight='bold', y=0.98)

    # --- 上半区：约束组面板 ---
    for idx, group in enumerate(groups):
        r = idx // top_cols
        c = idx % top_cols
        ax = fig.add_subplot(gs_top[r, c])
        _draw_constraint_group_panel(ax, group, widths, heights)

    # --- 下半区：总览 + 预期 + 算法 ---
    plot_idx = 0

    # 全局总览
    ax_overview = fig.add_subplot(gs_bottom[0, plot_idx])
    _draw_input_constraints_overview(input_data, ax_overview)
    plot_idx += 1

    # 预期布局
    if has_expected:
        ax_exp = fig.add_subplot(gs_bottom[0, plot_idx])
        exp_cost = ""
        try:
            problem = Problem(input_data)
            exp_x = [p[0] for p in expected["box_position"]]
            exp_y = [p[1] for p in expected["box_position"]]
            _, hpwl, area, _ = compute_cost(problem, exp_x, exp_y)
            exp_cost = f"\nCost={10*hpwl+area:.0f} HPWL={hpwl:.0f} Area={area:.0f}"
        except:
            pass
        _draw_boxes(ax_exp, input_data, expected["box_position"],
                    f"预期布局{exp_cost}")
        plot_idx += 1

    # 算法布局
    if has_algo:
        ax_algo = fig.add_subplot(gs_bottom[0, plot_idx])
        algo_cost = ""
        if "cost" in algo_result:
            algo_cost = (f"\nCost={algo_result['cost']:.0f} "
                         f"HPWL={algo_result['hpwl']:.0f} "
                         f"Area={algo_result['area']:.0f}")
        _draw_boxes(ax_algo, input_data, algo_result["box_position"],
                    f"算法布局{algo_cost}")
        plot_idx += 1

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"📊 可视化已保存: {save_path}")
    else:
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                f"viz_{case_name}.png")
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"📊 可视化已保存: {out_path}")


def viz_from_files(input_file: str, output_file: str = None,
                   viz_dir: str = 'visualizations') -> None:
    """
    从输入文件和可选的算法输出文件生成可视化图。

    Args:
        input_file: 输入用例 JSON 文件路径
        output_file: 算法输出 JSON 文件路径 (可选)
        viz_dir: 可视化输出目录
    """
    case_name = os.path.basename(input_file).replace(".json", "")

    with open(input_file, 'r') as f:
        data = json.load(f)

    algo_result = None
    if output_file and os.path.exists(output_file):
        with open(output_file, 'r') as f:
            algo_result = json.load(f)
        print(f"📂 输入: {input_file}")
        print(f"📂 输出: {output_file}")
    else:
        print(f"📂 输入: {input_file}")
        if output_file:
            print(f"⚠️  输出文件不存在: {output_file}")

    viz_path = os.path.join(viz_dir, f"{case_name}.png")
    visualize_case(data, algo_result=algo_result,
                   case_name=case_name, save_path=viz_path)


def solve_and_report(problem: Problem, time_limit: float = 115.0,
                     strategies: List[str] = None) -> Dict[str, Any]:
    """
    调用 solve() 并计算成本报告。
    solve() 返回原始坐标, 这里负责组装最终结果字典。
    """
    x_coords, y_coords, elapsed = solve(problem, time_limit=time_limit,
                                        strategies=strategies)
    _, hpwl, area, overlap = compute_cost(problem, x_coords, y_coords)
    real_cost = 10 * hpwl + area

    box_position = [[round(x_coords[i], 4), round(y_coords[i], 4)]
                    for i in range(problem.n)]

    return {
        "box_position": box_position,
        "cost": round(real_cost, 4),
        "hpwl": round(hpwl, 4),
        "area": round(area, 4),
        "overlap": round(overlap, 4),
        "elapsed_seconds": round(elapsed, 2),
    }


# ============================================================
# 约束验证器 (从 route.py 移出)
# ============================================================

def validate_layout(problem: Problem, x: List[float], y: List[float],
                    eps: float = 1e-3) -> List[str]:
    """验证布局,返回违反约束列表"""
    violations = []
    n = problem.n

    # 不重叠
    for i in range(n):
        for j in range(i + 1, n):
            ox = min(x[i] + problem.widths[i], x[j] + problem.widths[j]) - max(x[i], x[j])
            oy = min(y[i] + problem.heights[i], y[j] + problem.heights[j]) - max(y[i], y[j])
            if ox > eps and oy > eps:
                violations.append(f"重叠: box {i+1} 和 box {j+1}")

    # X轴对称
    for group in problem.sym_x_groups:
        axis_vals = []
        for pair in group.get("symmetry_pair", []):
            i, j = pair[0] - 1, pair[1] - 1
            axis_vals.append((x[i] + problem.widths[i]/2 + x[j] + problem.widths[j]/2) / 2)
        for s in group.get("self_symmetry", []):
            axis_vals.append(x[s-1] + problem.widths[s-1] / 2)
        if axis_vals:
            axis = sum(axis_vals) / len(axis_vals)
            for pair in group.get("symmetry_pair", []):
                i, j = pair[0] - 1, pair[1] - 1
                mid = (x[i] + problem.widths[i]/2 + x[j] + problem.widths[j]/2) / 2
                if abs(mid - axis) > eps:
                    violations.append(f"对称x: box {i+1} 和 {j+1}")
            for s in group.get("self_symmetry", []):
                s -= 1
                if abs(x[s] + problem.widths[s]/2 - axis) > eps:
                    violations.append(f"自对称x: box {s+1}")

    # Y轴对称
    for group in problem.sym_y_groups:
        axis_vals = []
        for pair in group.get("symmetry_pair", []):
            i, j = pair[0] - 1, pair[1] - 1
            axis_vals.append((y[i] + problem.heights[i]/2 + y[j] + problem.heights[j]/2) / 2)
        for s in group.get("self_symmetry", []):
            axis_vals.append(y[s-1] + problem.heights[s-1] / 2)
        if axis_vals:
            axis = sum(axis_vals) / len(axis_vals)
            for pair in group.get("symmetry_pair", []):
                i, j = pair[0] - 1, pair[1] - 1
                mid = (y[i] + problem.heights[i]/2 + y[j] + problem.heights[j]/2) / 2
                if abs(mid - axis) > eps:
                    violations.append(f"对称y: box {i+1} 和 {j+1}")
            for s in group.get("self_symmetry", []):
                s -= 1
                if abs(y[s] + problem.heights[s]/2 - axis) > eps:
                    violations.append(f"自对称y: box {s+1}")

    # 对齐
    for group in problem.align_left:
        boxes = [b - 1 for b in group]
        vals = [x[b] for b in boxes]
        if max(vals) - min(vals) > eps:
            violations.append(f"对齐left: boxes {group}")
    for group in problem.align_right:
        boxes = [b - 1 for b in group]
        vals = [x[b] + problem.widths[b] for b in boxes]
        if max(vals) - min(vals) > eps:
            violations.append(f"对齐right: boxes {group}")
    for group in problem.align_top:
        boxes = [b - 1 for b in group]
        vals = [y[b] + problem.heights[b] for b in boxes]
        if max(vals) - min(vals) > eps:
            violations.append(f"对齐top: boxes {group}")
    for group in problem.align_bottom:
        boxes = [b - 1 for b in group]
        vals = [y[b] for b in boxes]
        if max(vals) - min(vals) > eps:
            violations.append(f"对齐bottom: boxes {group}")

    # 重复组：检查尺寸一致性和位置偏移一致性
    for rg in problem.repeat_groups:
        groups = [[b - 1 for b in grp] for grp in rg]
        if len(groups) < 2:
            continue
        ref = groups[0]
        
        # 检查对应位置的 box 尺寸是否相同
        for ig in range(1, len(groups)):
            grp = groups[ig]
            for j, (ref_box, box) in enumerate(zip(ref, grp)):
                ref_w = problem.widths[ref_box]
                ref_h = problem.heights[ref_box]
                box_w = problem.widths[box]
                box_h = problem.heights[box]
                if abs(ref_w - box_w) > eps or abs(ref_h - box_h) > eps:
                    violations.append(
                        f"重复组尺寸矛盾: box {ref_box+1} [{ref_w}x{ref_h}] "
                        f"与 box {box+1} [{box_w}x{box_h}] 尺寸不同"
                    )
        
        # 检查位置偏移一致性
        ref_offsets = [(x[b] - x[ref[0]], y[b] - y[ref[0]]) for b in ref]
        for ig in range(1, len(groups)):
            grp = groups[ig]
            grp_offsets = [(x[b] - x[grp[0]], y[b] - y[grp[0]]) for b in grp]
            for j in range(len(ref)):
                dx_ref, dy_ref = ref_offsets[j]
                dx_grp, dy_grp = grp_offsets[j]
                if abs(dx_ref - dx_grp) > eps or abs(dy_ref - dy_grp) > eps:
                    violations.append(f"重复组位置偏移不一致: box {grp[j]+1}")

    return violations


def _verify_x_symmetry_constraint(data: Dict[str, Any], x: List[float], y: List[float],
                                   widths: List[float], eps: float) -> Tuple[List[str], List[str]]:
    """验证 X 轴对称约束，返回 (violations, details)"""
    violations = []
    details = []
    
    for gi, group in enumerate(data.get("symmetry_x", [])):
        axis_vals = []
        for pair in group.get("symmetry_pair", []):
            i, j = pair[0] - 1, pair[1] - 1
            mid = (x[i] + widths[i] / 2 + x[j] + widths[j] / 2) / 2
            axis_vals.append(mid)
        for s in group.get("self_symmetry", []):
            axis_vals.append(x[s - 1] + widths[s - 1] / 2)

        if axis_vals:
            axis = sum(axis_vals) / len(axis_vals)
            group_ok = True
            for pair in group.get("symmetry_pair", []):
                i, j = pair[0] - 1, pair[1] - 1
                mid = (x[i] + widths[i] / 2 + x[j] + widths[j] / 2) / 2
                if abs(mid - axis) > eps:
                    v = f"  X对称: box {pair[0]} & {pair[1]}, mid={mid:.3f}, axis={axis:.3f}, diff={abs(mid-axis):.6f}"
                    violations.append(v)
                    details.append(v)
                    group_ok = False
                if abs(y[i] - y[j]) > eps:
                    v = f"  X对称 y不等: box {pair[0]} y={y[i]:.3f}, box {pair[1]} y={y[j]:.3f}"
                    violations.append(v)
                    details.append(v)
                    group_ok = False
            for s in group.get("self_symmetry", []):
                si = s - 1
                if abs(x[si] + widths[si] / 2 - axis) > eps:
                    v = f"  X自对称: box {s}, center={x[si]+widths[si]/2:.3f}, axis={axis:.3f}"
                    violations.append(v)
                    details.append(v)
                    group_ok = False
            if group_ok:
                details.append(f"  X对称组{gi}: ✓ (axis={axis:.3f})")
    
    return violations, details


def _verify_y_symmetry_constraint(data: Dict[str, Any], x: List[float], y: List[float],
                                   heights: List[float], eps: float) -> Tuple[List[str], List[str]]:
    """验证 Y 轴对称约束，返回 (violations, details)"""
    violations = []
    details = []
    
    for gi, group in enumerate(data.get("symmetry_y", [])):
        axis_vals = []
        for pair in group.get("symmetry_pair", []):
            i, j = pair[0] - 1, pair[1] - 1
            mid = (y[i] + heights[i] / 2 + y[j] + heights[j] / 2) / 2
            axis_vals.append(mid)
        for s in group.get("self_symmetry", []):
            axis_vals.append(y[s - 1] + heights[s - 1] / 2)

        if axis_vals:
            axis = sum(axis_vals) / len(axis_vals)
            group_ok = True
            for pair in group.get("symmetry_pair", []):
                i, j = pair[0] - 1, pair[1] - 1
                mid = (y[i] + heights[i] / 2 + y[j] + heights[j] / 2) / 2
                if abs(mid - axis) > eps:
                    v = f"  Y对称: box {pair[0]} & {pair[1]}, mid={mid:.3f}, axis={axis:.3f}, diff={abs(mid-axis):.6f}"
                    violations.append(v)
                    details.append(v)
                    group_ok = False
                if abs(x[i] - x[j]) > eps:
                    v = f"  Y对称 x不等: box {pair[0]} x={x[i]:.3f}, box {pair[1]} x={x[j]:.3f}"
                    violations.append(v)
                    details.append(v)
                    group_ok = False
            for s in group.get("self_symmetry", []):
                si = s - 1
                if abs(y[si] + heights[si] / 2 - axis) > eps:
                    v = f"  Y自对称: box {s}, center={y[si]+heights[si]/2:.3f}, axis={axis:.3f}"
                    violations.append(v)
                    details.append(v)
                    group_ok = False
            if group_ok:
                details.append(f"  Y对称组{gi}: ✓ (axis={axis:.3f})")
    
    return violations, details


def _verify_alignment_constraint(data: Dict[str, Any], x: List[float], y: List[float],
                                  widths: List[float], heights: List[float], eps: float) -> Tuple[List[str], List[str]]:
    """验证对齐约束，返回 (violations, details)"""
    violations = []
    details = []
    
    align = data.get("align", {})
    for direction, groups in [("left", align.get("left", [])),
                               ("right", align.get("right", [])),
                               ("top", align.get("top", [])),
                               ("bottom", align.get("bottom", []))]:
        for gi, grp in enumerate(groups):
            if not grp:
                continue
            boxes = [b - 1 for b in grp]
            if direction == "left":
                vals = [x[b] for b in boxes]
            elif direction == "right":
                vals = [x[b] + widths[b] for b in boxes]
            elif direction == "top":
                vals = [y[b] + heights[b] for b in boxes]
            elif direction == "bottom":
                vals = [y[b] for b in boxes]

            spread = max(vals) - min(vals)
            if spread > eps:
                v = f"  {direction}对齐: boxes {grp}, spread={spread:.6f}"
                violations.append(v)
                details.append(v)
            else:
                details.append(f"  {direction}对齐 boxes{grp}: ✓ (val={vals[0]:.3f})")
    
    return violations, details


def _verify_repeat_group_constraint(data: Dict[str, Any], x: List[float], y: List[float],
                                     widths: List[float], heights: List[float], eps: float) -> Tuple[List[str], List[str]]:
    """验证重复组约束，返回 (violations, details)"""
    violations = []
    details = []
    
    for rgi, rg in enumerate(data.get("repeat_groups", [])):
        groups = [[b - 1 for b in grp] for grp in rg.get("groups", [])]
        if len(groups) < 2:
            continue
        ref = groups[0]
        rg_ok = True
        
        # 检查对应位置的 box 尺寸是否相同
        for ig in range(1, len(groups)):
            grp = groups[ig]
            for j, (ref_box, box) in enumerate(zip(ref, grp)):
                ref_w = widths[ref_box]
                ref_h = heights[ref_box]
                box_w = widths[box]
                box_h = heights[box]
                if abs(ref_w - box_w) > eps or abs(ref_h - box_h) > eps:
                    v = (f"  重复组{rgi}尺寸矛盾: box {ref_box+1} [{ref_w}x{ref_h}] "
                         f"与 box {box+1} [{box_w}x{box_h}] 尺寸不同")
                    violations.append(v)
                    details.append(v)
                    rg_ok = False
        
        # 检查位置偏移一致性
        ref_offsets = [(x[b] - x[ref[0]], y[b] - y[ref[0]]) for b in ref]
        for ig in range(1, len(groups)):
            grp = groups[ig]
            grp_offsets = [(x[b] - x[grp[0]], y[b] - y[grp[0]]) for b in grp]
            for j in range(len(ref)):
                dx_ref, dy_ref = ref_offsets[j]
                dx_grp, dy_grp = grp_offsets[j]
                if abs(dx_ref - dx_grp) > eps or abs(dy_ref - dy_grp) > eps:
                    v = (f"  重复组{rgi}: box {grp[j]+1} 偏移不一致 "
                         f"ref=({dx_ref:.3f},{dy_ref:.3f}) "
                         f"grp=({dx_grp:.3f},{dy_grp:.3f})")
                    violations.append(v)
                    details.append(v)
                    rg_ok = False
        if rg_ok:
            details.append(f"  重复组{rgi}: ✓")
    
    return violations, details


def _verify_no_overlap_constraint(n: int, x: List[float], y: List[float],
                                   widths: List[float], heights: List[float], eps: float) -> Tuple[List[str], List[str]]:
    """验证无重叠约束，返回 (violations, details)"""
    violations = []
    details = []
    
    overlap_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            ox = min(x[i] + widths[i], x[j] + widths[j]) - max(x[i], x[j])
            oy = min(y[i] + heights[i], y[j] + heights[j]) - max(y[i], y[j])
            if ox > eps and oy > eps:
                v = f"  重叠: box {i+1} & {j+1}, overlap=({ox:.3f}x{oy:.3f}={ox*oy:.3f})"
                violations.append(v)
                details.append(v)
                overlap_count += 1
    if overlap_count == 0:
        details.append("  无重叠: ✓")
    
    return violations, details


def verify_all_constraints(data: Dict[str, Any], positions: List[List[float]],
                           label: str = "", eps: float = 1e-3) -> Dict[str, Any]:
    """
    全面验证约束，返回结构化结果。

    检查:
      - X轴对称 (symmetry_pair + self_symmetry)
      - Y轴对称 (symmetry_pair + self_symmetry)
      - 对齐 (left, right, top, bottom)
      - 重复组
      - 无重叠

    返回: {
        "pass": bool,
        "violations": [...],
        "details": { "sym_x": ..., "sym_y": ..., "align": ..., "repeat": ..., "overlap": ... },
        "cost_info": { "hpwl": ..., "area": ..., "cost": ... }
    }
    """
    n = len(data["box_size"])
    widths = [s[0] for s in data["box_size"]]
    heights = [s[1] for s in data["box_size"]]
    x = [p[0] for p in positions]
    y = [p[1] for p in positions]

    violations = []
    details = {
        "sym_x": [],
        "sym_y": [],
        "align": [],
        "repeat": [],
        "overlap": []
    }

    # 1. X轴对称
    v, d = _verify_x_symmetry_constraint(data, x, y, widths, eps)
    violations.extend(v)
    details["sym_x"] = d

    # 2. Y轴对称
    v, d = _verify_y_symmetry_constraint(data, x, y, heights, eps)
    violations.extend(v)
    details["sym_y"] = d

    # 3. 对齐
    v, d = _verify_alignment_constraint(data, x, y, widths, heights, eps)
    violations.extend(v)
    details["align"] = d

    # 4. 重复组
    v, d = _verify_repeat_group_constraint(data, x, y, widths, heights, eps)
    violations.extend(v)
    details["repeat"] = d

    # 5. 无重叠
    v, d = _verify_no_overlap_constraint(n, x, y, widths, heights, eps)
    violations.extend(v)
    details["overlap"] = d

    # Cost 计算
    cost_info = {}
    try:
        problem = Problem(data)
        _, hpwl, area, overlap = compute_cost(problem, x, y)
        cost_info = {"hpwl": hpwl, "area": area, "overlap": overlap, "cost": 10 * hpwl + area}
    except Exception as e:
        cost_info = {"error": str(e)}

    return {
        "pass": len(violations) == 0,
        "violations": violations,
        "details": details,
        "cost_info": cost_info
    }


# ============================================================
# 测试用例发现
# ============================================================

TEST_CASES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_cases")

OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def discover_cases(filter_name: str = None) -> List[str]:
    """发现所有测试用例文件，返回排序后的路径列表"""
    pattern = os.path.join(TEST_CASES_DIR, "case*.json")
    files = sorted(glob.glob(pattern))
    if filter_name:
        files = [f for f in files if filter_name in os.path.basename(f)]
    return files


def load_case(filepath: str) -> Dict[str, Any]:
    """加载测试用例"""
    with open(filepath, "r") as f:
        return json.load(f)


# ============================================================
# 内置测试用例
# ============================================================

def _builtin_test_case() -> dict:
    return {
        "box_size": [
            [6, 4], [3, 5], [4, 2], [4, 2], [8, 6], [8, 6],
            [6, 6], [6, 6], [4, 2], [4, 2], [5, 3], [8, 4],
        ],
        "symmetry_x": [
            {"symmetry_pair": [[5, 6], [3, 4]], "self_symmetry": [1]}
        ],
        "symmetry_y": [
            {"symmetry_pair": [[7, 8]], "self_symmetry": [2]}
        ],
        "align": {
            "left": [[1, 3]],
            "right": [],
            "top": [],
            "bottom": [[1, 7, 2]]
        },
        "repeat_groups": [],
        "nets": [[1, 2, 3, 4, 5], [2, 6, 8], [10, 12]]
    }


# ============================================================
# 运行模式
# ============================================================

def run_single_case(data: Dict[str, Any], time_limit: float, case_name: str,
                    strategies: List[str] = None,
                    validate: bool = False) -> Dict[str, Any]:
    """
    运行单个用例。

    Args:
        data: 输入数据 (纯 input 格式或含 input/expected_output 的测试用例格式)
        time_limit: 时间限制
        case_name: 用例名称
        strategies: 初始化策略
        validate: 是否做约束验证

    Returns:
        结果字典
    """
    # 兼容两种格式: 纯 input 或 {input: ..., expected_output: ...}
    if "input" in data and "box_size" in data.get("input", {}):
        input_data = data["input"]
    else:
        input_data = data

    print(f"输入: {len(input_data['box_size'])} 个矩形, "
          f"{len(input_data.get('nets', []))} 个网络")
    print(f"时间限制: {time_limit}s")
    if strategies:
        print(f"初始化策略: {', '.join(strategies)}")
    print()

    problem = Problem(input_data)
    result = solve_and_report(problem, time_limit=time_limit, strategies=strategies)

    print(f"\n=== 结果 ===")
    print(f"Cost:    {result['cost']}")
    print(f"  HPWL:  {result['hpwl']}")
    print(f"  Area:  {result['area']}")
    print(f"  Overlap: {result['overlap']}")
    print(f"耗时:    {result['elapsed_seconds']}s")

    # 验证
    if validate:
        x = [p[0] for p in result["box_position"]]
        y = [p[1] for p in result["box_position"]]
        violations = validate_layout(problem, x, y)
        if not violations:
            print("✅ 所有约束满足")
        else:
            print(f"⚠️ 约束违反 ({len(violations)} 条):")
            for v in violations:
                print(f"   {v}")

    # 保存输出
    output = {"box_position": result["box_position"]}
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"output_{case_name.replace('.json', '')}.json"
    )
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n输出已保存到 {output_path}")

    return result


def run_case_test(filepath: str, time_limit: float = 115.0,
                  expected_only: bool = False,
                  visualize: bool = False,
                  viz_dir: str = 'visualizations') -> Dict[str, Any]:
    """
    对单个测试用例做完整验证:
      1. 验证 expected_output 约束
      2. 运行算法，验证算法输出约束
      3. 对比两者 cost
      4. 可视化（可选）
    """
    case_name = os.path.basename(filepath).replace(".json", "")
    data = load_case(filepath)
    input_data = data.get("input", data)
    desc = data.get("description", case_name)

    result = {
        "case": case_name,
        "description": desc,
        "n_boxes": len(input_data["box_size"]),
    }

    # --- 1. 验证 expected_output ---
    expected = data.get("expected_output")
    if expected and expected.get("box_position"):
        exp_result = verify_all_constraints(
            input_data, expected["box_position"], label="expected"
        )
        result["expected_valid"] = exp_result["pass"]
        result["expected_violations"] = exp_result["violations"]
        result["expected_cost"] = exp_result["cost_info"]
    else:
        result["expected_valid"] = None

    if expected_only:
        # 可视化（仅预期）
        if visualize:
            viz_path = os.path.join(viz_dir, f"{case_name}.png")
            visualize_case(data, case_name=case_name, save_path=viz_path)
        return result

    # --- 2. 验证输入合法性 ---
    try:
        validate_input(input_data)
    except ValueError as e:
        # 无解用例（如重复组尺寸矛盾）
        result["unsolvable"] = True
        result["unsolvable_reason"] = str(e)
        # 无解用例也可视化输入约束
        if visualize:
            viz_path = os.path.join(viz_dir, f"{case_name}_unsolvable.png")
            visualize_case(data, case_name=case_name, save_path=viz_path)
        return result

    # --- 3. 运行算法 ---
    problem = Problem(input_data)
    solve_result = solve_and_report(problem, time_limit=time_limit)
    elapsed = solve_result["elapsed_seconds"]

    algo_positions = solve_result["box_position"]
    algo_result = verify_all_constraints(
        input_data, algo_positions, label="algo"
    )

    result["algo_valid"] = algo_result["pass"]
    result["algo_violations"] = algo_result["violations"]
    result["algo_cost"] = algo_result["cost_info"]
    result["algo_overlap"] = solve_result.get("overlap", 0)
    result["elapsed"] = elapsed

    # --- 4. 对比 ---
    if result.get("expected_cost") and result["algo_cost"]:
        exp_c = result["expected_cost"].get("cost", float("inf"))
        algo_c = result["algo_cost"].get("cost", float("inf"))
        result["cost_ratio"] = algo_c / exp_c if exp_c > 0 else float("inf")

    # --- 5. 保存算法输出到 outputs/ ---
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    algo_output_path = os.path.join(OUTPUTS_DIR, f"{case_name}.json")
    with open(algo_output_path, 'w') as f:
        json.dump(solve_result, f, indent=2, ensure_ascii=False)

    # --- 6. 可视化 ---
    if visualize:
        viz_path = os.path.join(viz_dir, f"{case_name}.png")
        visualize_case(data, algo_result=solve_result,
                       case_name=case_name, save_path=viz_path)

    return result


def _run_case_test_worker(filepath: str, time_limit: float,
                          expected_only: bool,
                          visualize: bool = False,
                          viz_dir: str = 'visualizations') -> Dict[str, Any]:
    """
    子进程工作函数: 静默运行单个用例测试，返回结果。
    抑制 SA 的 print 输出，避免多进程输出混乱。
    """
    import io
    from contextlib import redirect_stdout

    # 捕获 SA 的维测打印，不输出到终端
    f = io.StringIO()
    try:
        with redirect_stdout(f):
            result = run_case_test(filepath, time_limit=time_limit,
                                   expected_only=expected_only,
                                   visualize=visualize,
                                   viz_dir=viz_dir)
        result["_sa_output"] = f.getvalue()
    except Exception as e:
        result = {
            "case": os.path.basename(filepath).replace(".json", ""),
            "error": str(e),
        }
    return result


def run_cases_parallel(filepaths: List[str], time_limit: float = 115.0,
                       expected_only: bool = False,
                       visualize: bool = False,
                       viz_dir: str = 'visualizations',
                       max_workers: int = None) -> List[Dict[str, Any]]:
    """
    并行运行多个测试用例。

    Args:
        filepaths: 测试用例文件路径列表
        time_limit: 每个用例的时间限制
        expected_only: 是否只验证预期输出
        max_workers: 最大并行进程数，默认 CPU 核心数

    Returns:
        结果列表，顺序与 filepaths 对应
    """
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), len(filepaths))

    total = len(filepaths)
    results = [None] * total
    case_names = [os.path.basename(f).replace(".json", "") for f in filepaths]

    print(f"🚀 并行模式: {total} 个用例, {max_workers} 进程")
    print(f"   用例: {', '.join(case_names)}")
    print()

    t0 = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {}
        for idx, filepath in enumerate(filepaths):
            future = executor.submit(_run_case_test_worker, filepath,
                                     time_limit, expected_only,
                                     visualize, viz_dir)
            future_to_idx[future] = idx

        completed = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                results[idx] = result
                completed += 1

                # 进度报告
                name = case_names[idx]
                if result.get("unsolvable"):
                    status = "⚠️无解"
                    elapsed = 0
                    cost = ""
                    ratio = ""
                elif "error" in result:
                    status = "💥"
                    elapsed = 0
                    cost = ""
                    ratio = ""
                else:
                    status = "✅" if result.get("algo_valid", result.get("expected_valid")) else "❌"
                    elapsed = result.get("elapsed", 0)
                    cost = ""
                    if result.get("algo_cost") and "cost" in result["algo_cost"]:
                        cost = f", cost={result['algo_cost']['cost']:.0f}"
                    ratio = ""
                    if result.get("cost_ratio"):
                        ratio = f", ratio={result['cost_ratio']:.3f}"

                wall = time.time() - t0
                print(f"  [{completed}/{total}] {status} {name} "
                      f"({elapsed:.1f}s{cost}{ratio}) "
                      f"[wall {wall:.1f}s]")

            except Exception as e:
                results[idx] = {
                    "case": case_names[idx],
                    "error": str(e),
                }
                completed += 1
                print(f"  [{completed}/{total}] 💥 {case_names[idx]}: {e}")

    wall = time.time() - t0
    print(f"\n⏱️  总耗时: {wall:.1f}s (wall clock)")

    return results


# ============================================================
# 报告输出
# ============================================================

def print_case_result(r: Dict[str, Any], verbose: bool = False):
    """打印单个测试用例的结果"""
    case = r["case"]
    desc = r.get("description", "")
    n = r.get("n_boxes", "?")

    print(f"\n{'='*70}")
    print(f"📦 {case} | {desc}")
    print(f"   矩形数: {n}")
    print(f"{'─'*70}")

    # Unsolvable
    if r.get("unsolvable"):
        print(f"   ⚠️  无解用例")
        print(f"   原因: {r.get('unsolvable_reason', '未知')}")
        return

    # Expected
    ev = r.get("expected_valid")
    if ev is not None:
        status = "✅ PASS" if ev else "❌ FAIL"
        print(f"   预期输出约束验证: {status}")
        if not ev and r.get("expected_violations"):
            for v in r["expected_violations"]:
                print(f"      {v}")
        ec = r.get("expected_cost", {})
        if ec and "cost" in ec:
            print(f"   预期 Cost: {ec['cost']:.1f} (HPWL={ec['hpwl']:.1f}, Area={ec['area']:.1f})")
    else:
        print(f"   预期输出: (无)")

    # Algo
    if "algo_valid" in r:
        av = r["algo_valid"]
        status = "✅ PASS" if av else "❌ FAIL"
        print(f"   算法输出约束验证: {status}")
        if not av and r.get("algo_violations"):
            for v in r["algo_violations"]:
                print(f"      {v}")
        ac = r.get("algo_cost", {})
        if ac and "cost" in ac:
            print(f"   算法 Cost: {ac['cost']:.1f} (HPWL={ac['hpwl']:.1f}, Area={ac['area']:.1f}, Overlap={r.get('algo_overlap', 0):.1f})")
        print(f"   耗时: {r.get('elapsed', 0):.1f}s")

        if "cost_ratio" in r:
            ratio = r["cost_ratio"]
            tag = "👍" if ratio <= 1.1 else ("⚠️" if ratio <= 1.5 else "📈")
            print(f"   Cost 比值: {ratio:.3f} {tag}")

    print(f"{'='*70}")


def print_summary(results: List[Dict[str, Any]]):
    """打印批量测试汇总"""
    print(f"\n\n{'='*70}")
    print("📊 测试汇总")
    print(f"{'='*70}")

    total = len(results)
    exp_pass = sum(1 for r in results if r.get("expected_valid") is True)
    exp_fail = sum(1 for r in results if r.get("expected_valid") is False)
    algo_pass = sum(1 for r in results if r.get("algo_valid") is True)
    algo_fail = sum(1 for r in results if r.get("algo_valid") is False)
    algo_run = sum(1 for r in results if "algo_valid" in r)

    print(f"   用例总数: {total}")
    print(f"   预期输出: {exp_pass} pass / {exp_fail} fail / {total - exp_pass - exp_fail} N/A")
    if algo_run:
        print(f"   算法输出: {algo_pass} pass / {algo_fail} fail")

    # 表格
    if algo_run:
        print(f"\n   {'用例':<45} {'预期':>6} {'算法':>6} {'比值':>8} {'耗时':>8}")
        print(f"   {'─'*45} {'─'*6} {'─'*6} {'─'*8} {'─'*8}")
        for r in results:
            if r.get("unsolvable"):
                print(f"   {r['case']:<45} {'⚠️无解':>6} {'⚠️无解':>6} {'–':>8} {'–':>8}")
            else:
                e = "✅" if r.get("expected_valid") else ("❌" if r.get("expected_valid") is False else "–")
                a = "✅" if r.get("algo_valid") else ("❌" if r.get("algo_valid") is False else "–")
                ratio = f"{r['cost_ratio']:.3f}" if "cost_ratio" in r else "–"
                elapsed = f"{r.get('elapsed', 0):.1f}s" if "elapsed" in r else "–"
                print(f"   {r['case']:<45} {e:>6} {a:>6} {ratio:>8} {elapsed:>8}")

    print(f"{'='*70}")


def print_run_summary(results: List[tuple]):
    """打印批量运行汇总（非验证模式）"""
    print(f"\n\n{'='*60}")
    print("=== 批量执行汇总 ===")
    print('='*60)
    print(f"{'用例':<30} {'Cost':>12} {'HPWL':>10} {'Area':>12} {'Overlap':>10} {'耗时':>8}")
    print('-'*60)

    success_count = 0
    total_cost = 0
    total_time = 0

    for case_name, result in results:
        if result is None:
            print(f"{case_name:<30} {'FAILED':>12}")
            continue

        success_count += 1
        total_cost += result['cost']
        total_time += result['elapsed_seconds']

        print(f"{case_name:<30} {result['cost']:>12.1f} "
              f"{result['hpwl']:>10.1f} {result['area']:>12.1f} "
              f"{result['overlap']:>10.1f} {result['elapsed_seconds']:>7.1f}s")

    print('-'*60)
    if success_count > 0:
        print(f"成功: {success_count}/{len(results)}")
        print(f"平均 Cost: {total_cost/success_count:.1f}")
        print(f"总耗时: {total_time:.1f}s")
    print('='*60)


# ============================================================
# 测试套件执行器
# ============================================================

def _run_test_suite(cases: List[str], time_limit: float, expected_only: bool,
                    verbose: bool, parallel: bool, max_workers: int = None,
                    visualize: bool = False, viz_dir: str = 'visualizations'):
    """统一测试套件执行入口，支持串行/并行。"""
    print(f"🦐 代码虾测试运行器")
    print(f"   找到 {len(cases)} 个用例")
    print(f"   时间限制: {time_limit}s/用例")
    if expected_only:
        print(f"   模式: 仅验证预期输出")
    if visualize:
        print(f"   可视化: 输出到 {viz_dir}/")
    if parallel:
        workers = max_workers or min(multiprocessing.cpu_count(), len(cases))
        print(f"   并行: {workers} 进程")
    print()

    if parallel and not expected_only and len(cases) > 1:
        # 并行执行
        results = run_cases_parallel(cases, time_limit=time_limit,
                                     expected_only=expected_only,
                                     visualize=visualize,
                                     viz_dir=viz_dir,
                                     max_workers=max_workers)
        # 详细报告
        if verbose:
            for r in results:
                if r:
                    print_case_result(r, verbose=verbose)
    else:
        # 串行执行
        results = []
        for filepath in cases:
            print(f"▶ 正在处理: {os.path.basename(filepath)}...")
            try:
                r = run_case_test(filepath, time_limit=time_limit,
                                  expected_only=expected_only,
                                  visualize=visualize,
                                  viz_dir=viz_dir)
                results.append(r)
                print_case_result(r, verbose=verbose)
            except Exception as e:
                print(f"❌ 执行失败: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    "case": os.path.basename(filepath).replace(".json", ""),
                    "error": str(e)
                })

    print_summary(results)

    any_fail = any(
        r.get("expected_valid") is False or r.get("algo_valid") is False
        for r in results if r
    )
    sys.exit(1 if any_fail else 0)


# ============================================================
# Main
# ============================================================

def _parse_arguments() -> Dict[str, Any]:
    """Parse command line arguments"""
    config = {
        'time_limit': 115.0,
        'strategies': None,
        'validate': False,
        'expected_only': False,
        'verbose': False,
        'visualize': True,
        'viz_dir': 'visualizations',
        'viz_from': None,
        'list_cases': False,
        'directory': None,
        'parallel': False,
        'max_workers': None,
        'input_files': [],
        'args': []
    }
    
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--time" and i + 1 < len(sys.argv):
            config['time_limit'] = float(sys.argv[i + 1])
            i += 2
        elif arg == "--strategies" and i + 1 < len(sys.argv):
            config['strategies'] = [s.strip() for s in sys.argv[i + 1].split(',')]
            i += 2
        elif arg == "--validate":
            config['validate'] = True
            i += 1
        elif arg == "--expected-only":
            config['validate'] = True
            config['expected_only'] = True
            i += 1
        elif arg == "--verbose" or arg == "-v":
            config['verbose'] = True
            i += 1
        elif arg == "--visualize" or arg == "-V":
            config['visualize'] = True
            i += 1
        elif arg == "--no-visualize":
            config['visualize'] = False
            i += 1
        elif arg == "--viz-dir" and i + 1 < len(sys.argv):
            config['viz_dir'] = sys.argv[i + 1]
            i += 2
        elif arg == "--viz" and i + 1 < len(sys.argv):
            config['viz_from'] = sys.argv[i + 1]
            i += 2
        elif arg == "--parallel" or arg == "-j":
            config['parallel'] = True
            if i + 1 < len(sys.argv) and sys.argv[i + 1].isdigit():
                config['max_workers'] = int(sys.argv[i + 1])
                i += 2
            else:
                i += 1
        elif arg == "--list-strategies":
            print("可用的初始化策略:")
            for strategy in SimulatedAnnealing.get_available_strategies():
                print(f"  - {strategy}")
            sys.exit(0)
        elif arg == "--list-cases":
            config['list_cases'] = True
            i += 1
        elif arg == "-d" and i + 1 < len(sys.argv):
            config['directory'] = sys.argv[i + 1]
            i += 2
        elif arg == "-t":
            config['args'].append("-t")
            i += 1
        else:
            config['args'].append(arg)
            i += 1
    
    return config


def _handle_list_cases():
    """List available test cases"""
    cases = discover_cases()
    if not cases:
        print(f"未找到测试用例 (目录: {TEST_CASES_DIR})")
        return
    print("可用测试用例:")
    for f in cases:
        data = load_case(f)
        name = os.path.basename(f).replace(".json", "")
        desc = data.get("description", "")
        print(f"  {name}: {desc}")


def _handle_directory_mode(config: Dict[str, Any]):
    """Handle directory-based execution mode"""
    directory = config['directory']
    use_test_mode = config['validate'] or (directory and "test_cases" in directory)
    
    if use_test_mode:
        # Test mode: case*.json files
        cases = sorted(glob.glob(os.path.join(directory, "case*.json")))
    else:
        # Run mode: all JSON files
        cases = sorted(glob.glob(os.path.join(directory, "*.json")))
    
    if not cases:
        print(f"错误: 目录 {directory} 中没有找到 JSON 文件")
        sys.exit(1)
    
    # Filter by names (support multiple names)
    positional = [a for a in config['args'] if not a.startswith("-")]
    if positional:
        cases = [f for f in cases
                 if any(name in os.path.basename(f) for name in positional)]
    
    if not cases:
        print(f"未找到匹配的用例 (filter={positional})")
        sys.exit(1)
    
    if use_test_mode:
        # Test mode
        _run_test_suite(cases, config['time_limit'], config['expected_only'],
                        config['verbose'], config['parallel'], config['max_workers'],
                        config['visualize'], config['viz_dir'])
    else:
        # Run mode
        return cases
    
    return None


def _handle_file_mode(config: Dict[str, Any]):
    """Handle file-based execution mode"""
    positional = [a for a in config['args'] if not a.startswith("-")]
    first = positional[0]
    
    if os.path.isfile(first):
        return positional
    else:
        # Treat as test_cases name filters
        cases = discover_cases()
        if cases:
            # Filter by any of the provided names
            filtered = [f for f in cases
                       if any(name in os.path.basename(f) for name in positional)]
            if filtered:
                return filtered
        print(f"错误: 没有匹配的用例 (filter={positional})")
        sys.exit(1)


def _print_usage():
    """Print usage information"""
    print("Usage:")
    print("  python main.py input.json [--time 115] [--strategies s1,s2]")
    print("  python main.py case1.json case2.json [--time 115]")
    print("  python main.py -d ./test_cases --validate [--time 30] [-j 4]")
    print("  python main.py -d ./test_cases --expected-only")
    print("  python main.py --validate case01 [--time 30]")
    print("  python main.py -t [--time 30]")
    print("  python main.py --list-cases")
    print("  python main.py --list-strategies")
    print("\nParallel mode:")
    print("  python main.py -d ./test_cases --validate --time 30 -j 8  # 8 processes")
    print("  python main.py -d ./test_cases --validate --time 30 -j    # auto (CPU count)")


def _run_batch_cases(input_files: List[str], config: Dict[str, Any]):
    """Run multiple cases in batch mode"""
    print(f"=== 批量执行 {len(input_files)} 个用例 ===")
    print(f"时间限制: {config['time_limit']}s/用例\n")
    
    results = []
    for i, filepath in enumerate(input_files, 1):
        print(f"\n{'='*60}")
        print(f"用例 {i}/{len(input_files)}: {os.path.basename(filepath)}")
        print('='*60)
        
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            result = run_single_case(data, config['time_limit'], 
                                   os.path.basename(filepath),
                                   config['strategies'], config['validate'])
            results.append((os.path.basename(filepath), result))
        except Exception as e:
            print(f"❌ 执行失败: {e}")
            results.append((os.path.basename(filepath), None))
    
    print_run_summary(results)


def main():
    config = _parse_arguments()
    
    # --viz: 从文件生成可视化
    if config['viz_from']:
        input_file = config['viz_from']
        case_name = os.path.basename(input_file).replace(".json", "")
        output_file = os.path.join(OUTPUTS_DIR, f"{case_name}.json")
        if not os.path.exists(output_file):
            output_file = None
        viz_from_files(input_file, output_file, config['viz_dir'])
        return
    
    # List test cases
    if config['list_cases']:
        _handle_list_cases()
        return
    
    # Built-in test case
    if "-t" in config['args']:
        run_single_case(_builtin_test_case(), config['time_limit'], 
                       "builtin_test", config['strategies'], config['validate'])
        return
    
    # Directory mode
    if config['directory']:
        input_files = _handle_directory_mode(config)
        if input_files is None:
            return
    elif config['args'] and not config['args'][0].startswith("-"):
        # File mode
        input_files = _handle_file_mode(config)
    else:
        _print_usage()
        sys.exit(1)
    
    # Test mode (name filter without directory)
    use_test_mode = config['validate'] or (config['directory'] and "test_cases" in config['directory'])
    if use_test_mode and not config['directory']:
        _run_test_suite(input_files, config['time_limit'], config['expected_only'],
                        config['verbose'], config['parallel'], config['max_workers'],
                        config['visualize'], config['viz_dir'])
        return
    
    # Run mode
    if len(input_files) == 1:
        with open(input_files[0], "r") as f:
            data = json.load(f)
        run_single_case(data, config['time_limit'], os.path.basename(input_files[0]),
                       config['strategies'], config['validate'])
    else:
        _run_batch_cases(input_files, config)


if __name__ == "__main__":
    main()
