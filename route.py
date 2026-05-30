"""
route.py - 基于多规则的模拟电路布局算法 (纯算法模块)

核心思路:约束化简 + 模拟退火
1. 将对称/对齐/重复组约束转化为线性方程组,高斯消元降维
2. 在独立变量的低维空间内用 SA 优化 Cost = 10*HPWL + Area
3. 重叠惩罚引导解朝向无重叠区域

本模块只包含求解算法，不包含 I/O、测试、验证功能。
"""
import math
import random
import time
from typing import List, Dict, Tuple, Set, Any

import numpy as np

# SA 优化时每个 box 向外膨胀的边距，迫使解保持间距避免微小重叠
OVERLAP_MARGIN = 0.02


# ============================================================
# 数据模型
# ============================================================

class Problem:
    """解析后的布局问题"""

    def __init__(self, data: Dict[str, Any]):
        self.n = len(data["box_size"])
        self.widths = [s[0] for s in data["box_size"]]
        self.heights = [s[1] for s in data["box_size"]]

        self.sym_x_groups: List[Dict] = data.get("symmetry_x", [])
        self.sym_y_groups: List[Dict] = data.get("symmetry_y", [])

        align = data.get("align", {})
        self.align_left: List[List[int]] = align.get("left", [])
        self.align_right: List[List[int]] = align.get("right", [])
        self.align_top: List[List[int]] = align.get("top", [])
        self.align_bottom: List[List[int]] = align.get("bottom", [])

        self.repeat_groups: List[List[List[int]]] = [
            rg["groups"] for rg in data.get("repeat_groups", [])
        ]

        self.nets: List[List[int]] = data.get("nets", [])

        # numpy 数组版本，用于向量化计算
        self.w_arr = np.array(self.widths, dtype=np.float64)
        self.h_arr = np.array(self.heights, dtype=np.float64)
        self.w_half = self.w_arr / 2.0
        self.h_half = self.h_arr / 2.0


# ============================================================
# 线性表达式 & 约束化简系统
# ============================================================

class LinearExpr:
    """线性表达式: sum(coeffs[k] * var_k) + const"""

    def __init__(self, coeffs: Dict[int, float] = None, const: float = 0.0):
        self.coeffs = coeffs or {}
        self.const = const

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return LinearExpr(self.coeffs.copy(), self.const + other)
        new_coeffs = self.coeffs.copy()
        for k, v in other.coeffs.items():
            new_coeffs[k] = new_coeffs.get(k, 0) + v
        return LinearExpr(new_coeffs, self.const + other.const)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return LinearExpr(self.coeffs.copy(), self.const - other)
        new_coeffs = self.coeffs.copy()
        for k, v in other.coeffs.items():
            new_coeffs[k] = new_coeffs.get(k, 0) - v
        return LinearExpr(new_coeffs, self.const - other.const)

    def __mul__(self, scalar: float):
        return LinearExpr({k: v * scalar for k, v in self.coeffs.items()},
                         self.const * scalar)

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return (self * -1) + other

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def evaluate(self, var_values: Dict[int, float]) -> float:
        result = self.const
        for k, v in self.coeffs.items():
            result += v * var_values[k]
        return result


class ConstraintSystem:
    """
    约束化简系统:将硬约束转化为线性等式,高斯消元降维。

    变量编号:
    - 0..n-1: x_i
    - n..2n-1: y_i
    - 2n+: 辅助变量(对称轴、重复组平移等)
    """

    def __init__(self, problem: Problem):
        self.problem = problem
        self.n = problem.n
        self.var_exprs: List[LinearExpr] = [
            LinearExpr({i: 1.0}, 0.0) for i in range(2 * self.n)
        ]
        self.equations: List[LinearExpr] = []
        self.next_var = 2 * self.n
        self.free_vars: Set[int] = set(range(2 * self.n))

        self._parse_constraints()
        self._solve()

    def _new_var(self) -> int:
        var_id = self.next_var
        self.next_var += 1
        self.var_exprs.append(LinearExpr({var_id: 1.0}, 0.0))
        return var_id

    def _add_equation(self, expr: LinearExpr):
        self.equations.append(expr)

    def _parse_constraints(self):
        self._parse_symmetry()
        self._parse_alignment()
        self._parse_repeat_groups()

    def _parse_symmetry(self):
        p = self.problem
        n = self.n

        for group in p.sym_x_groups:
            axis_var = self._new_var()
            axis_expr = LinearExpr({axis_var: 1.0}, 0.0)
            for pair in group.get("symmetry_pair", []):
                i, j = pair[0] - 1, pair[1] - 1
                cx_i = self.var_exprs[i] + p.widths[i] / 2
                cx_j = self.var_exprs[j] + p.widths[j] / 2
                self._add_equation(cx_i + cx_j - axis_expr * 2)
                # X对称要求 pair 的 y 坐标相同
                self._add_equation(self.var_exprs[n + i] - self.var_exprs[n + j])
            for s in group.get("self_symmetry", []):
                s -= 1
                cx_s = self.var_exprs[s] + p.widths[s] / 2
                self._add_equation(cx_s - axis_expr)

        for group in p.sym_y_groups:
            axis_var = self._new_var()
            axis_expr = LinearExpr({axis_var: 1.0}, 0.0)
            for pair in group.get("symmetry_pair", []):
                i, j = pair[0] - 1, pair[1] - 1
                cy_i = self.var_exprs[n + i] + p.heights[i] / 2
                cy_j = self.var_exprs[n + j] + p.heights[j] / 2
                self._add_equation(cy_i + cy_j - axis_expr * 2)
                # Y对称要求 pair 的 x 坐标相同
                self._add_equation(self.var_exprs[i] - self.var_exprs[j])
            for s in group.get("self_symmetry", []):
                s -= 1
                cy_s = self.var_exprs[n + s] + p.heights[s] / 2
                self._add_equation(cy_s - axis_expr)

    def _parse_alignment(self):
        p = self.problem
        n = self.n

        for group in p.align_left:
            boxes = [b - 1 for b in group]
            for i in range(1, len(boxes)):
                self._add_equation(self.var_exprs[boxes[0]] - self.var_exprs[boxes[i]])

        for group in p.align_right:
            boxes = [b - 1 for b in group]
            for i in range(1, len(boxes)):
                a, b = boxes[0], boxes[i]
                expr_a = self.var_exprs[a] + p.widths[a]
                expr_b = self.var_exprs[b] + p.widths[b]
                self._add_equation(expr_a - expr_b)

        for group in p.align_bottom:
            boxes = [b - 1 for b in group]
            for i in range(1, len(boxes)):
                self._add_equation(
                    self.var_exprs[n + boxes[0]] - self.var_exprs[n + boxes[i]])

        for group in p.align_top:
            boxes = [b - 1 for b in group]
            for i in range(1, len(boxes)):
                a, b = boxes[0], boxes[i]
                expr_a = self.var_exprs[n + a] + p.heights[a]
                expr_b = self.var_exprs[n + b] + p.heights[b]
                self._add_equation(expr_a - expr_b)

    def _parse_repeat_groups(self):
        p = self.problem
        n = self.n

        for rg in p.repeat_groups:
            groups = [[b - 1 for b in grp] for grp in rg]
            if len(groups) < 2:
                continue
            ref_group = groups[0]
            
            # 方向2：如果模板 group 的 box 数过多，强制网格排列
            # 阈值：4 个 box（8 个变量），超过就固定相对位置
            if len(ref_group) > 5:
                # 计算网格布局：按 box 顺序排列成紧凑网格
                num_boxes = len(ref_group)
                cols = int(num_boxes ** 0.5 + 0.5)
                rows = (num_boxes + cols - 1) // cols
                
                # 计算每列最大宽度和每行最大高度
                col_widths = []
                for c in range(cols):
                    indices = [r * cols + c for r in range(rows) if r * cols + c < num_boxes]
                    col_widths.append(max(p.widths[ref_group[i]] for i in indices))
                
                row_heights = []
                for r in range(rows):
                    indices = [r * cols + c for c in range(cols) if r * cols + c < num_boxes]
                    row_heights.append(max(p.heights[ref_group[i]] for i in indices))
                
                # 计算每列 X 偏移和每行 Y 偏移
                col_offsets = [0.0]
                for c in range(1, cols):
                    col_offsets.append(col_offsets[-1] + col_widths[c - 1])
                
                row_offsets = [0.0]
                for r in range(1, rows):
                    row_offsets.append(row_offsets[-1] + row_heights[r - 1])
                
                # 模板 group 的第一个 box 作为锚点 (x0, y0)
                anchor_box = ref_group[0]
                
                # 对模板 group 内的其他 box，添加网格约束
                for i in range(1, len(ref_group)):
                    box = ref_group[i]
                    col = i % cols
                    row = i // cols
                    
                    dx = col_offsets[col]
                    dy = row_offsets[row]
                    
                    # 添加方程：box_x = anchor_x + dx, box_y = anchor_y + dy
                    self._add_equation(
                        self.var_exprs[box] - self.var_exprs[anchor_box] - LinearExpr({}, dx))
                    self._add_equation(
                        self.var_exprs[n + box] - self.var_exprs[n + anchor_box] - LinearExpr({}, dy))
            
            # 其他 group 与模板的偏移（保持不变）
            for g_idx in range(1, len(groups)):
                group = groups[g_idx]
                dx_var = self._new_var()
                dy_var = self._new_var()
                dx_expr = LinearExpr({dx_var: 1.0}, 0.0)
                dy_expr = LinearExpr({dy_var: 1.0}, 0.0)
                for ref_box, box in zip(ref_group, group):
                    self._add_equation(
                        self.var_exprs[box] - self.var_exprs[ref_box] - dx_expr)
                    self._add_equation(
                        self.var_exprs[n + box] - self.var_exprs[n + ref_box] - dy_expr)

    def _solve(self):
        """高斯消元求解线性方程组"""
        if not self.equations:
            return

        num_vars = self.next_var
        num_eqs = len(self.equations)
        A = np.zeros((num_eqs, num_vars + 1))

        for i, eq in enumerate(self.equations):
            for var_id, coeff in eq.coeffs.items():
                A[i, var_id] = coeff
            A[i, num_vars] = -eq.const

        # 高斯消元 (RREF)
        pivot_cols = []
        row = 0
        for col in range(num_vars):
            if row >= num_eqs:
                break
            max_row = row
            for r in range(row + 1, num_eqs):
                if abs(A[r, col]) > abs(A[max_row, col]):
                    max_row = r
            if abs(A[max_row, col]) < 1e-10:
                continue
            A[[row, max_row]] = A[[max_row, row]]
            pivot = A[row, col]
            A[row] /= pivot
            for r in range(num_eqs):
                if r != row and abs(A[r, col]) > 1e-10:
                    A[r] -= A[r, col] * A[row]
            pivot_cols.append(col)
            row += 1

        # 识别自由变量
        pivot_set = set(pivot_cols)
        self.free_vars = set(range(num_vars)) - pivot_set

        # 更新 var_exprs: 主变量用自由变量表示
        for i, pcol in enumerate(pivot_cols):
            coeffs = {}
            const = A[i, num_vars]
            for j in range(num_vars):
                if j != pcol and abs(A[i, j]) > 1e-10:
                    coeffs[j] = -A[i, j]
            self.var_exprs[pcol] = LinearExpr(coeffs, const)

        # 递归展开: 主变量表达式中可能引用其他主变量
        changed = True
        max_iter = 100
        while changed and max_iter > 0:
            changed = False
            max_iter -= 1
            for var_id in range(num_vars):
                expr = self.var_exprs[var_id]
                new_coeffs = {}
                new_const = expr.const
                for k, v in expr.coeffs.items():
                    sub = self.var_exprs[k]
                    if len(sub.coeffs) == 1 and k in sub.coeffs and sub.coeffs[k] == 1.0 and k != var_id:
                        # k is still a free var or identity
                        new_coeffs[k] = new_coeffs.get(k, 0) + v
                    elif k != var_id and not (len(sub.coeffs) == 1 and k in sub.coeffs and sub.coeffs[k] == 1.0):
                        for sk, sv in sub.coeffs.items():
                            new_coeffs[sk] = new_coeffs.get(sk, 0) + v * sv
                        new_const += v * sub.const
                        changed = True
                    else:
                        new_coeffs[k] = new_coeffs.get(k, 0) + v
                if changed:
                    self.var_exprs[var_id] = LinearExpr(new_coeffs, new_const)

    def decode(self, free_values: Dict[int, float]) -> Tuple[List[float], List[float]]:
        """给定自由变量的值,解码出所有 box 的 x, y 坐标"""
        x = [0.0] * self.n
        y = [0.0] * self.n
        for i in range(self.n):
            x[i] = self.var_exprs[i].evaluate(free_values)
            y[i] = self.var_exprs[self.n + i].evaluate(free_values)
        return x, y


# ============================================================
# 代价函数
# ============================================================

def compute_hpwl(problem: Problem, x, y) -> float:
    """向量化 HPWL 计算"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    cx = x + problem.w_half
    cy = y + problem.h_half
    total = 0.0
    for net in problem.nets:
        if len(net) < 2:
            continue
        idx = np.array([b - 1 for b in net])
        total += float(np.max(cx[idx]) - np.min(cx[idx]))
        total += float(np.max(cy[idx]) - np.min(cy[idx]))
    return total


def compute_area(problem: Problem, x, y) -> float:
    """向量化 Area 计算"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) == 0:
        return 0.0
    min_x = float(np.min(x))
    min_y = float(np.min(y))
    max_x = float(np.max(x + problem.w_arr))
    max_y = float(np.max(y + problem.h_arr))
    return (max_x - min_x) * (max_y - min_y)


def compute_overlap_penalty(problem: Problem, x, y, margin: float = 0.0) -> float:
    """
    向量化重叠惩罚。margin > 0 时，每个 box 四边各扩 margin/2。
    用 numpy broadcasting 替代 O(n²) Python 双层循环。
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = problem.n
    if n < 2:
        return 0.0

    half = margin / 2.0
    w = problem.w_arr + margin
    h = problem.h_arr + margin

    xl = x - half
    xr = xl + w
    yl = y - half
    yr = yl + h

    # Broadcasting: (n,1) vs (1,n) → (n,n)
    ox = np.maximum(0.0, np.minimum(xr[:, None], xr[None, :]) - np.maximum(xl[:, None], xl[None, :]))
    oy = np.maximum(0.0, np.minimum(yr[:, None], yr[None, :]) - np.maximum(yl[:, None], yl[None, :]))
    overlap_matrix = ox * oy

    # 只求上三角（不含对角线），每对只算一次
    return float(np.sum(overlap_matrix[np.triu_indices(n, k=1)]))


def compute_proximity_penalty(problem: Problem, x, y,
                               repeat_group_weight: float = 2.0,
                               net_weight: float = 0.5,
                               group_size_power: float = 0.5) -> float:
    """
    向量化组内紧凑度惩罚。
    
    每个组的贡献按组规模缩放: weight * bbox_area * (n_members / 2) ^ group_size_power
    组内 box 越多，惩罚越大，促使大组优先紧凑。
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    cx = x + problem.w_half
    cy = y + problem.h_half
    total = 0.0
    
    # 1. 重复组
    for rg in problem.repeat_groups:
        for group in rg:
            idx = np.array([b - 1 for b in group])
            if len(idx) < 2:
                continue
            bbox_area = float((np.max(cx[idx]) - np.min(cx[idx])) *
                             (np.max(cy[idx]) - np.min(cy[idx])))
            size_factor = (len(idx) / 2.0) ** group_size_power
            total += repeat_group_weight * bbox_area * size_factor
    
    # 2. Net
    for net in problem.nets:
        if len(net) < 2:
            continue
        idx = np.array([b - 1 for b in net])
        bbox_area = float((np.max(cx[idx]) - np.min(cx[idx])) *
                         (np.max(cy[idx]) - np.min(cy[idx])))
        size_factor = (len(idx) / 2.0) ** group_size_power
        total += net_weight * bbox_area * size_factor
    
    return total


def compute_cost(problem: Problem, x, y,
                 overlap_lambda: float = 0.0,
                 margin: float = 0.0,
                 proximity_weight: float = 0.0,
                 group_size_power: float = 0.5) -> Tuple[float, float, float, float]:
    """返回 (total_cost, hpwl, area, overlap)"""
    hpwl = compute_hpwl(problem, x, y)
    area = compute_area(problem, x, y)
    overlap = compute_overlap_penalty(problem, x, y, margin=margin)
    total = 10 * hpwl + area + overlap_lambda * overlap
    
    if proximity_weight > 0:
        proximity = compute_proximity_penalty(problem, x, y,
                                               group_size_power=group_size_power)
        total += proximity_weight * proximity
    
    return total, hpwl, area, overlap


# ============================================================
# 模拟退火优化器
# ============================================================

class SimulatedAnnealing:
    def __init__(self, problem: Problem, cs: ConstraintSystem,
                 time_limit: float = 110.0, seed: int = 42):
        self.problem = problem
        self.cs = cs
        self.time_limit = time_limit
        self.seed = seed

        self.free_vars = sorted(cs.free_vars)
        self.num_vars = len(self.free_vars)

        self._precompute_matrices()

        self.best_values: Dict[int, float] = {}
        self.best_cost = float('inf')
        self.best_x: List[float] = []
        self.best_y: List[float] = []
        self.best_overlap = float('inf')

        self.iterations = 0
        self.accepted = 0

    def _precompute_matrices(self):
        """预计算 HPWL/Area 的矩阵形式以加速评估"""
        n = self.problem.n
        nv = self.num_vars
        cs = self.cs

        # 构建 x_i, y_i 关于自由变量的系数矩阵
        self.x_coeffs = np.zeros((n, nv))
        self.x_const = np.zeros(n)
        self.y_coeffs = np.zeros((n, nv))
        self.y_const = np.zeros(n)

        fv_idx = {fv: j for j, fv in enumerate(self.free_vars)}

        for i in range(n):
            expr_x = cs.var_exprs[i]
            self.x_const[i] = expr_x.const
            for fv, c in expr_x.coeffs.items():
                if fv in fv_idx:
                    self.x_coeffs[i, fv_idx[fv]] = c

            expr_y = cs.var_exprs[n + i]
            self.y_const[i] = expr_y.const
            for fv, c in expr_y.coeffs.items():
                if fv in fv_idx:
                    self.y_coeffs[i, fv_idx[fv]] = c



    def _decode_fast(self, var_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """快速解码: 矩阵运算，返回 numpy 数组"""
        x = self.x_coeffs @ var_array + self.x_const
        y = self.y_coeffs @ var_array + self.y_const
        return x, y

    def _compute_layout_scale(self, x, y) -> float:
        """计算布局尺度，用于自适应扰动"""
        if len(x) == 0:
            return 1.0
        x_range = float(np.max(x) - np.min(x)) + np.max(self.problem.w_arr)
        y_range = float(np.max(y) - np.min(y)) + np.max(self.problem.h_arr)
        return float(max(x_range, y_range, 1.0))

    def _fit_to_target(self, target_x: List[float], target_y: List[float]) -> np.ndarray:
        """将目标坐标映射到约束空间的自由变量值"""
        n = self.problem.n
        nv = self.num_vars
        cs = self.cs

        # 构建目标向量
        target = np.zeros(2 * n)
        for i in range(n):
            target[i] = target_x[i]
            target[n + i] = target_y[i]

        # 构建系数矩阵
        fv_idx = {fv: j for j, fv in enumerate(self.free_vars)}
        A = np.zeros((2 * n, nv))
        b = np.zeros(2 * n)

        for i in range(n):
            expr_x = cs.var_exprs[i]
            for fv, c in expr_x.coeffs.items():
                if fv in fv_idx:
                    A[i, fv_idx[fv]] = c
            b[i] = target_x[i] - expr_x.const

            expr_y = cs.var_exprs[n + i]
            for fv, c in expr_y.coeffs.items():
                if fv in fv_idx:
                    A[n + i, fv_idx[fv]] = c
            b[n + i] = target_y[i] - expr_y.const

        # 最小二乘求解
        result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return result

    # ---- 初始化策略 ----

    def _initial_network_aware(self, seed_offset=0) -> np.ndarray:
        """Network-aware initialization: cluster boxes by net connectivity"""
        n = self.problem.n
        w, h = self.problem.widths, self.problem.heights
        random.seed(self.seed + seed_offset)

        # Build net connectivity graph
        net_neighbors = {i: set() for i in range(n)}
        for net in self.problem.nets:
            boxes = [b - 1 for b in net]
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    net_neighbors[boxes[i]].add(boxes[j])
                    net_neighbors[boxes[j]].add(boxes[i])

        # BFS order from most connected box
        order = sorted(range(n), key=lambda i: -len(net_neighbors[i]))
        visited = set()
        bfs_order = []
        for start in order:
            if start in visited:
                continue
            queue = [start]
            visited.add(start)
            while queue:
                node = queue.pop(0)
                bfs_order.append(node)
                for nb in sorted(net_neighbors[node], key=lambda x: -len(net_neighbors[x])):
                    if nb not in visited:
                        visited.add(nb)
                        queue.append(nb)
        for i in range(n):
            if i not in visited:
                bfs_order.append(i)

        total_area = sum(w[i] * h[i] for i in range(n))
        side = math.sqrt(total_area) * 1.05

        target_x = [0.0] * n
        target_y = [0.0] * n
        xc, yc, rh, gap = 0.0, 0.0, 0.0, 0.1

        for i in bfs_order:
            if xc + w[i] > side and xc > 0:
                xc, yc = 0.0, yc + rh + gap
                rh = 0.0
            target_x[i], target_y[i] = xc, yc
            xc += w[i] + gap
            rh = max(rh, h[i])

        return self._fit_to_target(target_x, target_y)

    def _initial_compact_grid(self, seed_offset=0) -> np.ndarray:
        """Compact grid initialization: tight row-based placement"""
        n = self.problem.n
        w, h = self.problem.widths, self.problem.heights
        total_area = sum(w[i] * h[i] for i in range(n))
        side = math.sqrt(total_area) * 1.05
        order = sorted(range(n), key=lambda i: h[i], reverse=True)
        random.seed(self.seed + seed_offset)

        target_x = [0.0] * n
        target_y = [0.0] * n
        xc, yc, rh, gap = 0.0, 0.0, 0.0, 0.2

        for i in order:
            if xc + w[i] > side and xc > 0:
                xc, yc = 0.0, yc + rh + gap
                rh = 0.0
            target_x[i], target_y[i] = xc, yc
            xc += w[i] + gap
            rh = max(rh, h[i])

        return self._fit_to_target(target_x, target_y)

    def _initial_random_tight(self, seed_offset=0) -> np.ndarray:
        """Random tight initialization: random order with tight packing"""
        n = self.problem.n
        w, h = self.problem.widths, self.problem.heights
        random.seed(self.seed + seed_offset + 200)
        total_area = sum(w[i] * h[i] for i in range(n))
        side = math.sqrt(total_area) * 1.2
        order = list(range(n))
        random.shuffle(order)

        target_x = [0.0] * n
        target_y = [0.0] * n
        xc, yc, rh = 0.0, 0.0, 0.0

        for i in order:
            if xc + w[i] > side and xc > 0:
                xc = random.uniform(0, 2)
                yc += rh + random.uniform(0.5, 2)
                rh = 0.0
            target_x[i], target_y[i] = xc, yc
            xc += w[i] + random.uniform(0.2, 1.5)
            rh = max(rh, h[i])

        return self._fit_to_target(target_x, target_y)

    def _initial_clustered(self, seed_offset=0) -> np.ndarray:
        """Clustered initialization: group by net connectivity"""
        n = self.problem.n
        w, h = self.problem.widths, self.problem.heights
        net_boxes = set()
        for net in self.problem.nets:
            for b in net:
                net_boxes.add(b - 1)
        order = sorted(range(n), key=lambda i: (i not in net_boxes, -w[i] * h[i]))
        total_area = sum(w[i] * h[i] for i in range(n))
        side = math.sqrt(total_area) * 1.15

        target_x = [0.0] * n
        target_y = [0.0] * n
        xc, yc, rh, gap = 0.0, 0.0, 0.0, 0.3

        for i in order:
            if xc + w[i] > side and xc > 0:
                xc, yc = 0.0, yc + rh + gap
                rh = 0.0
            target_x[i], target_y[i] = xc, yc
            xc += w[i] + gap
            rh = max(rh, h[i])

        return self._fit_to_target(target_x, target_y)

    def _initial_quadratic_placement(self, seed_offset=0) -> np.ndarray:
        """
        Quadratic placement initialization: minimize wirelength using quadratic programming

        Solves: min Σ_net Σ_{i,j in net} [(xi-xj)2 + (yi-yj)2]
        This naturally clusters connected components and provides a good initial layout.
        """
        n = self.problem.n
        w, h = self.problem.widths, self.problem.heights
        random.seed(self.seed + seed_offset)

        # Build connectivity matrix C[i][j] = number of nets connecting i and j
        C = np.zeros((n, n))
        for net in self.problem.nets:
            boxes = [b - 1 for b in net]
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    C[boxes[i], boxes[j]] += 1
                    C[boxes[j], boxes[i]] += 1

        # Build Laplacian matrix L = D - C
        # D is degree matrix (diagonal)
        D = np.diag(C.sum(axis=1))
        L = D - C

        # Add anchor constraints to avoid trivial solution
        # Anchor the first box at origin and spread others
        L[0, 0] += 1000  # Strong anchor for box 0

        # Solve Lx = 0 and Ly = 0
        # Use eigenvalue decomposition for stability
        eigenvalues, eigenvectors = np.linalg.eigh(L)

        # Skip the smallest eigenvalue (near 0, corresponds to trivial solution)
        # Use the next 2 eigenvectors for x and y coordinates
        if len(eigenvalues) > 2:
            x_raw = eigenvectors[:, 1]  # Second smallest eigenvector
            y_raw = eigenvectors[:, 2]  # Third smallest eigenvector
        else:
            # Fallback if not enough eigenvectors
            x_raw = np.random.randn(n)
            y_raw = np.random.randn(n)

        # Scale to reasonable size
        total_area = sum(w[i] * h[i] for i in range(n))
        target_side = math.sqrt(total_area) * 1.5

        x_range = x_raw.max() - x_raw.min()
        y_range = y_raw.max() - y_raw.min()

        if x_range > 0:
            x_raw = (x_raw - x_raw.min()) / x_range * target_side
        if y_range > 0:
            y_raw = (y_raw - y_raw.min()) / y_range * target_side

        # Add some randomness to break symmetry
        noise_scale = target_side * 0.05
        x_raw += np.random.randn(n) * noise_scale
        y_raw += np.random.randn(n) * noise_scale

        target_x = x_raw.tolist()
        target_y = y_raw.tolist()

        return self._fit_to_target(target_x, target_y)

    @classmethod
    def get_available_strategies(cls) -> List[str]:
        """Get list of available initialization strategies"""
        return [
            'network_aware',
            'compact_grid',
            'random_tight',
            'clustered',
            'quadratic_placement'
        ]

    def _get_strategy_function(self, strategy_name: str, seed_offset: int = 0):
        """Get strategy function by name"""
        strategies = {
            'network_aware': self._initial_network_aware,
            'compact_grid': self._initial_compact_grid,
            'random_tight': self._initial_random_tight,
            'clustered': self._initial_clustered,
            'quadratic_placement': self._initial_quadratic_placement,
        }
        if strategy_name not in strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}. "
                           f"Available: {self.get_available_strategies()}")
        return lambda: strategies[strategy_name](seed_offset)

    def run(self, strategies: List[str] = None, 
            initial_vars: 'np.ndarray | None' = None) -> Tuple[List[float], List[float], float]:
        """
        Run simulated annealing optimization.

        Args:
            strategies: List of strategy names to use. If None, uses default sequence.
                       Available strategies: network_aware, compact_grid, random_tight,
                                           clustered, quadratic_placement

        Returns:
            (x_coords, y_coords, cost)
        """
        t0 = time.time()
        self._global_t0 = t0

        if strategies is None:
            strategies = [
                'quadratic_placement',
                'network_aware',
                'compact_grid',
                'clustered',
                'quadratic_placement',
                'random_tight'
            ]

        round_num = 0

        if initial_vars is not None:
            # External initial solution provided (e.g., from Phase A+B greedy)
            # Skip exploration, directly seed best and go to Phase 2
            init_x, init_y = self._decode_fast(initial_vars)
            _, init_hpwl, init_area, init_overlap = \
                compute_cost(self.problem, init_x, init_y)
            init_real_cost = 10 * init_hpwl + init_area
            self._update_best(initial_vars, init_x, init_y,
                             init_real_cost, init_overlap)
            print(f"    [初始解] cost={init_real_cost:.2f}, overlap={init_overlap:.2f}")
        else:
            # Build init functions
            init_strategies = []
            for i, strategy_name in enumerate(strategies):
                seed_offset = i * 1000
                init_strategies.append(self._get_strategy_function(strategy_name, seed_offset))

            # Phase 1: Try multiple initial strategies
            exploration_time = min(self.time_limit * 0.25, 15.0)

            while (time.time() - t0) < exploration_time and round_num < len(init_strategies):
                round_num += 1
                strategy = init_strategies[round_num - 1]

                time_budget = min(
                    (exploration_time - (time.time() - t0)) / max(1, len(init_strategies) - round_num + 1),
                    8.0
                )
                time_budget = max(time_budget, 2.0)

                self._run_round(strategy, time_budget, round_num)
                elapsed = time.time() - t0
                print(f"    [轮 {round_num}] cost={self.best_cost:.2f}, time={elapsed:.1f}s")

        # Phase 2: Multi-round restart optimization from best solution
        # Split remaining time into sub-rounds to avoid temperature dead zone.
        # Each sub-round restarts temperature, preventing the SA from getting
        # stuck at T=0 for most of the budget (the root cause of 120s < 30s).
        remaining = self.time_limit - (time.time() - t0)
        if remaining > 1.0 and self.best_cost < float('inf'):
            # Sub-round duration: 15-25s depending on remaining time
            # Short enough to restart temp multiple times,
            # long enough to converge within each round
            sub_round_sec = min(25.0, max(15.0, remaining / 4.0))
            sub_round_num = 0
            while (time.time() - t0) < self.time_limit - 1.0:
                sub_round_num += 1
                round_num += 1
                sub_remaining = self.time_limit - (time.time() - t0)
                if sub_remaining < 2.0:
                    break
                budget = min(sub_round_sec, sub_remaining - 0.5)
                if budget < 5.0:
                    budget = sub_remaining
                if budget < 5.0:
                    break  # Too little time for a meaningful round

                # Start from current best, with slight random perturbation
                # to escape local optima (except first sub-round)
                best_arr = np.array([self.best_values[fv] for fv in self.free_vars])
                if sub_round_num > 1:
                    # Add scaled noise to escape local optima.
                    # 15% of layout spread — enough to explore new regions
                    # without destroying the overall structure.
                    spread = float(np.ptp(best_arr))
                    noise_scale = spread * 0.15
                    best_arr = best_arr + np.random.randn(len(best_arr)) * noise_scale

                elapsed = time.time() - t0
                print(f"  [阶段2.{sub_round_num}] 重启优化, "
                      f"budget={budget:.1f}s, prev_best={self.best_cost:.2f}")
                self._run_round(
                    lambda arr=best_arr: arr,
                    budget, round_num
                )
                elapsed = time.time() - t0
                print(f"    [阶段2.{sub_round_num}] cost={self.best_cost:.2f}, "
                      f"time={elapsed:.1f}s")

        elapsed = time.time() - t0
        print(f"\nSA 完成: {round_num} 轮, iter={self.iterations}, "
              f"best_cost={self.best_cost:.2f}, overlap={self.best_overlap:.2f}, "
              f"time={elapsed:.1f}s")

        # 返回 list 格式以保持接口兼容
        bx = self.best_x.tolist() if isinstance(self.best_x, np.ndarray) else list(self.best_x)
        by = self.best_y.tolist() if isinstance(self.best_y, np.ndarray) else list(self.best_y)
        return bx, by, self.best_cost

    def _run_round(self, init_fn, time_budget, round_num):
        local_t0 = time.time()
        global_t0 = self._global_t0

        current_arr = init_fn()
        current_x, current_y = self._decode_fast(current_arr)
        _, current_hpwl, current_area, current_overlap = \
            compute_cost(self.problem, current_x, current_y)
        _, _, _, current_overlap_m = \
            compute_cost(self.problem, current_x, current_y, margin=OVERLAP_MARGIN)
        current_real_cost = 10 * current_hpwl + current_area
        layout_scale = self._compute_layout_scale(current_x, current_y)

        self._update_best(current_arr, current_x, current_y,
                         current_real_cost, current_overlap)

        max_temp = layout_scale * 100.0
        temperature = max_temp
        # Dynamic alpha: scale so temperature reaches ~1% of max_temp
        # at ~80% of the time budget. Estimate ~15K iter/sec.
        est_iters = time_budget * 15000
        target_iters = max(est_iters * 0.8, 10000)
        alpha = 0.01 ** (1.0 / target_iters)  # alpha^N = 0.01
        no_improve = 0

        while (time.time() - local_t0) < time_budget:
            self.iterations += 1

            progress = (time.time() - local_t0) / time_budget
            if current_overlap_m > 0.01:
                overlap_lambda = 50000.0 * (1 + progress * 100)
                area_lambda = 0.0
            else:
                overlap_lambda = 0.0
                if progress < 0.3:
                    area_lambda = 0.0
                else:
                    area_lambda = 0.5 + (progress - 0.3) * 1.0

            new_arr = self._perturb(current_arr, temperature, max_temp, layout_scale)
            new_x, new_y = self._decode_fast(new_arr)

            # 基于规模的自适应 proximity 权重（5档 + 组规模缩放）
            n_boxes = self.problem.n
            if n_boxes <= 15:
                # 小规模：禁用，基础 cost 足够
                proximity_w = 0.0
                group_size_power = 0.0
            elif n_boxes <= 24:
                # 中小规模：弱引导
                proximity_w = 0.5
                group_size_power = 0.3
            elif n_boxes <= 40:
                # 中大规模：中等引导，逐渐衰减
                proximity_w = 1.0 - 0.5 * progress
                group_size_power = 0.5
            elif n_boxes <= 70:
                # 大规模：强引导
                proximity_w = 2.0 - 1.0 * progress
                group_size_power = 0.7
            else:
                # 超大规模：最强引导
                proximity_w = 3.0 - 1.5 * progress
                group_size_power = 1.0

            new_total, new_hpwl, new_area, new_overlap_m = \
                compute_cost(self.problem, new_x, new_y, overlap_lambda,
                             margin=OVERLAP_MARGIN, proximity_weight=proximity_w,
                             group_size_power=group_size_power)
            new_real_overlap = compute_overlap_penalty(self.problem, new_x, new_y)
            new_real_cost = 10 * new_hpwl + new_area  # True cost, no penalties

            # For acceptance: new_total already has overlap+proximity from compute_cost.
            # Add area_lambda here to guide area optimization.
            new_total_guided = new_total + area_lambda * new_area

            current_total = current_real_cost + area_lambda * current_area + \
                            overlap_lambda * current_overlap_m
            if proximity_w > 0:
                current_total += proximity_w * compute_proximity_penalty(
                    self.problem, current_x, current_y,
                    group_size_power=group_size_power)
            delta = new_total_guided - current_total

            accept = False
            if delta < 0:
                accept = True
            elif temperature > 0.001:
                prob = math.exp(-delta / temperature)
                if random.random() < prob:
                    accept = True

            if accept:
                current_arr = new_arr
                current_real_cost = new_real_cost  # True cost only
                current_area = new_area  # Track area for area_lambda in next delta
                current_x, current_y = new_x, new_y
                current_overlap = new_real_overlap
                current_overlap_m = new_overlap_m
                self.accepted += 1
                no_improve = 0

                if self.iterations % 1000 == 0:
                    layout_scale = self._compute_layout_scale(current_x, current_y)

                self._update_best(current_arr, current_x, current_y,
                                new_real_cost, current_overlap)
            else:
                no_improve += 1

            temperature *= alpha

            if no_improve > 8000:
                temperature = max(max_temp * 0.2, 1.0)
                no_improve = 0

            if self.iterations % 100000 == 0:
                elapsed = time.time() - global_t0
                print(f"    iter={self.iterations}, T={temperature:.1f}, "
                      f"cost={self.best_cost:.2f}, ov={self.best_overlap:.2f}, "
                      f"time={elapsed:.1f}s")

    def _update_best(self, var_arr, x, y, real_cost, overlap):
        if overlap < 0.001:
            if real_cost < self.best_cost or self.best_overlap >= 0.001:
                self.best_cost = real_cost
                self.best_values = {fv: float(var_arr[j])
                                   for j, fv in enumerate(self.free_vars)}
                self.best_x = np.array(x)
                self.best_y = np.array(y)
                self.best_overlap = overlap
        elif self.best_overlap >= 0.001 and overlap < self.best_overlap:
            self.best_values = {fv: float(var_arr[j])
                               for j, fv in enumerate(self.free_vars)}
            self.best_x = np.array(x)
            self.best_y = np.array(y)
            self.best_overlap = overlap
            self.best_cost = real_cost

    def _perturb(self, var_array: np.ndarray, temperature: float,
                 max_temp: float, layout_scale: float) -> np.ndarray:
        new_vars = var_array.copy()
        ratio = temperature / max_temp

        # Use median box size as perturbation unit
        all_dims = np.concatenate([self.problem.w_arr, self.problem.h_arr])
        box_scale = float(np.median(all_dims))

        r = random.random()
        if r < 0.15 and self.best_overlap < 0.001:
            # Compression move: scale all vars toward center of mass
            factor = 1.0 - random.uniform(0.005, 0.03)
            center = np.mean(new_vars)
            new_vars = center + (new_vars - center) * factor
        elif r < 0.25:
            # Area-focused move: shrink boundary boxes toward center
            if len(self.best_x) > 0:
                factor = 1.0 - random.uniform(0.01, 0.05)
                new_vars = new_vars * factor + (1 - factor) * np.mean(new_vars)
        else:
            # Standard perturbation
            if ratio > 0.5:
                scale = layout_scale * ratio * 0.3
            elif ratio > 0.1:
                scale = layout_scale * ratio * 0.15
            else:
                scale = box_scale * max(ratio, 0.01) * 2.0

            # Pick 1-3 variables to perturb
            num_perturb = min(random.randint(1, 3), self.num_vars)
            indices = random.sample(range(self.num_vars), num_perturb)

            for idx in indices:
                delta = scale * random.gauss(0, 1) / max(abs(random.gauss(0, 1)), 0.1)
                delta = max(min(delta, scale * 5), -scale * 5)
                new_vars[idx] += delta

        return new_vars


# ============================================================
# 后处理
# ============================================================

def _postprocess(problem: Problem, cs: ConstraintSystem,
                 x_coords, y_coords,
                 remaining_time: float = 10.0):
    """后处理:在约束满足前提下微调消除残余重叠"""
    n = problem.n
    free_vars = sorted(cs.free_vars)
    num_vars = len(free_vars)
    x_coords = np.asarray(x_coords, dtype=np.float64)
    y_coords = np.asarray(y_coords, dtype=np.float64)

    A = np.zeros((2 * n, num_vars))
    b = np.zeros(2 * n)

    for i in range(n):
        expr_x = cs.var_exprs[i]
        for j, fv in enumerate(free_vars):
            A[i, j] = expr_x.coeffs.get(fv, 0.0)
        b[i] = x_coords[i] - expr_x.const
        expr_y = cs.var_exprs[n + i]
        for j, fv in enumerate(free_vars):
            A[n + i, j] = expr_y.coeffs.get(fv, 0.0)
        b[n + i] = y_coords[i] - expr_y.const

    result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    current_values = {fv: float(result[j]) for j, fv in enumerate(free_vars)}

    best_values = current_values.copy()
    best_overlap_m = compute_overlap_penalty(problem, x_coords, y_coords,
                                              margin=OVERLAP_MARGIN)
    best_real_overlap = compute_overlap_penalty(problem, x_coords, y_coords)
    best_x = x_coords.copy()
    best_y = y_coords.copy()

    t0 = time.time()
    max_dim = max(float(np.max(problem.w_arr)), float(np.max(problem.h_arr)))
    step = max_dim * 0.5

    for iteration in range(10000):
        if time.time() - t0 > min(remaining_time, 15.0):
            break
        new_values = current_values.copy()
        var = free_vars[iteration % num_vars]
        delta = step * (1 if iteration % 2 == 0 else -1) * (0.5 + 0.5 * np.random.random())
        new_values[var] += delta
        new_x, new_y = cs.decode(new_values)
        new_overlap_m = compute_overlap_penalty(problem, new_x, new_y,
                                                 margin=OVERLAP_MARGIN)
        new_real_overlap = compute_overlap_penalty(problem, new_x, new_y)

        if new_overlap_m < best_overlap_m:
            best_overlap_m = new_overlap_m
            best_real_overlap = new_real_overlap
            best_values = new_values.copy()
            best_x = np.array(new_x)
            best_y = np.array(new_y)
            current_values = new_values
            if new_real_overlap < 0.001:
                break
        step *= 0.999

    return best_x.tolist(), best_y.tolist()


def _compress_layout(problem: Problem, cs: ConstraintSystem, 
                     x_coords, y_coords,
                     remaining_time: float = 5.0):
    """
    全局压缩后处理：仅在能改善总成本时应用
    """
    free_vars = sorted(cs.free_vars)
    num_vars = len(free_vars)
    x_coords = np.asarray(x_coords, dtype=np.float64)
    y_coords = np.asarray(y_coords, dtype=np.float64)

    # 将当前坐标映射回自由变量
    A = np.zeros((2 * problem.n, num_vars))
    b_vec = np.zeros(2 * problem.n)

    for i in range(problem.n):
        expr_x = cs.var_exprs[i]
        for j, fv in enumerate(free_vars):
            A[i, j] = expr_x.coeffs.get(fv, 0.0)
        b_vec[i] = x_coords[i] - expr_x.const
        expr_y = cs.var_exprs[problem.n + i]
        for j, fv in enumerate(free_vars):
            A[problem.n + i, j] = expr_y.coeffs.get(fv, 0.0)
        b_vec[problem.n + i] = y_coords[i] - expr_y.const

    result, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
    current_arr = result

    best_x, best_y = cs.decode(
        {fv: float(current_arr[j]) for j, fv in enumerate(free_vars)})
    _, hpwl0, area0, overlap0 = compute_cost(problem, x_coords, y_coords)
    _, new_hpwl, new_area, new_overlap = compute_cost(problem, best_x, best_y)
    best_cost = 10 * new_hpwl + new_area
    initial_cost = 10 * hpwl0 + area0

    print(f"    压缩前: cost={best_cost:.1f}, area={area0:.1f}, hpwl={hpwl0:.1f}, overlap={overlap0:.2f}")

    t0 = time.time()
    improved = True
    iteration = 0

    while improved and (time.time() - t0) < remaining_time:
        improved = False
        iteration += 1
        
        scales_x = [0.99, 0.98, 0.97, 0.96, 0.95, 1.00, 1.01, 1.02]
        scales_y = [0.99, 0.98, 0.97, 0.96, 0.95, 1.00, 1.01, 1.02]
        
        best_scale_combo = None
        best_scale_cost = best_cost
        
        for sx in scales_x:
            for sy in scales_y:
                if sx == 1.00 and sy == 1.00:
                    continue
                if time.time() - t0 > remaining_time * 0.85:
                    break
                
                new_arr = current_arr.copy()
                scale = (sx + sy) / 2.0
                centroid = np.mean(current_arr)
                new_arr = centroid + (current_arr - centroid) * scale
                
                new_x, new_y = cs.decode(
                    {fv: float(new_arr[j]) for j, fv in enumerate(free_vars)})
                
                overlap = compute_overlap_penalty(problem, new_x, new_y)
                # Only accept if no overlap introduced (overlap < 0.001)
                # or if it reduces existing overlap
                if overlap > 0.001 and overlap >= overlap0:
                    continue
                
                _, new_hpwl, new_area, new_overlap = compute_cost(problem, new_x, new_y)
                new_cost = 10 * new_hpwl + new_area
                
                # Only accept if cost improves AND overlap doesn't worsen
                if new_cost < best_scale_cost * 0.995 and new_overlap <= overlap0:
                    best_scale_cost = new_cost
                    best_scale_combo = (sx, sy)
        
        if best_scale_combo is not None:
            sx, sy = best_scale_combo
            scale = (sx + sy) / 2.0
            centroid = np.mean(current_arr)
            current_arr = centroid + (current_arr - centroid) * scale
            
            best_x, best_y = cs.decode(
                {fv: float(current_arr[j]) for j, fv in enumerate(free_vars)})
            
            _, new_hpwl, new_area, new_overlap = compute_cost(problem, best_x, best_y)
            best_cost = best_scale_cost
            improved = True
            
            print(f"    压缩 iter={iteration}, scale=({sx:.2f},{sy:.2f}): "
                  f"cost={best_cost:.1f}, area={new_area:.1f}, hpwl={new_hpwl:.1f}, "
                  f"overlap={new_overlap:.2f}")
    
    print(f"    压缩后: cost={best_cost:.1f}, 改善={100*(1-best_cost/(10*hpwl0+area0)):.1f}%")
    bx = best_x.tolist() if isinstance(best_x, np.ndarray) else list(best_x)
    by = best_y.tolist() if isinstance(best_y, np.ndarray) else list(best_y)
    return bx, by


# ============================================================
# 两阶段分区布局 (Partitioned Placement)
# ============================================================

def _get_constrained_boxes(problem: Problem) -> Set[int]:
    """返回所有受几何约束的 box 索引 (0-indexed)"""
    constrained = set()
    for group in problem.sym_x_groups:
        for pair in group.get('symmetry_pair', []):
            constrained.update(b-1 for b in pair)
        for s in group.get('self_symmetry', []):
            constrained.add(s-1)
    for group in problem.sym_y_groups:
        for pair in group.get('symmetry_pair', []):
            constrained.update(b-1 for b in pair)
        for s in group.get('self_symmetry', []):
            constrained.add(s-1)
    for group_list in [problem.align_left, problem.align_right,
                       problem.align_top, problem.align_bottom]:
        for group in group_list:
            constrained.update(b-1 for b in group)
    for rg in problem.repeat_groups:
        for grp in rg:
            constrained.update(b-1 for b in grp)
    return constrained


def _build_sub_problem(problem: Problem, constrained: Set[int]) -> Dict:
    """构建只含约束 box 的子问题，重新索引"""
    clist = sorted(constrained)
    old2new = {old: new for new, old in enumerate(clist)}
    cset = set(clist)

    sub_sizes = [[problem.widths[i], problem.heights[i]] for i in clist]

    def remap_sym(groups):
        result = []
        for g in groups:
            ng = {'symmetry_pair': [], 'self_symmetry': []}
            for pair in g.get('symmetry_pair', []):
                a, b = pair[0]-1, pair[1]-1
                if a in cset and b in cset:
                    ng['symmetry_pair'].append([old2new[a]+1, old2new[b]+1])
            for s in g.get('self_symmetry', []):
                si = s-1
                if si in cset:
                    ng['self_symmetry'].append(old2new[si]+1)
            if ng['symmetry_pair'] or ng['self_symmetry']:
                result.append(ng)
        return result

    def remap_align(align_groups):
        result = []
        for group in align_groups:
            boxes = [b-1 for b in group if b-1 in cset]
            if len(boxes) >= 2:
                result.append([old2new[b]+1 for b in boxes])
        return result

    def remap_repeat(rg_list):
        result = []
        for rg in rg_list:
            new_groups = []
            valid = True
            for grp in rg:
                boxes = [b-1 for b in grp]
                if not all(b in cset for b in boxes):
                    valid = False
                    break
                new_groups.append([old2new[b]+1 for b in boxes])
            if valid and len(new_groups) >= 2:
                result.append({'groups': new_groups})
        return result

    def remap_nets(nets):
        result = []
        for net in nets:
            boxes = [b-1 for b in net]
            relevant = [old2new[b]+1 for b in boxes if b in cset]
            if len(relevant) >= 2:
                result.append(relevant)
        return result

    sub_align = {}
    for key in ['left', 'right', 'top', 'bottom']:
        sub_align[key] = remap_align(getattr(problem, f'align_{key}', []))

    return {
        'box_size': sub_sizes,
        'symmetry_x': remap_sym(problem.sym_x_groups),
        'symmetry_y': remap_sym(problem.sym_y_groups),
        'repeat_groups': remap_repeat(problem.repeat_groups),
        'align': sub_align,
        'nets': remap_nets(problem.nets),
    }


def _classify_free_boxes(problem: Problem, constrained_set: Set[int], free_boxes: list):
    """将 free box 分为三类: cross_net, free_net, isolated"""
    cross_net, free_net, isolated = [], [], []
    for fb in free_boxes:
        fb_id = fb + 1  # 1-indexed
        has_cross = False
        has_free = False
        for net in problem.nets:
            if fb_id in net:
                for b in net:
                    bi = b - 1
                    if bi != fb and bi in constrained_set:
                        has_cross = True
                        break
                if not has_cross:
                    for b in net:
                        bi = b - 1
                        if bi != fb and bi not in constrained_set:
                            has_free = True
                            break
            if has_cross:
                break
        if has_cross:
            cross_net.append(fb)
        elif has_free:
            free_net.append(fb)
        else:
            isolated.append(fb)
    return cross_net, free_net, isolated


def _find_whitespace_slots(placed, bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y):
    """扫描 BBox 内的空白矩形区域，返回 [(x, y, w, h), ...]"""
    slots = []
    margin = OVERLAP_MARGIN
    for (px, py, pw, ph, _) in placed:
        # 右侧空位
        rx = px + pw + margin
        if rx < bbox_max_x:
            rw = bbox_max_x - rx
            if rw > 0:
                slots.append((rx, bbox_min_y, rw, bbox_max_y - bbox_min_y))
        # 左侧空位
        lw = px - margin - bbox_min_x
        if lw > 0:
            slots.append((bbox_min_x, bbox_min_y, lw, bbox_max_y - bbox_min_y))
        # 上方空位
        ty = py + ph + margin
        if ty < bbox_max_y:
            th = bbox_max_y - ty
            if th > 0:
                slots.append((bbox_min_x, ty, bbox_max_x - bbox_min_x, th))
        # 下方空位
        bh = py - margin - bbox_min_y
        if bh > 0:
            slots.append((bbox_min_x, bbox_min_y, bbox_max_x - bbox_min_x, bh))

    # 过滤：检查 slot 中心是否与任何已放置 box 重叠
    valid_slots = []
    for (sx, sy, sw, sh) in slots:
        if sw <= 0 or sh <= 0:
            continue
        cx = sx + sw / 2
        cy = sy + sh / 2
        ok = True
        for (px, py, pw, ph, _) in placed:
            if px <= cx <= px + pw and py <= cy <= py + ph:
                ok = False
                break
        if ok:
            valid_slots.append((sx, sy, sw, sh))

    valid_slots.sort(key=lambda s: s[2] * s[3], reverse=True)
    return valid_slots


def _place_free_box_smart(problem: Problem, fb: int, box_type: str,
                           placed: list,
                           bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y,
                           constrained_set: Set[int]) -> Tuple[float, float]:
    """智能放置 free box: 先找空位，再填坑"""
    w = float(problem.w_arr[fb])
    h = float(problem.h_arr[fb])
    fb_id = fb + 1

    cur_bbox_area = max(1.0, (bbox_max_x - bbox_min_x) * (bbox_max_y - bbox_min_y))

    candidates = []

    # 1. Whitespace slots（BBox 内空位）
    slots = _find_whitespace_slots(placed, bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y)
    for (sx, sy, sw, sh) in slots[:15]:
        if sw >= w and sh >= h:
            candidates.append((sx, sy))
            candidates.append((sx + sw - w, sy))
            candidates.append((sx, sy + sh - h))
            candidates.append((sx + sw - w, sy + sh - h))
            candidates.append((sx + (sw - w) / 2, sy + (sh - h) / 2))

    # 2. 已放置 box 的边缘（紧贴放置）
    for (px, py, pw, ph, _) in placed:
        candidates.append((px + pw + OVERLAP_MARGIN, py))
        candidates.append((px - w - OVERLAP_MARGIN, py))
        candidates.append((px, py + ph + OVERLAP_MARGIN))
        candidates.append((px, py - h - OVERLAP_MARGIN))

    # 3. 对 cross_net / free_net，找 connected box 附近
    if box_type in ("cross_net", "free_net"):
        connected_centers = []
        for net in problem.nets:
            if fb_id in net:
                for b in net:
                    bi = b - 1
                    if bi != fb:
                        for (px, py, pw, ph, pidx) in placed:
                            if pidx == bi:
                                connected_centers.append((px + pw/2, py + ph/2))
        if connected_centers:
            avg_cx = sum(c[0] for c in connected_centers) / len(connected_centers)
            avg_cy = sum(c[1] for c in connected_centers) / len(connected_centers)
            for dx in [-2, -1, 0, 1, 2]:
                for dy in [-2, -1, 0, 1, 2]:
                    candidates.append((avg_cx - w/2 + dx * w, avg_cy - h/2 + dy * h))


    # 评估每个候选
    best_pos = (bbox_max_x + OVERLAP_MARGIN, bbox_min_y)  # fallback
    best_score = float('inf')

    for (cx, cy) in candidates:
        # 检查重叠
        overlap = False
        for (px, py, pw, ph, _) in placed:
            ox = max(0, min(cx + w, px + pw) - max(cx, px))
            oy = max(0, min(cy + h, py + ph) - max(cy, py))
            if ox * oy > 0.001:
                overlap = True
                break
        if overlap:
            continue

        # 计算 BBox 扩张
        new_min_x = min(bbox_min_x, cx)
        new_min_y = min(bbox_min_y, cy)
        new_max_x = max(bbox_max_x, cx + w)
        new_max_y = max(bbox_max_y, cy + h)
        new_W = new_max_x - new_min_x
        new_H = new_max_y - new_min_y
        new_bbox_area = new_W * new_H
        area_delta = max(0, new_bbox_area - cur_bbox_area)

        # 形状惩罚：偏离正方形越远，惩罚越大
        aspect_ratio = max(new_W, new_H) / max(1.0, min(new_W, new_H))
        shape_penalty = (aspect_ratio - 1.0) ** 2

        # 计算 HPWL 贡献
        hpwl_contrib = 0.0
        for net in problem.nets:
            if fb_id in net:
                xs = [cx + w/2]
                ys = [cy + h/2]
                for b in net:
                    bi = b - 1
                    if bi != fb:
                        for (px, py, pw, ph, pidx) in placed:
                            if pidx == bi:
                                xs.append(px + pw/2)
                                ys.append(py + ph/2)
                if len(xs) > 1:
                    hpwl_contrib += (max(xs) - min(xs)) + (max(ys) - min(ys))

        # 统一评分: hpwl + alpha*area_delta + beta*shape_penalty
        # HPWL 最优先：同 net 靠近
        # area_delta 次优先：不同 net 可嵌套
        # shape_penalty：鼓励正方形布局
        score = hpwl_contrib + 1.0 * area_delta + 50.0 * shape_penalty
        if score < best_score:
            best_score = score
            best_pos = (cx, cy)

    return best_pos


def _partitioned_solve(problem: Problem, time_limit: float = 115.0,
                       strategies: List[str] = None):
    """
    两阶段分区布局:
    Phase A: 只优化约束 box (低维空间 SA)
    Phase B: 贪心初始化 + SA 精炼 free box
    """
    t0 = time.time()

    constrained = _get_constrained_boxes(problem)
    n = problem.n
    free_boxes = sorted(set(range(n)) - constrained)

    print(f"分区布局: {len(constrained)} constrained, {len(free_boxes)} free")

    # 如果约束 box 太少或太多，不分区
    if len(constrained) < 4 or len(constrained) > n * 0.8:
        print(f"  约束 box 比例不适合分区，使用标准求解")
        return None

    # Phase A: 构建子问题并求解
    sub_data = _build_sub_problem(problem, constrained)
    sub_problem = Problem(sub_data)
    sub_cs = ConstraintSystem(sub_problem)

    clist = sorted(constrained)
    sub_n_free = len(sub_cs.free_vars)
    print(f"  Phase A: {len(clist)} constrained boxes -> {sub_n_free} free vars "
          f"(vs full {len(ConstraintSystem(problem).free_vars)})")

    sub_sa = SimulatedAnnealing(sub_problem, sub_cs,
                                time_limit=time_limit * 0.4)
    sub_x, sub_y, sub_cost = sub_sa.run(strategies=strategies)

    # 后处理
    remaining = time_limit * 0.4 - (time.time() - t0)
    if remaining > 1.0:
        sub_x, sub_y = _postprocess(sub_problem, sub_cs, sub_x, sub_y,
                                     remaining_time=remaining)

    elapsed_a = time.time() - t0
    sub_hpwl = compute_hpwl(sub_problem, sub_x, sub_y)
    sub_area = compute_area(sub_problem, sub_x, sub_y)
    print(f"  Phase A 完成: cost={10*sub_hpwl+sub_area:.1f}, "
          f"hpwl={sub_hpwl:.1f}, area={sub_area:.1f}, time={elapsed_a:.1f}s")

    # bbox expansion 会破坏对称/自对称约束，已移除
    # Phase B 的 smart placement 会在 bbox 外自动找空位放置 free boxes

    # Phase B: 贪心初始化 + 全局 SA
    phase_b_budget = time_limit * 0.6
    print(f"  Phase B: 贪心初始化 + 全局 SA (budget={phase_b_budget:.0f}s)")

    # 初始化全坐标数组
    final_x = [0.0] * n
    final_y = [0.0] * n

    # 填入约束 box 坐标
    old2new = {old: new for new, old in enumerate(clist)}
    for i, old_idx in enumerate(clist):
        final_x[old_idx] = sub_x[i]
        final_y[old_idx] = sub_y[i]

    # 分类 free box
    cross_net, free_net, isolated = _classify_free_boxes(problem, constrained, free_boxes)
    print(f"    分类: cross_net={len(cross_net)}, free_net={len(free_net)}, isolated={len(isolated)}")

    # 构建已放置 box 列表: [(x, y, w, h, global_idx), ...]
    placed = []
    for c in constrained:
        placed.append((final_x[c], final_y[c], float(problem.w_arr[c]), float(problem.h_arr[c]), c))

    # 当前 BBox
    b_min_x = min(p[0] for p in placed)
    b_min_y = min(p[1] for p in placed)
    b_max_x = max(p[0] + p[2] for p in placed)
    b_max_y = max(p[1] + p[3] for p in placed)

    def _update_bbox(pos, bw, bh):
        nonlocal b_min_x, b_min_y, b_max_x, b_max_y
        b_min_x = min(b_min_x, pos[0])
        b_min_y = min(b_min_y, pos[1])
        b_max_x = max(b_max_x, pos[0] + bw)
        b_max_y = max(b_max_y, pos[1] + bh)

    # Step 1: 放置 cross_net free box
    for fb in cross_net:
        pos = _place_free_box_smart(problem, fb, "cross_net", placed,
                                     b_min_x, b_min_y, b_max_x, b_max_y, constrained)
        final_x[fb] = pos[0]
        final_y[fb] = pos[1]
        bw, bh = float(problem.w_arr[fb]), float(problem.h_arr[fb])
        placed.append((pos[0], pos[1], bw, bh, fb))
        _update_bbox(pos, bw, bh)

    # Step 2: 放置 free_net (按 net 聚类，同 net 的先放一起)
    visited = set()
    net_groups = []
    for fb in free_net:
        if fb in visited:
            continue
        group = {fb}
        fb_id = fb + 1
        for net in problem.nets:
            if fb_id in net:
                for b in net:
                    bi = b - 1
                    if bi in free_net and bi not in visited:
                        group.add(bi)
        net_groups.append(sorted(group))
        visited.update(group)

    for group in net_groups:
        for fb in group:
            pos = _place_free_box_smart(problem, fb, "free_net", placed,
                                         b_min_x, b_min_y, b_max_x, b_max_y, constrained)
            final_x[fb] = pos[0]
            final_y[fb] = pos[1]
            bw, bh = float(problem.w_arr[fb]), float(problem.h_arr[fb])
            placed.append((pos[0], pos[1], bw, bh, fb))
            _update_bbox(pos, bw, bh)

    # Step 3: 放置 isolated (纯填空，按面积从大到小)
    isolated_sorted = sorted(isolated,
                              key=lambda fb: float(problem.w_arr[fb]) * float(problem.h_arr[fb]),
                              reverse=True)
    for fb in isolated_sorted:
        pos = _place_free_box_smart(problem, fb, "isolated", placed,
                                     b_min_x, b_min_y, b_max_x, b_max_y, constrained)
        final_x[fb] = pos[0]
        final_y[fb] = pos[1]
        bw, bh = float(problem.w_arr[fb]), float(problem.h_arr[fb])
        placed.append((pos[0], pos[1], bw, bh, fb))
        _update_bbox(pos, bw, bh)

    greedy_time = time.time() - t0 - elapsed_a
    print(f"    贪心初始化完成: time={greedy_time:.1f}s")

    # Step 4: 全局 SA 精炼
    remaining_time = phase_b_budget - greedy_time
    if remaining_time > 5.0:
        print(f"    全局 SA 精炼: {remaining_time:.0f}s")
        
        # 构建全局约束系统和 SA
        global_cs = ConstraintSystem(problem)
        global_sa = SimulatedAnnealing(problem, global_cs, time_limit=remaining_time)
        
        # 把当前坐标编码为全局变量数组（逆解码）
        global_vars = global_sa._fit_to_target(final_x, final_y)
        
        # 以全局变量数组为初始解，跑全局 SA
        final_x_np, final_y_np, final_cost = global_sa.run(strategies=strategies, initial_vars=global_vars)
        
        final_x = list(final_x_np)
        final_y = list(final_y_np)
        
        global_time = time.time() - t0 - elapsed_a - greedy_time
        print(f"    全局 SA 完成: time={global_time:.1f}s")

    elapsed_b = time.time() - t0 - elapsed_a
    print(f"  Phase B 完成: time={elapsed_b:.1f}s")

    # 计算最终指标
    final_hpwl = compute_hpwl(problem, final_x, final_y)
    final_area = compute_area(problem, final_x, final_y)
    final_overlap = compute_overlap_penalty(problem, final_x, final_y)
    final_cost = 10 * final_hpwl + final_area

    print(f"  最终: cost={final_cost:.1f}, hpwl={final_hpwl:.1f}, "
          f"area={final_area:.1f}, overlap={final_overlap:.2f}")

    elapsed = time.time() - t0
    return final_x, final_y, elapsed


# ============================================================
# 求解入口
# ============================================================

def solve(problem: Problem, time_limit: float = 115.0,
          strategies: List[str] = None) -> Dict[str, Any]:
    """
    求解布局问题（纯算法，无 I/O 副作用，仅保留必要的维测打印）。

    Args:
        problem: 布局问题实例
        time_limit: 时间限制(秒)
        strategies: 初始化策略序列

    Returns:
        dict with keys: box_position, cost, hpwl, area, overlap, elapsed_seconds
    """
    t0 = time.time()

    # 尝试分区求解（降维优化）
    try:
        result = _partitioned_solve(problem, time_limit, strategies)
        if result is not None:
            return result
    except Exception as e:
        print(f"  分区求解失败，回退标准流程: {e}")

    # 1. 约束化简
    cs = ConstraintSystem(problem)
    print(f"约束化简: {problem.n} boxes -> {len(cs.free_vars)} 独立变量")

    # 2. 模拟退火
    sa = SimulatedAnnealing(problem, cs, time_limit=time_limit)
    x_coords, y_coords, cost = sa.run(strategies=strategies)

    # 3. 后处理去重叠
    remaining = time_limit - (time.time() - t0)
    if remaining > 1.0:
        x_coords, y_coords = _postprocess(problem, cs, x_coords, y_coords,
                                           remaining_time=remaining)

    # 4. 全局压缩后处理
    remaining = time_limit - (time.time() - t0)
    if remaining > 2.0:
        x_coords, y_coords = _compress_layout(problem, cs, x_coords, y_coords,
                                              remaining_time=remaining)

    elapsed = time.time() - t0
    return x_coords, y_coords, elapsed
