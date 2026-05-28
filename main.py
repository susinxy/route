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
from typing import List, Dict, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from route import (Problem, ConstraintSystem, SimulatedAnnealing,
                   solve, compute_cost, compute_hpwl, compute_area,
                   compute_overlap_penalty)


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

    # 重复组
    for rg in problem.repeat_groups:
        groups = [[b - 1 for b in grp] for grp in rg]
        if len(groups) < 2:
            continue
        ref = groups[0]
        ref_offsets = [(x[b] - x[ref[0]], y[b] - y[ref[0]]) for b in ref]
        for ig in range(1, len(groups)):
            grp = groups[ig]
            grp_offsets = [(x[b] - x[grp[0]], y[b] - y[grp[0]]) for b in grp]
            for j in range(len(ref)):
                dx_ref, dy_ref = ref_offsets[j]
                dx_grp, dy_grp = grp_offsets[j]
                if abs(dx_ref - dx_grp) > eps or abs(dy_ref - dy_grp) > eps:
                    violations.append(f"重复组: box {grp[j]+1}")

    return violations


# ============================================================
# 详细约束验证 (用于测试报告)
# ============================================================

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

    # ---- 1. X轴对称 ----
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
                    details["sym_x"].append(v)
                    group_ok = False
                if abs(y[i] - y[j]) > eps:
                    v = f"  X对称 y不等: box {pair[0]} y={y[i]:.3f}, box {pair[1]} y={y[j]:.3f}"
                    violations.append(v)
                    details["sym_x"].append(v)
                    group_ok = False
            for s in group.get("self_symmetry", []):
                si = s - 1
                if abs(x[si] + widths[si] / 2 - axis) > eps:
                    v = f"  X自对称: box {s}, center={x[si]+widths[si]/2:.3f}, axis={axis:.3f}"
                    violations.append(v)
                    details["sym_x"].append(v)
                    group_ok = False
            if group_ok:
                details["sym_x"].append(f"  X对称组{gi}: ✓ (axis={axis:.3f})")

    # ---- 2. Y轴对称 ----
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
                    details["sym_y"].append(v)
                    group_ok = False
                if abs(x[i] - x[j]) > eps:
                    v = f"  Y对称 x不等: box {pair[0]} x={x[i]:.3f}, box {pair[1]} x={x[j]:.3f}"
                    violations.append(v)
                    details["sym_y"].append(v)
                    group_ok = False
            for s in group.get("self_symmetry", []):
                si = s - 1
                if abs(y[si] + heights[si] / 2 - axis) > eps:
                    v = f"  Y自对称: box {s}, center={y[si]+heights[si]/2:.3f}, axis={axis:.3f}"
                    violations.append(v)
                    details["sym_y"].append(v)
                    group_ok = False
            if group_ok:
                details["sym_y"].append(f"  Y对称组{gi}: ✓ (axis={axis:.3f})")

    # ---- 3. 对齐 ----
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
                details["align"].append(v)
            else:
                details["align"].append(f"  {direction}对齐 boxes{grp}: ✓ (val={vals[0]:.3f})")

    # ---- 4. 重复组 ----
    for rgi, rg in enumerate(data.get("repeat_groups", [])):
        groups = [[b - 1 for b in grp] for grp in rg.get("groups", [])]
        if len(groups) < 2:
            continue
        ref = groups[0]
        ref_offsets = [(x[b] - x[ref[0]], y[b] - y[ref[0]]) for b in ref]
        rg_ok = True
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
                    details["repeat"].append(v)
                    rg_ok = False
        if rg_ok:
            details["repeat"].append(f"  重复组{rgi}: ✓")

    # ---- 5. 无重叠 ----
    overlap_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            ox = min(x[i] + widths[i], x[j] + widths[j]) - max(x[i], x[j])
            oy = min(y[i] + heights[i], y[j] + heights[j]) - max(y[i], y[j])
            if ox > eps and oy > eps:
                v = f"  重叠: box {i+1} & {j+1}, overlap=({ox:.3f}x{oy:.3f}={ox*oy:.3f})"
                violations.append(v)
                details["overlap"].append(v)
                overlap_count += 1
    if overlap_count == 0:
        details["overlap"].append("  无重叠: ✓")

    # ---- Cost 计算 ----
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
                  expected_only: bool = False) -> Dict[str, Any]:
    """
    对单个测试用例做完整验证:
      1. 验证 expected_output 约束
      2. 运行算法，验证算法输出约束
      3. 对比两者 cost
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
        return result

    # --- 2. 运行算法 ---
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

    # --- 3. 对比 ---
    if result.get("expected_cost") and result["algo_cost"]:
        exp_c = result["expected_cost"].get("cost", float("inf"))
        algo_c = result["algo_cost"].get("cost", float("inf"))
        result["cost_ratio"] = algo_c / exp_c if exp_c > 0 else float("inf")

    return result


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
# Main
# ============================================================

def main():
    time_limit = 115.0
    input_files = []
    strategies = None
    validate = False
    expected_only = False
    verbose = False
    list_cases = False
    directory = None
    filter_name = None

    # 解析命令行参数
    args = []
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--time" and i + 1 < len(sys.argv):
            time_limit = float(sys.argv[i + 1])
            i += 2
        elif arg == "--strategies" and i + 1 < len(sys.argv):
            strategies = [s.strip() for s in sys.argv[i + 1].split(',')]
            i += 2
        elif arg == "--validate":
            validate = True
            i += 1
        elif arg == "--expected-only":
            validate = True
            expected_only = True
            i += 1
        elif arg == "--verbose" or arg == "-v":
            verbose = True
            i += 1
        elif arg == "--list-strategies":
            print("可用的初始化策略:")
            for strategy in SimulatedAnnealing.get_available_strategies():
                print(f"  - {strategy}")
            return
        elif arg == "--list-cases":
            list_cases = True
            i += 1
        elif arg == "-d" and i + 1 < len(sys.argv):
            directory = sys.argv[i + 1]
            i += 2
        elif arg == "-t":
            args.append("-t")
            i += 1
        else:
            args.append(arg)
            i += 1

    # ---- 列出测试用例 ----
    if list_cases:
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
        return

    # ---- 内置测试用例 ----
    if "-t" in args:
        run_single_case(_builtin_test_case(), time_limit, "builtin_test", strategies,
                        validate=validate)
        return

    # ---- 测试用例模式 (--validate 或目录在 test_cases 下) ----
    use_test_mode = validate or (directory and "test_cases" in directory)

    if directory:
        if use_test_mode:
            # 测试模式：目录下的 case*.json
            cases = sorted(glob.glob(os.path.join(directory, "case*.json")))
        else:
            # 运行模式：目录下所有 JSON
            cases = sorted(glob.glob(os.path.join(directory, "*.json")))

        if not cases:
            print(f"错误: 目录 {directory} 中没有找到 JSON 文件")
            sys.exit(1)

        # 过滤
        positional = [a for a in args if not a.startswith("-")]
        if positional:
            filter_name = positional[0]
            cases = [f for f in cases if filter_name in os.path.basename(f)]

        if not cases:
            print(f"未找到匹配的用例 (filter={filter_name})")
            sys.exit(1)

        if use_test_mode:
            # 测试模式
            print(f"🦐 代码虾测试运行器")
            print(f"   找到 {len(cases)} 个用例")
            print(f"   时间限制: {time_limit}s/用例")
            if expected_only:
                print(f"   模式: 仅验证预期输出")
            print()

            results = []
            for filepath in cases:
                print(f"▶ 正在处理: {os.path.basename(filepath)}...")
                try:
                    r = run_case_test(filepath, time_limit=time_limit,
                                      expected_only=expected_only)
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
                for r in results
            )
            sys.exit(1 if any_fail else 0)
        else:
            # 运行模式
            input_files = cases

    elif args and not args[0].startswith("-"):
        # 文件或名称
        positional = args
        first = positional[0]

        if os.path.isfile(first):
            input_files = positional
        else:
            # 当作 test_cases 下的名称过滤
            cases = discover_cases(first)
            if cases:
                input_files = cases
                use_test_mode = True
            else:
                print(f"错误: 文件 {first} 不存在，也没有匹配的测试用例")
                sys.exit(1)

    if not input_files:
        print("Usage:")
        print("  python main.py input.json [--time 115] [--strategies s1,s2]")
        print("  python main.py case1.json case2.json [--time 115]")
        print("  python main.py -d ./test_cases --validate [--time 30]")
        print("  python main.py -d ./test_cases --expected-only")
        print("  python main.py --validate case01 [--time 30]")
        print("  python main.py -t [--time 30]")
        print("  python main.py --list-cases")
        print("  python main.py --list-strategies")
        sys.exit(1)

    # ---- 运行模式 ----
    if use_test_mode and not directory:
        # 按名称过滤的测试用例
        print(f"🦐 代码虾测试运行器")
        print(f"   找到 {len(input_files)} 个用例")
        print(f"   时间限制: {time_limit}s/用例")
        if expected_only:
            print(f"   模式: 仅验证预期输出")
        print()

        results = []
        for filepath in input_files:
            print(f"▶ 正在处理: {os.path.basename(filepath)}...")
            try:
                r = run_case_test(filepath, time_limit=time_limit,
                                  expected_only=expected_only)
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
            for r in results
        )
        sys.exit(1 if any_fail else 0)

    # 普通运行模式
    if len(input_files) == 1:
        with open(input_files[0], "r") as f:
            data = json.load(f)
        run_single_case(data, time_limit, os.path.basename(input_files[0]), strategies,
                        validate=validate)
    else:
        print(f"=== 批量执行 {len(input_files)} 个用例 ===")
        print(f"时间限制: {time_limit}s/用例\n")

        results = []
        for i, filepath in enumerate(input_files, 1):
            print(f"\n{'='*60}")
            print(f"用例 {i}/{len(input_files)}: {os.path.basename(filepath)}")
            print('='*60)

            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                result = run_single_case(data, time_limit, os.path.basename(filepath),
                                         strategies, validate=validate)
                results.append((os.path.basename(filepath), result))
            except Exception as e:
                print(f"❌ 执行失败: {e}")
                results.append((os.path.basename(filepath), None))

        print_run_summary(results)


if __name__ == "__main__":
    main()
