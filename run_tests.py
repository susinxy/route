"""
run_tests.py - 全面约束验证测试脚本

功能:
  1. 验证每个用例的 expected_output 是否满足所有约束
  2. 运行算法，验证算法输出是否满足所有约束
  3. 对比 expected_output 和算法输出的 cost

用法:
  python run_tests.py                          # 跑所有用例
  python run_tests.py case01                   # 跑指定用例
  python run_tests.py --time 60                # 自定义时间限制
  python run_tests.py --expected-only          # 只验证 expected_output
  python run_tests.py --list                   # 列出所有用例
"""
import json
import sys
import os
import glob
import time
from typing import Dict, Any, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from route import Problem, solve, validate_layout, compute_cost, compute_hpwl, compute_area


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
# 约束验证 (扩展版，带详细报告)
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
                # y 坐标也要相同
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
                # x 坐标也要相同
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
# 运行单个用例的测试
# ============================================================

def run_case_test(filepath: str, time_limit: float = 115.0,
                  expected_only: bool = False) -> Dict[str, Any]:
    """
    对单个用例做完整测试:
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
        result["expected_valid"] = None  # 无 expected_output

    if expected_only:
        return result

    # --- 2. 运行算法 ---
    problem = Problem(input_data)
    t0 = time.time()
    solve_result = solve(problem, time_limit=time_limit)
    elapsed = time.time() - t0

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
    """打印单个用例的结果"""
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

    if verbose:
        print(f"   ─── 约束详情 ───")
        for key in ["expected", "algo"]:
            vkey = f"{key}_violations"
            if vkey in r:
                print(f"   [{key}] violations: {len(r[vkey])}")

    print(f"{'='*70}")


def print_summary(results: List[Dict[str, Any]]):
    """打印汇总"""
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


# ============================================================
# Main
# ============================================================

def main():
    time_limit = 115.0
    filter_name = None
    expected_only = False
    verbose = False
    list_only = False

    i = 1
    positional = []
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--time" and i + 1 < len(sys.argv):
            time_limit = float(sys.argv[i + 1])
            i += 2
        elif arg == "--expected-only":
            expected_only = True
            i += 1
        elif arg == "--verbose" or arg == "-v":
            verbose = True
            i += 1
        elif arg == "--list":
            list_only = True
            i += 1
        else:
            positional.append(arg)
            i += 1

    if positional:
        filter_name = positional[0]

    cases = discover_cases(filter_name)
    if not cases:
        print(f"未找到测试用例 (filter={filter_name})")
        print(f"用例目录: {TEST_CASES_DIR}")
        sys.exit(1)

    if list_only:
        print("可用测试用例:")
        for f in cases:
            data = load_case(f)
            name = os.path.basename(f).replace(".json", "")
            desc = data.get("description", "")
            print(f"  {name}: {desc}")
        return

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
            r = run_case_test(filepath, time_limit=time_limit, expected_only=expected_only)
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

    # 退出码: 有任何失败就返回 1
    any_fail = any(
        r.get("expected_valid") is False or r.get("algo_valid") is False
        for r in results
    )
    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
