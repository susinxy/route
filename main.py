"""
主入口

用法:
  python main.py input.json          # 从文件读取输入
  python main.py -t                  # 运行内置测试用例
  python main.py -t --time 30        # 指定时间限制
"""
import json
import sys
import time
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import Problem
from constraint_system import ConstraintSystem
from sa import SimulatedAnnealing
from evaluator import compute_cost, count_overlaps
from validator import validate_and_report


def solve(problem: Problem, time_limit: float = 115.0) -> dict:
    t0 = time.time()

    # 1. 约束化简
    cs = ConstraintSystem(problem)

    # 2. 模拟退火优化
    sa = SimulatedAnnealing(problem, cs, time_limit=time_limit)
    x_coords, y_coords, cost = sa.run()

    # 3. 后处理：去重叠微调
    print("\n后处理: 去重叠微调...")
    x_coords, y_coords = _postprocess(problem, cs, x_coords, y_coords,
                                       remaining_time=time_limit - (time.time() - t0))

    # 4. 验证
    is_valid, violations = validate_and_report(problem, x_coords, y_coords)

    elapsed = time.time() - t0
    _, hpwl, area, overlap = compute_cost(problem, x_coords, y_coords)
    real_cost = 10 * hpwl + area

    # 5. 构建输出
    box_position = []
    for i in range(problem.n):
        box_position.append([round(x_coords[i], 4), round(y_coords[i], 4)])

    result = {
        "box_position": box_position,
        "cost": round(real_cost, 4),
        "hpwl": round(hpwl, 4),
        "area": round(area, 4),
        "overlap": round(overlap, 4),
        "violations": violations,
        "elapsed_seconds": round(elapsed, 2),
        "sa_iterations": sa.iterations,
    }
    return result


def _postprocess(problem, cs, x_coords, y_coords, remaining_time=10.0):
    """
    后处理：在约束满足的前提下，通过微调独立变量消除残余重叠。
    """
    import numpy as np
    from evaluator import compute_overlap_penalty

    n = problem.n
    free_vars = sorted(cs.free_vars)
    num_vars = len(free_vars)

    # 反推当前独立变量值
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

    # 迭代去重叠
    best_values = current_values.copy()
    best_overlap = compute_overlap_penalty(problem, x_coords, y_coords)
    best_x = x_coords.copy()
    best_y = y_coords.copy()

    t0 = time.time()
    max_dim = max(max(problem.widths), max(problem.heights))
    step = max_dim * 0.5

    for iteration in range(10000):
        if time.time() - t0 > min(remaining_time, 15.0):
            break

        new_values = current_values.copy()
        var = free_vars[iteration % num_vars]
        delta = step * (1 if iteration % 2 == 0 else -1) * (0.5 + 0.5 * np.random.random())
        new_values[var] += delta

        new_x, new_y = cs.decode(new_values)
        new_overlap = compute_overlap_penalty(problem, new_x, new_y)

        if new_overlap < best_overlap:
            best_overlap = new_overlap
            best_values = new_values.copy()
            best_x = new_x.copy()
            best_y = new_y.copy()
            current_values = new_values

            if new_overlap < 0.001:
                break

        step *= 0.999

    return best_x, best_y


def generate_test_case() -> dict:
    """
    可行测试用例：
    - 12 个矩形
    - 对称约束（不冲突）
    - 对齐约束（不导致必然重叠）
    - 网络连接
    """
    return {
        "box_size": [
            [6, 4],    # 1
            [3, 5],    # 2
            [4, 2],    # 3
            [4, 2],    # 4
            [8, 6],    # 5
            [8, 6],    # 6
            [6, 6],    # 7
            [6, 6],    # 8
            [4, 2],    # 9
            [4, 2],    # 10
            [5, 3],    # 11
            [8, 4],    # 12
        ],
        # 简化对称约束，避免冲突
        "symmetry_x": [
            {"symmetry_pair": [[5, 6], [3, 4]], "self_symmetry": [1]}
        ],
        "symmetry_y": [
            {"symmetry_pair": [[7, 8]], "self_symmetry": [2]}
        ],
        # 简化对齐约束
        "align": {
            "left": [[1, 3]],
            "right": [],
            "top": [],
            "bottom": [[1, 7, 2]]
        },
        "repeat_groups": [],
        "nets": [
            [1, 2, 3, 4, 5],
            [2, 6, 8],
            [10, 12]
        ]
    }


def main():
    time_limit = 115.0

    if len(sys.argv) < 2:
        print("Usage: python main.py <input.json> | python main.py -t [--time N]")
        sys.exit(1)

    if sys.argv[1] == "-t":
        data = generate_test_case()
        if "--time" in sys.argv:
            idx = sys.argv.index("--time")
            if idx + 1 < len(sys.argv):
                time_limit = float(sys.argv[idx + 1])
    else:
        with open(sys.argv[1], "r") as f:
            data = json.load(f)
        if "--time" in sys.argv:
            idx = sys.argv.index("--time")
            if idx + 1 < len(sys.argv):
                time_limit = float(sys.argv[idx + 1])

    print(f"输入: {len(data['box_size'])} 个矩形, "
          f"{len(data.get('nets', []))} 个网络")
    print(f"时间限制: {time_limit}s")

    problem = Problem(data)
    result = solve(problem, time_limit=time_limit)

    print(f"\n=== 结果 ===")
    print(f"Cost: {result['cost']}")
    print(f"  HPWL: {result['hpwl']}")
    print(f"  Area: {result['area']}")
    print(f"  Overlap: {result['overlap']}")
    print(f"耗时: {result['elapsed_seconds']}s")
    print(f"SA迭代: {result['sa_iterations']}")

    output = {"box_position": result["box_position"]}
    output_path = "output.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n输出已保存到 {output_path}")

    return result


if __name__ == "__main__":
    main()
