"""
布局验证器

验证所有硬约束是否满足
"""
from typing import List, Tuple
from models import Problem


def validate_layout(problem: Problem,
                    x_coords: List[float],
                    y_coords: List[float],
                    eps: float = 1e-3) -> List[str]:
    """
    验证布局是否满足所有约束

    Returns:
        violations: 违反的约束列表（空列表表示全部满足）
    """
    violations = []
    n = problem.n

    # 1. 检查不重叠
    for i in range(n):
        for j in range(i + 1, n):
            overlap_x = max(0,
                           min(x_coords[i] + problem.widths[i],
                               x_coords[j] + problem.widths[j]) -
                           max(x_coords[i], x_coords[j]))
            overlap_y = max(0,
                           min(y_coords[i] + problem.heights[i],
                               y_coords[j] + problem.heights[j]) -
                           max(y_coords[i], y_coords[j]))

            if overlap_x > eps and overlap_y > eps:
                violations.append(f"重叠: box {i+1} 和 box {j+1}")

    # 2. 检查对称约束
    # X轴对称
    for group in problem.sym_x_groups:
        axis_vals = []

        for pair in group.get("symmetry_pair", []):
            i, j = pair[0] - 1, pair[1] - 1
            cx_i = x_coords[i] + problem.widths[i] / 2
            cx_j = x_coords[j] + problem.widths[j] / 2
            axis_vals.append((cx_i + cx_j) / 2)

        for s in group.get("self_symmetry", []):
            s -= 1
            axis_vals.append(x_coords[s] + problem.widths[s] / 2)

        if axis_vals:
            axis = sum(axis_vals) / len(axis_vals)

            for pair in group.get("symmetry_pair", []):
                i, j = pair[0] - 1, pair[1] - 1
                cx_i = x_coords[i] + problem.widths[i] / 2
                cx_j = x_coords[j] + problem.widths[j] / 2
                mid = (cx_i + cx_j) / 2
                if abs(mid - axis) > eps:
                    violations.append(
                        f"对称x: box {i+1} 和 {j+1} 中心x不一致 (mid={mid:.4f}, axis={axis:.4f})"
                    )

            for s in group.get("self_symmetry", []):
                s -= 1
                cx_s = x_coords[s] + problem.widths[s] / 2
                if abs(cx_s - axis) > eps:
                    violations.append(
                        f"自对称x: box {s+1} 不在轴上 (cx={cx_s:.4f}, axis={axis:.4f})"
                    )

    # Y轴对称
    for group in problem.sym_y_groups:
        axis_vals = []

        for pair in group.get("symmetry_pair", []):
            i, j = pair[0] - 1, pair[1] - 1
            cy_i = y_coords[i] + problem.heights[i] / 2
            cy_j = y_coords[j] + problem.heights[j] / 2
            axis_vals.append((cy_i + cy_j) / 2)

        for s in group.get("self_symmetry", []):
            s -= 1
            axis_vals.append(y_coords[s] + problem.heights[s] / 2)

        if axis_vals:
            axis = sum(axis_vals) / len(axis_vals)

            for pair in group.get("symmetry_pair", []):
                i, j = pair[0] - 1, pair[1] - 1
                cy_i = y_coords[i] + problem.heights[i] / 2
                cy_j = y_coords[j] + problem.heights[j] / 2
                mid = (cy_i + cy_j) / 2
                if abs(mid - axis) > eps:
                    violations.append(
                        f"对称y: box {i+1} 和 {j+1} 中心y不一致 (mid={mid:.4f}, axis={axis:.4f})"
                    )

            for s in group.get("self_symmetry", []):
                s -= 1
                cy_s = y_coords[s] + problem.heights[s] / 2
                if abs(cy_s - axis) > eps:
                    violations.append(
                        f"自对称y: box {s+1} 不在轴上 (cy={cy_s:.4f}, axis={axis:.4f})"
                    )

    # 3. 检查对齐约束
    for group in problem.align_left:
        boxes = [b - 1 for b in group]
        vals = [x_coords[b] for b in boxes]
        if max(vals) - min(vals) > eps:
            violations.append(f"对齐left: boxes {group} 不满足 (range={max(vals)-min(vals):.4f})")

    for group in problem.align_right:
        boxes = [b - 1 for b in group]
        vals = [x_coords[b] + problem.widths[b] for b in boxes]
        if max(vals) - min(vals) > eps:
            violations.append(f"对齐right: boxes {group} 不满足 (range={max(vals)-min(vals):.4f})")

    for group in problem.align_top:
        boxes = [b - 1 for b in group]
        vals = [y_coords[b] + problem.heights[b] for b in boxes]
        if max(vals) - min(vals) > eps:
            violations.append(f"对齐top: boxes {group} 不满足 (range={max(vals)-min(vals):.4f})")

    for group in problem.align_bottom:
        boxes = [b - 1 for b in group]
        vals = [y_coords[b] for b in boxes]
        if max(vals) - min(vals) > eps:
            violations.append(f"对齐bottom: boxes {group} 不满足 (range={max(vals)-min(vals):.4f})")

    # 4. 检查重复组
    for rg in problem.repeat_groups:
        groups = [[b - 1 for b in grp] for grp in rg]
        if len(groups) < 2:
            continue

        ref = groups[0]
        ref_offsets = [(x_coords[b] - x_coords[ref[0]],
                        y_coords[b] - y_coords[ref[0]]) for b in ref]

        for ig in range(1, len(groups)):
            grp = groups[ig]
            grp_offsets = [(x_coords[b] - x_coords[grp[0]],
                            y_coords[b] - y_coords[grp[0]]) for b in grp]

            for j in range(len(ref)):
                dx_ref, dy_ref = ref_offsets[j]
                dx_grp, dy_grp = grp_offsets[j]
                if abs(dx_ref - dx_grp) > eps or abs(dy_ref - dy_grp) > eps:
                    violations.append(
                        f"重复组: 组{ig+1}的box {grp[j]+1} 相对位置与参考组不一致"
                    )

    return violations


def validate_and_report(problem: Problem,
                        x_coords: List[float],
                        y_coords: List[float]) -> Tuple[bool, List[str]]:
    """
    验证并输出报告

    Returns:
        (is_valid, violations)
    """
    violations = validate_layout(problem, x_coords, y_coords)
    is_valid = len(violations) == 0

    if is_valid:
        print("✅ 所有约束满足")
    else:
        print(f"⚠️ 约束违反 ({len(violations)} 条):")
        for v in violations[:20]:
            print(f"  - {v}")
        if len(violations) > 20:
            print(f"  ... 还有 {len(violations) - 20} 条")

    return is_valid, violations
