"""
route.py - 基于多规则的模拟电路布局算法

核心思路:约束化简 + 模拟退火
1. 将对称/对齐/重复组约束转化为线性方程组,高斯消元降维
2. 在独立变量的低维空间内用 SA 优化 Cost = 10*HPWL + Area
3. 重叠惩罚引导解朝向无重叠区域
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
            A[i, -1] = -eq.const

        pivot_row = 0
        pivot_cols = []

        for col in range(num_vars):
            if pivot_row >= num_eqs:
                break
            max_row = pivot_row
            for row in range(pivot_row + 1, num_eqs):
                if abs(A[row, col]) > abs(A[max_row, col]):
                    max_row = row
            if abs(A[max_row, col]) < 1e-10:
                continue
            A[[pivot_row, max_row]] = A[[max_row, pivot_row]]
            pivot_val = A[pivot_row, col]
            A[pivot_row] /= pivot_val
            for row in range(num_eqs):
                if row != pivot_row and abs(A[row, col]) > 1e-10:
                    A[row] -= A[row, col] * A[pivot_row]
            pivot_cols.append(col)
            pivot_row += 1

        pivot_set = set(pivot_cols)
        self.free_vars = set(range(num_vars)) - pivot_set

        self.var_exprs = [LinearExpr({}, 0.0) for _ in range(num_vars)]
        for fv in self.free_vars:
            self.var_exprs[fv] = LinearExpr({fv: 1.0}, 0.0)

        for i, col in enumerate(pivot_cols):
            expr = LinearExpr({}, A[i, -1])
            for fv in self.free_vars:
                coeff = -A[i, fv]
                if abs(coeff) > 1e-10:
                    expr.coeffs[fv] = coeff
            self.var_exprs[col] = expr

    def decode(self, var_values: Dict[int, float]) -> Tuple[List[float], List[float]]:
        """从独立变量值解码出坐标"""
        n = self.n
        x_coords = [self.var_exprs[i].evaluate(var_values) for i in range(n)]
        y_coords = [self.var_exprs[n + i].evaluate(var_values) for i in range(n)]
        return x_coords, y_coords


# ============================================================
# 评价函数
# ============================================================

def compute_hpwl(problem: Problem, x: List[float], y: List[float]) -> float:
    hpwl = 0.0
    for net in problem.nets:
        if len(net) < 2:
            continue
        cxs = [x[b - 1] + problem.widths[b - 1] / 2 for b in net]
        cys = [y[b - 1] + problem.heights[b - 1] / 2 for b in net]
        hpwl += (max(cxs) - min(cxs)) + (max(cys) - min(cys))
    return hpwl


def compute_area(problem: Problem, x: List[float], y: List[float]) -> float:
    min_x = min(x)
    min_y = min(y)
    max_x = max(x[i] + problem.widths[i] for i in range(problem.n))
    max_y = max(y[i] + problem.heights[i] for i in range(problem.n))
    return (max_x - min_x) * (max_y - min_y)


def compute_overlap_penalty(problem: Problem, x: List[float], y: List[float],
                             margin: float = 0.0) -> float:
    """
    计算重叠惩罚。margin > 0 时，每个 box 四边各扩 margin/2，
    使 SA 在优化过程中主动保持间距，避免微小重叠。
    """
    n = problem.n
    half = margin / 2
    total = 0.0
    for i in range(n):
        wi = problem.widths[i] + margin
        hi = problem.heights[i] + margin
        xi = x[i] - half
        yi = y[i] - half
        for j in range(i + 1, n):
            wj = problem.widths[j] + margin
            hj = problem.heights[j] + margin
            xj = x[j] - half
            yj = y[j] - half
            ox = max(0, min(xi + wi, xj + wj) - max(xi, xj))
            oy = max(0, min(yi + hi, yj + hj) - max(yi, yj))
            total += ox * oy
    return total


def compute_cost(problem: Problem, x: List[float], y: List[float],
                 overlap_lambda: float = 0.0,
                 margin: float = 0.0) -> Tuple[float, float, float, float]:
    """返回 (total_cost, hpwl, area, overlap)"""
    hpwl = compute_hpwl(problem, x, y)
    area = compute_area(problem, x, y)
    overlap = compute_overlap_penalty(problem, x, y, margin=margin)
    total = 10 * hpwl + area + overlap_lambda * overlap
    return total, hpwl, area, overlap


# ============================================================
# 验证器
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
        n = self.problem.n
        self.A_x = np.zeros((n, self.num_vars))
        self.A_y = np.zeros((n, self.num_vars))
        self.b_x = np.zeros(n)
        self.b_y = np.zeros(n)

        for i in range(n):
            expr_x = self.cs.var_exprs[i]
            self.b_x[i] = expr_x.const
            for j, fv in enumerate(self.free_vars):
                self.A_x[i, j] = expr_x.coeffs.get(fv, 0.0)
            expr_y = self.cs.var_exprs[n + i]
            self.b_y[i] = expr_y.const
            for j, fv in enumerate(self.free_vars):
                self.A_y[i, j] = expr_y.coeffs.get(fv, 0.0)

    def _decode_fast(self, var_array: np.ndarray) -> Tuple[List[float], List[float]]:
        x = self.A_x @ var_array + self.b_x
        y = self.A_y @ var_array + self.b_y
        return x.tolist(), y.tolist()

    def _fit_to_target(self, target_x: List[float], target_y: List[float]) -> np.ndarray:
        A = np.vstack([self.A_x, self.A_y])
        b = np.concatenate([np.array(target_x) - self.b_x, np.array(target_y) - self.b_y])
        result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return result

    # ========================================================================
    # Initialization Strategies
    # ========================================================================

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
        target_side = math.sqrt(total_area) * 1.1

        x_range = x_raw.max() - x_raw.min()
        y_range = y_raw.max() - y_raw.min()

        if x_range > 0:
            x_raw = (x_raw - x_raw.min()) / x_range * target_side
        if y_range > 0:
            y_raw = (y_raw - y_raw.min()) / y_range * target_side

        # Legalize: remove overlaps using simple greedy approach
        target_x, target_y = self._legalize_placement(x_raw, y_raw, w, h)

        return self._fit_to_target(target_x, target_y)

    def _legalize_placement(self, x_raw, y_raw, w, h):
        """
        Legalize placement: remove overlaps using greedy row-based packing
        Sort by x-coordinate, then pack into rows
        """
        n = len(x_raw)

        # Sort boxes by x-coordinate
        order = sorted(range(n), key=lambda i: x_raw[i])

        # Pack into rows
        total_area = sum(w[i] * h[i] for i in range(n))
        side = math.sqrt(total_area) * 1.1

        target_x = [0.0] * n
        target_y = [0.0] * n
        xc, yc, rh, gap = 0.0, 0.0, 0.0, 0.5

        for i in order:
            if xc + w[i] > side and xc > 0:
                xc, yc = 0.0, yc + rh + gap
                rh = 0.0
            target_x[i], target_y[i] = xc, yc
            xc += w[i] + gap
            rh = max(rh, h[i])

        return target_x, target_y

    def _perturb(self, var_array: np.ndarray, temperature: float,
                 max_temp: float, layout_scale: float) -> np.ndarray:
        new_vars = var_array.copy()
        ratio = temperature / max_temp

        # Use median box size as perturbation unit
        all_dims = self.problem.widths + self.problem.heights
        box_scale = sorted(all_dims)[len(all_dims) // 2]

        r = random.random()
        if r < 0.15 and self.best_overlap < 0.001:
            # Compression move: scale all vars toward center of mass
            factor = 1.0 - random.uniform(0.005, 0.03)
            center = np.mean(new_vars)
            new_vars = center + (new_vars - center) * factor
        elif r < 0.25:
            # Area-focused move: shrink boundary boxes toward center
            x, y = self._decode_fast(new_vars)
            w, h = self.problem.widths, self.problem.heights
            
            # Find bounding box
            min_x = min(x)
            max_x = max(x[i] + w[i] for i in range(len(x)))
            min_y = min(y)
            max_y = max(y[i] + h[i] for i in range(len(y)))
            
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            
            # Identify boundary boxes (within 20% of edges)
            margin_x = (max_x - min_x) * 0.2
            margin_y = (max_y - min_y) * 0.2
            
            boundary_boxes = []
            for i in range(len(x)):
                box_cx = x[i] + w[i] / 2
                box_cy = y[i] + h[i] / 2
                if (box_cx < min_x + margin_x or box_cx > max_x - margin_x or
                    box_cy < min_y + margin_y or box_cy > max_y - margin_y):
                    boundary_boxes.append(i)
            
            if boundary_boxes:
                # Move 1-2 boundary boxes toward center
                num_to_move = min(random.randint(1, 2), len(boundary_boxes))
                selected = random.sample(boundary_boxes, num_to_move)
                
                shrink_factor = random.uniform(0.1, 0.3)
                target_x_arr = list(x)
                target_y_arr = list(y)
                
                for box_id in selected:
                    # Move toward center
                    target_x_arr[box_id] = x[box_id] + (center_x - x[box_id]) * shrink_factor
                    target_y_arr[box_id] = y[box_id] + (center_y - y[box_id]) * shrink_factor
                
                new_vars = self._fit_to_target(target_x_arr, target_y_arr)
            else:
                self._standard_perturb(new_vars, ratio, box_scale)
        elif r < 0.35:
            # Smart move: move a box toward center of its connected neighbors
            x, y = self._decode_fast(new_vars)
            box_id = random.randint(0, self.problem.n - 1)

            connected = set()
            for net in self.problem.nets:
                if (box_id + 1) in net:
                    connected.update(b - 1 for b in net)
            connected.discard(box_id)

            if connected:
                cx = sum(x[b] + self.problem.widths[b]/2 for b in connected) / len(connected)
                cy = sum(y[b] + self.problem.heights[b]/2 for b in connected) / len(connected)

                target_x = cx - self.problem.widths[box_id]/2
                target_y = cy - self.problem.heights[box_id]/2

                blend = random.uniform(0.1, 0.5)
                target_x_arr = list(x)
                target_y_arr = list(y)
                target_x_arr[box_id] = x[box_id] * (1 - blend) + target_x * blend
                target_y_arr[box_id] = y[box_id] * (1 - blend) + target_y * blend

                new_vars = self._fit_to_target(target_x_arr, target_y_arr)
            else:
                self._standard_perturb(new_vars, ratio, box_scale)
        elif r < 0.30:
            # Swap move: swap positions of two boxes
            x, y = self._decode_fast(new_vars)
            i, j = random.sample(range(self.problem.n), 2)

            target_x = list(x)
            target_y = list(y)
            target_x[i], target_x[j] = x[j], x[i]
            target_y[i], target_y[j] = y[j], y[i]

            new_vars = self._fit_to_target(target_x, target_y)
        else:
            self._standard_perturb(new_vars, ratio, box_scale)

        return new_vars

    def _standard_perturb(self, new_vars, ratio, box_scale):
        num_perturb = max(1, int(self.num_vars * 0.3 * (ratio ** 0.3) + 1))
        num_perturb = min(num_perturb, self.num_vars)
        selected = random.sample(range(self.num_vars), num_perturb)
        scale = box_scale * 2.0 * max(ratio, 0.01)

        for idx in selected:
            delta = scale * random.gauss(0, 1) / max(abs(random.gauss(0, 1)), 0.1)
            delta = max(min(delta, scale * 5), -scale * 5)
            new_vars[idx] += delta

    def _compute_layout_scale(self, x: List[float], y: List[float]) -> float:
        if not x:
            return 100.0
        w, h = self.problem.widths, self.problem.heights
        x_range = max(x[i] + w[i] for i in range(len(x))) - min(x)
        y_range = max(y[i] + h[i] for i in range(len(y))) - min(y)
        return max(x_range, y_range, 1.0)

    # ========================================================================
    # Strategy Registry
    # ========================================================================

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
        strategy_map = {
            'network_aware': self._initial_network_aware,
            'compact_grid': self._initial_compact_grid,
            'random_tight': self._initial_random_tight,
            'clustered': self._initial_clustered,
            'quadratic_placement': self._initial_quadratic_placement,
        }

        if strategy_name not in strategy_map:
            raise ValueError(f"Unknown strategy: {strategy_name}. "
                           f"Available: {self.get_available_strategies()}")

        return lambda: strategy_map[strategy_name](seed_offset)

    def run(self, strategies: List[str] = None) -> Tuple[List[float], List[float], float]:
        """
        Run the simulated annealing optimization

        Args:
            strategies: List of strategy names to use. If None, uses default sequence.
                       Available strategies: network_aware, compact_grid, random_tight,
                                           clustered, quadratic_placement
        """
        random.seed(self.seed)
        np.random.seed(self.seed)
        t0 = time.time()
        self._global_t0 = t0

        # Default strategy sequence if not specified
        if strategies is None:
            strategies = [
                'quadratic_placement',
                'network_aware',
                'compact_grid',
                'clustered',
                'quadratic_placement',
                'random_tight',
            ]

        # Build strategy functions
        init_strategies = []
        for i, strategy_name in enumerate(strategies):
            seed_offset = i * 10  # Different seed for each strategy
            init_strategies.append(self._get_strategy_function(strategy_name, seed_offset))

        round_num = 0
        best_initial_arr = None
        exploration_time = min(30, self.time_limit * 0.25)  # 前 25% 时间用于探索

        # 阶段1:快速探索,找到最好的初始解
        print(f"  [阶段1] 探索多种初始策略 ({exploration_time:.0f}s)...")
        while (time.time() - t0) < exploration_time and round_num < len(init_strategies):
            remaining = exploration_time - (time.time() - t0)
            if remaining < 3:
                break

            strategy = init_strategies[round_num]
            round_time = min(remaining, 8)  # 每轮只用 8s 快速评估
            self._run_round(strategy, round_time, round_num)

            # 记录最好的初始解对应的变量值
            if best_initial_arr is None or self.best_cost < float('inf'):
                best_initial_arr = np.array([self.best_values[fv] for fv in self.free_vars])

            round_num += 1
            elapsed = time.time() - t0
            print(f"    [轮 {round_num}] cost={self.best_cost:.2f}, time={elapsed:.1f}s")

        # 阶段2:深度优化最好的解(单次长轮)
        print(f"  [阶段2] 深度优化最佳初始解...")
        if best_initial_arr is not None:
            remaining = self.time_limit - (time.time() - t0)
            if remaining > 5:
                def use_best_initial():
                    return best_initial_arr.copy()
                self._run_round(use_best_initial, remaining, round_num)
                round_num += 1
                elapsed = time.time() - t0
                print(f"    [深度优化] cost={self.best_cost:.2f}, time={elapsed:.1f}s")

        elapsed = time.time() - t0
        print(f"\nSA 完成: {round_num} 轮, iter={self.iterations}, "
              f"best_cost={self.best_cost:.2f}, overlap={self.best_overlap:.2f}, "
              f"time={elapsed:.1f}s")

        return self.best_x, self.best_y, self.best_cost

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
        alpha = 0.99995
        no_improve = 0

        while (time.time() - local_t0) < time_budget:
            self.iterations += 1

            progress = (time.time() - local_t0) / time_budget
            if current_overlap_m > 0.01:
                # 更强的重叠惩罚（含 margin），迫使快速消除重叠并保持间距
                overlap_lambda = 50000.0 * (1 + progress * 100)
                area_lambda = 0.0
            else:
                overlap_lambda = 0.0
                # Area penalty: increase over time, especially in deep optimization
                # Phase 1 (0-0.3): no area penalty, focus on HPWL
                # Phase 2 (0.3-1.0): gradually increase area penalty
                if progress < 0.3:
                    area_lambda = 0.0
                else:
                    area_lambda = 0.5 + (progress - 0.3) * 1.0  # 0.5 -> 1.2

            new_arr = self._perturb(current_arr, temperature, max_temp, layout_scale)
            new_x, new_y = self._decode_fast(new_arr)
            # 用 margin 版 overlap 做 SA 引导
            new_total, new_hpwl, new_area, new_overlap_m = \
                compute_cost(self.problem, new_x, new_y, overlap_lambda,
                             margin=OVERLAP_MARGIN)
            # 同时算真实 overlap 用于报告
            new_real_overlap = compute_overlap_penalty(self.problem, new_x, new_y)
            new_real_cost = 10 * new_hpwl + new_area + area_lambda * new_area

            current_total = current_real_cost + overlap_lambda * current_overlap_m
            delta = new_total - current_total

            accept = False
            if delta < 0:
                accept = True
            elif temperature > 0.001:
                prob = math.exp(-delta / temperature)
                if random.random() < prob:
                    accept = True

            if accept:
                current_arr = new_arr
                current_real_cost = new_real_cost
                current_x, current_y = new_x, new_y
                current_overlap = new_real_overlap
                current_overlap_m = new_overlap_m
                self.accepted += 1
                no_improve = 0

                if self.iterations % 1000 == 0:
                    layout_scale = self._compute_layout_scale(current_x, current_y)

                self._update_best(current_arr, current_x, current_y,
                                current_real_cost, current_overlap)
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
                self.best_x = x.copy()
                self.best_y = y.copy()
                self.best_overlap = overlap
        elif self.best_overlap >= 0.001 and overlap < self.best_overlap:
            self.best_values = {fv: float(var_arr[j])
                               for j, fv in enumerate(self.free_vars)}
            self.best_x = x.copy()
            self.best_y = y.copy()
            self.best_overlap = overlap
            self.best_cost = real_cost


# ============================================================
# 求解入口
# ============================================================

def _compress_layout(problem: Problem, cs: ConstraintSystem, 
                     x_coords: List[float], y_coords: List[float],
                     remaining_time: float = 5.0) -> Tuple[List[float], List[float]]:
    """
    全局压缩后处理：仅在能改善总成本时应用
    
    策略：
    1. 小幅缩放（0.95-1.05）变量值
    2. 只有当新cost < 原cost时才接受
    3. 迭代尝试直到时间用尽或无改善
    """
    import numpy as np
    t0 = time.time()
    
    n = problem.n
    free_vars = sorted(cs.free_vars)
    num_vars = len(free_vars)
    
    # 反推当前变量值
    A = np.zeros((2 * n, num_vars))
    b_vec = np.zeros(2 * n)
    
    for i in range(n):
        expr_x = cs.var_exprs[i]
        for j, fv in enumerate(free_vars):
            A[i, j] = expr_x.coeffs.get(fv, 0.0)
        b_vec[i] = x_coords[i] - expr_x.const
        
        expr_y = cs.var_exprs[n + i]
        for j, fv in enumerate(free_vars):
            A[n + i, j] = expr_y.coeffs.get(fv, 0.0)
        b_vec[n + i] = y_coords[i] - expr_y.const
    
    result, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
    current_arr = result.copy()
    
    # 计算当前cost
    _, hpwl0, area0, overlap0 = compute_cost(problem, x_coords, y_coords)
    best_cost = 10 * hpwl0 + area0
    best_x = x_coords.copy()
    best_y = y_coords.copy()
    best_arr = current_arr.copy()
    
    print(f"    压缩前: cost={best_cost:.1f}, area={area0:.1f}, hpwl={hpwl0:.1f}, overlap={overlap0:.2f}")
    
    # 改进策略：独立缩放x和y方向，更细粒度搜索
    improved = True
    iteration = 0
    
    while improved and time.time() - t0 < remaining_time * 0.9 and iteration < 30:
        improved = False
        iteration += 1
        
        # 尝试不同缩放比例组合（x和y独立）
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
                
                # 分别缩放x和y方向的变量
                new_arr = current_arr.copy()
                # 简化：统一缩放（因为变量混合了x和y）
                scale = (sx + sy) / 2.0
                centroid = np.mean(current_arr)
                new_arr = centroid + (current_arr - centroid) * scale
                
                new_x, new_y = cs.decode(
                    {fv: float(new_arr[j]) for j, fv in enumerate(free_vars)})
                
                # 检查重叠
                overlap = compute_overlap_penalty(problem, new_x, new_y)
                if overlap > 15.0:  # 允许少量重叠
                    continue
                
                _, new_hpwl, new_area, new_overlap = compute_cost(problem, new_x, new_y)
                new_cost = 10 * new_hpwl + new_area
                
                # 如果改善超过0.5%，记录下来
                if new_cost < best_scale_cost * 0.995:
                    best_scale_cost = new_cost
                    best_scale_combo = (sx, sy)
        
        # 应用最佳缩放
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
    return best_x, best_y


def solve(problem: Problem, time_limit: float = 115.0,
          strategies: List[str] = None) -> Dict[str, Any]:
    """
    求解布局问题。

    Args:
        problem: 布局问题实例
        time_limit: 时间限制(秒)
        strategies: 初始化策略序列,如 ['quadratic_placement', 'network_aware']
                   如果为 None,使用默认策略序列

    Returns:
        dict with keys: box_position, cost, hpwl, area, overlap, violations, elapsed_seconds
    """
    t0 = time.time()

    # 1. 约束化简
    cs = ConstraintSystem(problem)
    print(f"约束化简: {problem.n} boxes -> {len(cs.free_vars)} 独立变量")

    # 2. 模拟退火
    sa = SimulatedAnnealing(problem, cs, time_limit=time_limit)
    x_coords, y_coords, cost = sa.run(strategies=strategies)

    # 3. 后处理去重叠
    remaining = time_limit - (time.time() - t0)
    if remaining > 1.0:
        print("后处理: 去重叠微调...")
        x_coords, y_coords = _postprocess(problem, cs, x_coords, y_coords,
                                           remaining_time=remaining)

    # 4. 全局压缩后处理
    remaining = time_limit - (time.time() - t0)
    if remaining > 2.0:
        print("后处理: 全局压缩...")
        x_coords, y_coords = _compress_layout(problem, cs, x_coords, y_coords,
                                              remaining_time=remaining)

    # 5. 验证
    violations = validate_layout(problem, x_coords, y_coords)
    elapsed = time.time() - t0

    _, hpwl, area, overlap = compute_cost(problem, x_coords, y_coords)
    real_cost = 10 * hpwl + area

    if not violations:
        print("✅ 所有约束满足")
    else:
        print(f"⚠️ 约束违反 ({len(violations)} 条)")

    box_position = [[round(x_coords[i], 4), round(y_coords[i], 4)]
                    for i in range(problem.n)]

    return {
        "box_position": box_position,
        "cost": round(real_cost, 4),
        "hpwl": round(hpwl, 4),
        "area": round(area, 4),
        "overlap": round(overlap, 4),
        "violations": violations,
        "elapsed_seconds": round(elapsed, 2),
    }


def _postprocess(problem: Problem, cs: ConstraintSystem,
                 x_coords: List[float], y_coords: List[float],
                 remaining_time: float = 10.0) -> Tuple[List[float], List[float]]:
    """后处理:在约束满足前提下微调消除残余重叠"""
    n = problem.n
    free_vars = sorted(cs.free_vars)
    num_vars = len(free_vars)

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
    # 用 margin 版 overlap 做引导，让后处理也主动保持间距
    best_overlap_m = compute_overlap_penalty(problem, x_coords, y_coords,
                                              margin=OVERLAP_MARGIN)
    best_real_overlap = compute_overlap_penalty(problem, x_coords, y_coords)
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
        new_overlap_m = compute_overlap_penalty(problem, new_x, new_y,
                                                 margin=OVERLAP_MARGIN)
        new_real_overlap = compute_overlap_penalty(problem, new_x, new_y)

        if new_overlap_m < best_overlap_m:
            best_overlap_m = new_overlap_m
            best_real_overlap = new_real_overlap
            best_values = new_values.copy()
            best_x = new_x.copy()
            best_y = new_y.copy()
            current_values = new_values
            if new_real_overlap < 0.001:
                break
        step *= 0.999

    return best_x, best_y
