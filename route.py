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
        
        # 验证重复组的尺寸一致性
        self._validate_repeat_groups()
    
    def _validate_repeat_groups(self):
        """
        验证重复组的尺寸一致性。
        
        重复组约束 [[a, b, ...], [c, d, ...]] 要求：
        - 对应位置的 box 尺寸必须相同：a与c、b与d、...
        - 组内的相对位置偏移相同
        
        如果尺寸不一致，说明输入数据存在根本矛盾，无解。
        """
        for rg in self.repeat_groups:
            if len(rg) < 2:
                continue
            
            ref_group = rg[0]  # 参考组
            for g_idx in range(1, len(rg)):
                group = rg[g_idx]  # 待检查组
                
                if len(ref_group) != len(group):
                    raise ValueError(
                        f"重复组结构错误：参考组 {ref_group} 有 {len(ref_group)} 个元素，"
                        f"但组 {group} 有 {len(group)} 个元素，长度不一致"
                    )
                
                # 检查对应位置的 box 尺寸是否相同
                for pos, (ref_box_id, box_id) in enumerate(zip(ref_group, group)):
                    ref_w = self.widths[ref_box_id - 1]
                    ref_h = self.heights[ref_box_id - 1]
                    box_w = self.widths[box_id - 1]
                    box_h = self.heights[box_id - 1]
                    
                    if abs(ref_w - box_w) > 1e-6 or abs(ref_h - box_h) > 1e-6:
                        raise ValueError(
                            f"重复组尺寸矛盾：参考组的 box {ref_box_id} 尺寸为 [{ref_w}, {ref_h}]，"
                            f"但组 {g_idx} 的 box {box_id} 尺寸为 [{box_w}, {box_h}]。"
                            f"重复组要求对应位置的 box 尺寸必须相同，此问题无解。"
                        )


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

def compute_hpwl(problem: Problem, x: List[float], y: List[float]) -> float:
    total = 0.0
    for net in problem.nets:
        if len(net) < 2:
            continue
        boxes = [b - 1 for b in net]
        min_x = min(x[b] + problem.widths[b] / 2 for b in boxes)
        max_x = max(x[b] + problem.widths[b] / 2 for b in boxes)
        min_y = min(y[b] + problem.heights[b] / 2 for b in boxes)
        max_y = max(y[b] + problem.heights[b] / 2 for b in boxes)
        total += (max_x - min_x) + (max_y - min_y)
    return total


def compute_area(problem: Problem, x: List[float], y: List[float]) -> float:
    if not x:
        return 0.0
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

    def _decode_fast(self, var_array: np.ndarray) -> Tuple[List[float], List[float]]:
        """快速解码: 矩阵运算"""
        x = self.x_coeffs @ var_array + self.x_const
        y = self.y_coeffs @ var_array + self.y_const
        return x.tolist(), y.tolist()

    def _compute_layout_scale(self, x: List[float], y: List[float]) -> float:
        """计算布局尺度，用于自适应扰动"""
        if not x:
            return 1.0
        x_range = max(x) - min(x) + max(self.problem.widths)
        y_range = max(y) - min(y) + max(self.problem.heights)
        return max(x_range, y_range, 1.0)

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

    def run(self, strategies: List[str] = None) -> Tuple[List[float], List[float], float]:
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

        # Build init functions
        init_strategies = []
        for i, strategy_name in enumerate(strategies):
            # Use different seeds for different rounds
            seed_offset = i * 1000
            init_strategies.append(self._get_strategy_function(strategy_name, seed_offset))

        # Phase 1: Try multiple initial strategies
        exploration_time = min(self.time_limit * 0.25, 15.0)
        round_num = 0

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

        # Phase 2: Deep optimization from best solution
        remaining = self.time_limit - (time.time() - t0)
        if remaining > 1.0 and self.best_cost < float('inf'):
            round_num += 1
            elapsed = time.time() - t0
            print(f"  [阶段2] 深度优化最佳初始解...")
            self._run_round(
                lambda: np.array([self.best_values[fv] for fv in self.free_vars]),
                remaining, round_num
            )
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
            if self.best_x:
                x, y = self._decode_fast(new_vars)
                cx = (min(x) + max(x)) / 2
                cy = (min(y) + max(y)) / 2
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


def _compress_layout(problem: Problem, cs: ConstraintSystem, 
                     x_coords: List[float], y_coords: List[float],
                     remaining_time: float = 5.0) -> Tuple[List[float], List[float]]:
    """
    全局压缩后处理：仅在能改善总成本时应用
    """
    free_vars = sorted(cs.free_vars)
    num_vars = len(free_vars)

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
        
        # 尝试多种缩放比例
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
