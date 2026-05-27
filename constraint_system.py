"""
约束化简系统

核心思想：
将每个矩形的左下角坐标 (x_i, y_i) 表示为一组独立自由变量的线性函数：
  x_i = sum(a_ij * v_j) + c_i
  y_i = sum(b_ij * v_j) + d_i

其中 v_j 是独立变量（对称轴位置、对齐基准、重复组平移、自由坐标等）。

通过高斯消元，将强约束（对称、对齐、重复组）转化为线性等式，
从而把 2n 个坐标变量降维到几十个独立变量。
"""
import numpy as np
from typing import List, Dict, Tuple, Set
from models import Problem


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
    约束化简系统

    变量编号约定：
    - 0 到 n-1: x_i (矩形 i 的 x 坐标，0-indexed)
    - n 到 2n-1: y_i (矩形 i 的 y 坐标，0-indexed)
    - 2n 及以后: 辅助变量（对称轴、对齐基准、重复组平移等）
    """

    def __init__(self, problem: Problem):
        self.problem = problem
        self.n = problem.n

        # 每个变量表达为：var_i = LinearExpr(free_vars)
        # 初始时 var_i = v_i（每个变量都是自由的）
        self.var_exprs: List[LinearExpr] = [
            LinearExpr({i: 1.0}, 0.0) for i in range(2 * self.n)
        ]

        # 等式约束列表: (var_a, var_b) 表示 var_a = var_b
        # 或更一般地：LinearExpr = 0
        self.equations: List[LinearExpr] = []

        # 辅助变量计数器
        self.next_var = 2 * self.n

        # 独立变量集合
        self.free_vars: Set[int] = set(range(2 * self.n))

        # 解析所有约束
        self._parse_constraints()

        # 高斯消元求解
        self._solve()

    def _new_var(self) -> int:
        """分配新的辅助变量"""
        var_id = self.next_var
        self.next_var += 1
        self.var_exprs.append(LinearExpr({var_id: 1.0}, 0.0))
        return var_id

    def _add_equation(self, expr: LinearExpr):
        """添加等式约束: expr = 0"""
        self.equations.append(expr)

    def _add_equality(self, expr_a: LinearExpr, expr_b: LinearExpr):
        """添加等式约束: expr_a = expr_b，即 expr_a - expr_b = 0"""
        self._add_equation(expr_a - expr_b)

    def _parse_constraints(self):
        """解析所有约束，生成等式"""
        self._parse_symmetry()
        self._parse_alignment()
        self._parse_repeat_groups()

    def _parse_symmetry(self):
        """解析对称约束"""
        p = self.problem
        n = self.n

        # X轴对称（约束 x 坐标）
        for group in p.sym_x_groups:
            # 引入对称轴变量
            axis_var = self._new_var()
            axis_expr = LinearExpr({axis_var: 1.0}, 0.0)

            # symmetry_pair: 两个矩形关于 x 轴对称
            for pair in group.get("symmetry_pair", []):
                i, j = pair[0] - 1, pair[1] - 1  # 转 0-indexed
                # (x_i + w_i/2) + (x_j + w_j/2) = 2 * axis
                cx_i = self.var_exprs[i] + p.widths[i] / 2
                cx_j = self.var_exprs[j] + p.widths[j] / 2
                self._add_equation(cx_i + cx_j - axis_expr * 2)

            # self_symmetry: 矩形自身关于 x 轴对称
            for s in group.get("self_symmetry", []):
                s -= 1  # 0-indexed
                # x_s + w_s/2 = axis
                cx_s = self.var_exprs[s] + p.widths[s] / 2
                self._add_equation(cx_s - axis_expr)

        # Y轴对称（约束 y 坐标）
        for group in p.sym_y_groups:
            axis_var = self._new_var()
            axis_expr = LinearExpr({axis_var: 1.0}, 0.0)

            for pair in group.get("symmetry_pair", []):
                i, j = pair[0] - 1, pair[1] - 1
                # (y_i + h_i/2) + (y_j + h_j/2) = 2 * axis
                cy_i = self.var_exprs[n + i] + p.heights[i] / 2
                cy_j = self.var_exprs[n + j] + p.heights[j] / 2
                self._add_equation(cy_i + cy_j - axis_expr * 2)

            for s in group.get("self_symmetry", []):
                s -= 1
                cy_s = self.var_exprs[n + s] + p.heights[s] / 2
                self._add_equation(cy_s - axis_expr)

    def _parse_alignment(self):
        """解析对齐约束"""
        p = self.problem
        n = self.n

        # 左对齐: x_i = x_j
        for group in p.align_left:
            boxes = [b - 1 for b in group]
            for i in range(1, len(boxes)):
                self._add_equation(self.var_exprs[boxes[0]] - self.var_exprs[boxes[i]])

        # 右对齐: x_i + w_i = x_j + w_j
        for group in p.align_right:
            boxes = [b - 1 for b in group]
            for i in range(1, len(boxes)):
                a, b = boxes[0], boxes[i]
                expr_a = self.var_exprs[a] + p.widths[a]
                expr_b = self.var_exprs[b] + p.widths[b]
                self._add_equation(expr_a - expr_b)

        # 下对齐: y_i = y_j
        for group in p.align_bottom:
            boxes = [b - 1 for b in group]
            for i in range(1, len(boxes)):
                self._add_equation(
                    self.var_exprs[n + boxes[0]] - self.var_exprs[n + boxes[i]]
                )

        # 上对齐: y_i + h_i = y_j + h_j
        for group in p.align_top:
            boxes = [b - 1 for b in group]
            for i in range(1, len(boxes)):
                a, b = boxes[0], boxes[i]
                expr_a = self.var_exprs[n + a] + p.heights[a]
                expr_b = self.var_exprs[n + b] + p.heights[b]
                self._add_equation(expr_a - expr_b)

    def _parse_repeat_groups(self):
        """解析重复组约束"""
        p = self.problem
        n = self.n

        for rg in p.repeat_groups:
            groups = [[b - 1 for b in grp] for grp in rg]
            if len(groups) < 2:
                continue

            ref_group = groups[0]

            # 其他组与参考组的相对位置相同
            for g_idx in range(1, len(groups)):
                group = groups[g_idx]

                # 引入平移变量 dx, dy
                dx_var = self._new_var()
                dy_var = self._new_var()
                dx_expr = LinearExpr({dx_var: 1.0}, 0.0)
                dy_expr = LinearExpr({dy_var: 1.0}, 0.0)

                # 对应矩形的坐标差相同
                for i, (ref_box, box) in enumerate(zip(ref_group, group)):
                    # x_box - x_ref = dx
                    self._add_equation(
                        self.var_exprs[box] - self.var_exprs[ref_box] - dx_expr
                    )
                    # y_box - y_ref = dy
                    self._add_equation(
                        self.var_exprs[n + box] - self.var_exprs[n + ref_box] - dy_expr
                    )

    def _solve(self):
        """高斯消元求解线性方程组"""
        if not self.equations:
            return

        # 构建增广矩阵
        # 变量总数：next_var
        num_vars = self.next_var
        num_eqs = len(self.equations)

        # 矩阵 A * x = b
        A = np.zeros((num_eqs, num_vars + 1))

        for i, eq in enumerate(self.equations):
            for var_id, coeff in eq.coeffs.items():
                A[i, var_id] = coeff
            A[i, -1] = -eq.const  # 移到右边

        # 高斯消元（带部分主元选择）
        pivot_row = 0
        pivot_cols = []

        for col in range(num_vars):
            if pivot_row >= num_eqs:
                break

            # 找主元
            max_row = pivot_row
            for row in range(pivot_row + 1, num_eqs):
                if abs(A[row, col]) > abs(A[max_row, col]):
                    max_row = row

            if abs(A[max_row, col]) < 1e-10:
                continue  # 该列全为 0

            # 交换行
            A[[pivot_row, max_row]] = A[[max_row, pivot_row]]

            # 归一化
            pivot_val = A[pivot_row, col]
            A[pivot_row] /= pivot_val

            # 消元
            for row in range(num_eqs):
                if row != pivot_row and abs(A[row, col]) > 1e-10:
                    factor = A[row, col]
                    A[row] -= factor * A[pivot_row]

            pivot_cols.append(col)
            pivot_row += 1

        # 提取独立变量（非主元列）
        pivot_set = set(pivot_cols)
        self.free_vars = set(range(num_vars)) - pivot_set

        # 构建每个变量的表达式（用独立变量表示）
        # 对于主元变量：var_pivot = -sum(A[pivot_row, free_var] * free_var) + A[pivot_row, -1]
        self.var_exprs = [LinearExpr({}, 0.0) for _ in range(num_vars)]

        # 独立变量表达为自身
        for fv in self.free_vars:
            self.var_exprs[fv] = LinearExpr({fv: 1.0}, 0.0)

        # 主元变量用独立变量表达
        for i, col in enumerate(pivot_cols):
            expr = LinearExpr({}, A[i, -1])
            for fv in self.free_vars:
                coeff = -A[i, fv]
                if abs(coeff) > 1e-10:
                    expr.coeffs[fv] = coeff
            self.var_exprs[col] = expr

        print(f"约束化简完成:")
        print(f"  原始变量数: {2 * self.n} (x_i, y_i)")
        print(f"  辅助变量数: {self.next_var - 2 * self.n}")
        print(f"  总变量数: {num_vars}")
        print(f"  等式约束数: {num_eqs}")
        print(f"  独立变量数: {len(self.free_vars)}")

    def decode(self, var_values: Dict[int, float]) -> Tuple[List[float], List[float]]:
        """
        从独立变量值解码出所有矩形的坐标

        Args:
            var_values: 独立变量的取值 {var_id: value}

        Returns:
            (x_coords, y_coords): 每个矩形的左下角坐标
        """
        n = self.n
        x_coords = []
        y_coords = []

        for i in range(n):
            x = self.var_exprs[i].evaluate(var_values)
            y = self.var_exprs[n + i].evaluate(var_values)
            x_coords.append(x)
            y_coords.append(y)

        return x_coords, y_coords

    def get_initial_values(self) -> Dict[int, float]:
        """生成初始独立变量值（简单构造）"""
        # 简单策略：所有变量设为 0，然后逐步调整避免重叠
        values = {fv: 0.0 for fv in self.free_vars}

        # 对角线放置：矩形 i 放在 (i * 100, i * 100)
        # 需要根据哪些变量对应哪些坐标来设置
        # 这里简化：先设为 0，后续由 SA 优化
        return values

    def get_variable_ranges(self) -> Dict[int, Tuple[float, float]]:
        """
        估计每个独立变量的合理范围
        用于 SA 扰动时选择合适的步长
        """
        ranges = {}
        max_dim = max(max(self.problem.widths), max(self.problem.heights))
        n = self.problem.n

        # 粗略估计：坐标范围大约在 [0, n * max_dim]
        max_coord = n * max_dim

        for fv in self.free_vars:
            ranges[fv] = (-max_coord, max_coord)

        return ranges
