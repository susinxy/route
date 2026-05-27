"""
模拟退火优化器 v5

关键改进：
1. 更多轮次重启（每轮 15-25s），避免陷入局部最优
2. 紧凑初始解：使用 bin-packing 启发式
3. 更大扰动幅度：允许跳出局部最优
4. 针对 Area 的优化：鼓励紧凑布局
"""
import random
import math
import time
from typing import List, Dict, Tuple
import numpy as np
from models import Problem
from constraint_system import ConstraintSystem
from evaluator import compute_cost, compute_overlap_penalty


class SimulatedAnnealing:
    def __init__(self, problem: Problem, cs: ConstraintSystem,
                 time_limit: float = 110.0, seed: int = 42):
        self.problem = problem
        self.cs = cs
        self.time_limit = time_limit
        self.seed = seed

        self.free_vars = sorted(cs.free_vars)
        self.num_vars = len(self.free_vars)

        # 预计算系数矩阵
        self._precompute_matrices()

        self.best_values: Dict[int, float] = {}
        self.best_cost = float('inf')
        self.best_x: List[float] = []
        self.best_y: List[float] = []
        self.best_overlap = float('inf')

        self.iterations = 0
        self.accepted = 0

    def _precompute_matrices(self):
        """预计算系数矩阵"""
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
        """快速解码"""
        x = self.A_x @ var_array + self.b_x
        y = self.A_y @ var_array + self.b_y
        return x.tolist(), y.tolist()

    def _fit_to_target(self, target_x: List[float], target_y: List[float]) -> np.ndarray:
        """最小二乘拟合"""
        A = np.vstack([self.A_x, self.A_y])
        b = np.concatenate([
            np.array(target_x) - self.b_x,
            np.array(target_y) - self.b_y
        ])
        result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return result

    def _initial_compact_grid(self, seed_offset=0) -> np.ndarray:
        """紧凑网格布局"""
        n = self.problem.n
        w = self.problem.widths
        h = self.problem.heights

        # 估算紧凑尺寸
        total_area = sum(w[i] * h[i] for i in range(n))
        side = math.sqrt(total_area) * 1.1  # 只留 10% 空间

        # 按高度排序（更易紧密排列）
        order = sorted(range(n), key=lambda i: h[i], reverse=True)
        random.seed(self.seed + seed_offset)

        target_x = [0.0] * n
        target_y = [0.0] * n
        x_cursor = 0.0
        y_cursor = 0.0
        row_height = 0.0
        gap = 0.5  # 极小间距

        for i in order:
            if x_cursor + w[i] > side and x_cursor > 0:
                x_cursor = 0.0
                y_cursor += row_height + gap
                row_height = 0.0
            target_x[i] = x_cursor
            target_y[i] = y_cursor
            x_cursor += w[i] + gap
            row_height = max(row_height, h[i])

        return self._fit_to_target(target_x, target_y)

    def _initial_random_tight(self, seed_offset=0) -> np.ndarray:
        """随机紧凑布局"""
        n = self.problem.n
        w = self.problem.widths
        h = self.problem.heights

        random.seed(self.seed + seed_offset + 200)
        total_area = sum(w[i] * h[i] for i in range(n))
        side = math.sqrt(total_area) * 1.2

        order = list(range(n))
        random.shuffle(order)

        target_x = [0.0] * n
        target_y = [0.0] * n
        x_cursor = 0.0
        y_cursor = 0.0
        row_height = 0.0

        for i in order:
            if x_cursor + w[i] > side and x_cursor > 0:
                x_cursor = random.uniform(0, 2)
                y_cursor += row_height + random.uniform(0.5, 2)
                row_height = 0.0
            target_x[i] = x_cursor
            target_y[i] = y_cursor
            x_cursor += w[i] + random.uniform(0.2, 1.5)
            row_height = max(row_height, h[i])

        return self._fit_to_target(target_x, target_y)

    def _initial_clustered(self, seed_offset=0) -> np.ndarray:
        """聚类布局：把相关网络聚一起"""
        n = self.problem.n
        w = self.problem.widths
        h = self.problem.heights

        # 简单策略：按网络聚类
        net_boxes = set()
        for net in self.problem.nets:
            for b in net:
                net_boxes.add(b - 1)

        # 先放网络相关的，再放其他的
        order = sorted(range(n), key=lambda i: (i not in net_boxes, -w[i] * h[i]))

        total_area = sum(w[i] * h[i] for i in range(n))
        side = math.sqrt(total_area) * 1.15

        target_x = [0.0] * n
        target_y = [0.0] * n
        x_cursor = 0.0
        y_cursor = 0.0
        row_height = 0.0
        gap = 0.3

        for i in order:
            if x_cursor + w[i] > side and x_cursor > 0:
                x_cursor = 0.0
                y_cursor += row_height + gap
                row_height = 0.0
            target_x[i] = x_cursor
            target_y[i] = y_cursor
            x_cursor += w[i] + gap
            row_height = max(row_height, h[i])

        return self._fit_to_target(target_x, target_y)

    def _perturb(self, var_array: np.ndarray, temperature: float,
                 max_temp: float, layout_scale: float) -> np.ndarray:
        """扰动"""
        new_vars = var_array.copy()
        ratio = temperature / max_temp

        # 扰动变量数：高温多变量，低温少变量
        num_perturb = max(1, int(self.num_vars * 0.3 * (ratio ** 0.3) + 1))
        num_perturb = min(num_perturb, self.num_vars)

        selected = random.sample(range(self.num_vars), num_perturb)

        # 扰动幅度：与布局尺寸匹配，允许大幅跳跃
        scale = layout_scale * 0.5 * max(ratio, 0.01)

        for idx in selected:
            # 使用 Cauchy 分布（重尾，更容易跳出局部最优）
            delta = scale * random.gauss(0, 1) / max(abs(random.gauss(0, 1)), 0.1)
            delta = max(min(delta, scale * 3), -scale * 3)  # 截断
            new_vars[idx] += delta

        return new_vars

    def _compute_layout_scale(self, x: List[float], y: List[float]) -> float:
        """计算布局尺寸"""
        if not x:
            return 100.0
        w = self.problem.widths
        h = self.problem.heights
        x_range = max(x[i] + w[i] for i in range(len(x))) - min(x)
        y_range = max(y[i] + h[i] for i in range(len(y))) - min(y)
        return max(x_range, y_range, 1.0)

    def run(self) -> Tuple[List[float], List[float], float]:
        """运行 SA"""
        random.seed(self.seed)
        np.random.seed(self.seed)
        t0 = time.time()
        self._global_t0 = t0

        # 更多初始解策略
        init_strategies = [
            lambda: self._initial_compact_grid(0),
            lambda: self._initial_clustered(0),
            lambda: self._initial_random_tight(0),
            lambda: self._initial_compact_grid(10),
            lambda: self._initial_random_tight(10),
            lambda: self._initial_clustered(10),
        ]

        round_num = 0
        while (time.time() - t0) < self.time_limit:
            remaining = self.time_limit - (time.time() - t0)
            if remaining < 3:
                break

            # 选择策略
            if round_num < len(init_strategies):
                strategy = init_strategies[round_num]
            else:
                # 从最优解重启（加扰动）
                def restart_from_best():
                    arr = np.array([self.best_values[fv] for fv in self.free_vars])
                    scale = self._compute_layout_scale(self.best_x, self.best_y) * 0.1
                    arr += np.random.normal(0, scale, self.num_vars)
                    return arr
                strategy = restart_from_best

            # 每轮 15-25 秒
            round_time = min(remaining, random.uniform(15, 25))
            self._run_round(strategy, round_time, round_num)
            round_num += 1

            elapsed = time.time() - t0
            print(f"  [轮 {round_num}] cost={self.best_cost:.2f}, "
                  f"overlap={self.best_overlap:.2f}, time={elapsed:.1f}s")

        elapsed = time.time() - t0
        print(f"\nSA 完成: {round_num} 轮, iter={self.iterations}, "
              f"best_cost={self.best_cost:.2f}, overlap={self.best_overlap:.2f}, "
              f"time={elapsed:.1f}s")

        return self.best_x, self.best_y, self.best_cost

    def _run_round(self, init_fn, time_budget, round_num):
        """单轮 SA"""
        local_t0 = time.time()
        global_t0 = self._global_t0

        # 初始解
        current_arr = init_fn()
        current_x, current_y = self._decode_fast(current_arr)

        # 初始评价
        _, current_hpwl, current_area, current_overlap = \
            compute_cost(self.problem, current_x, current_y)
        current_real_cost = 10 * current_hpwl + current_area

        layout_scale = self._compute_layout_scale(current_x, current_y)

        self._update_best(current_arr, current_x, current_y,
                         current_real_cost, current_overlap)

        # SA 参数
        max_temp = layout_scale * 100.0  # 更高初始温度
        temperature = max_temp
        alpha = 0.99995  # 更快降温（配合短轮次）

        no_improve = 0

        while (time.time() - local_t0) < time_budget:
            self.iterations += 1

            # 自适应 λ
            progress = (time.time() - local_t0) / time_budget
            if current_overlap > 0.01:
                overlap_lambda = 5000.0 * (1 + progress * 50)
            else:
                overlap_lambda = 0.0

            # 扰动
            new_arr = self._perturb(current_arr, temperature, max_temp, layout_scale)
            new_x, new_y = self._decode_fast(new_arr)

            # 评价
            new_total, new_hpwl, new_area, new_overlap = \
                compute_cost(self.problem, new_x, new_y, overlap_lambda)
            new_real_cost = 10 * new_hpwl + new_area

            current_total = current_real_cost + overlap_lambda * current_overlap
            delta = new_total - current_total

            # 接受准则
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
                current_overlap = new_overlap
                self.accepted += 1
                no_improve = 0

                if self.iterations % 1000 == 0:
                    layout_scale = self._compute_layout_scale(current_x, current_y)

                self._update_best(current_arr, current_x, current_y,
                                current_real_cost, current_overlap)
            else:
                no_improve += 1

            # 降温
            temperature *= alpha

            # 重加热
            if no_improve > 8000:
                temperature = max(max_temp * 0.2, 1.0)
                no_improve = 0

            # 定期输出
            if self.iterations % 100000 == 0:
                elapsed = time.time() - global_t0
                print(f"    iter={self.iterations}, T={temperature:.1f}, "
                      f"cost={self.best_cost:.2f}, ov={self.best_overlap:.2f}, "
                      f"time={elapsed:.1f}s")

    def _update_best(self, var_arr, x, y, real_cost, overlap):
        """更新全局最优"""
        if overlap < 0.01:
            if real_cost < self.best_cost or self.best_overlap >= 0.01:
                self.best_cost = real_cost
                self.best_values = {fv: float(var_arr[j]) for j, fv in enumerate(self.free_vars)}
                self.best_x = x.copy()
                self.best_y = y.copy()
                self.best_overlap = overlap
        elif self.best_overlap >= 0.01 and overlap < self.best_overlap:
            self.best_values = {fv: float(var_arr[j]) for j, fv in enumerate(self.free_vars)}
            self.best_x = x.copy()
            self.best_y = y.copy()
            self.best_overlap = overlap
            self.best_cost = real_cost
