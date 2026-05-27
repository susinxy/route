"""
评价函数

计算布局的 Cost = 10 × HPWL + Area + λ × OverlapPenalty
"""
from typing import List, Tuple
from models import Problem


def compute_hpwl(problem: Problem, x_coords: List[float], y_coords: List[float]) -> float:
    """计算所有网络的半周长线长之和"""
    hpwl = 0.0
    for net in problem.nets:
        if len(net) < 2:
            continue

        # 获取网络中所有矩形的中心坐标
        centers_x = []
        centers_y = []
        for box_idx in net:
            i = box_idx - 1  # 转 0-indexed
            cx = x_coords[i] + problem.widths[i] / 2
            cy = y_coords[i] + problem.heights[i] / 2
            centers_x.append(cx)
            centers_y.append(cy)

        # HPWL = (max_x - min_x) + (max_y - min_y)
        net_hpwl = (max(centers_x) - min(centers_x)) + (max(centers_y) - min(centers_y))
        hpwl += net_hpwl

    return hpwl


def compute_area(problem: Problem, x_coords: List[float], y_coords: List[float]) -> float:
    """计算包围所有矩形的最小矩形面积"""
    min_x = min(x_coords)
    min_y = min(y_coords)
    max_x = max(x_coords[i] + problem.widths[i] for i in range(problem.n))
    max_y = max(y_coords[i] + problem.heights[i] for i in range(problem.n))

    return (max_x - min_x) * (max_y - min_y)


def compute_overlap_penalty(problem: Problem, x_coords: List[float], y_coords: List[float]) -> float:
    """
    计算所有矩形对的重叠面积之和

    两个矩形的重叠面积 = max(0, overlap_x) * max(0, overlap_y)
    """
    n = problem.n
    total_overlap = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            # 矩形 i 的范围: [x_i, x_i + w_i] × [y_i, y_i + h_i]
            # 矩形 j 的范围: [x_j, x_j + w_j] × [y_j, y_j + h_j]

            # X 方向重叠
            overlap_x = max(0,
                           min(x_coords[i] + problem.widths[i],
                               x_coords[j] + problem.widths[j]) -
                           max(x_coords[i], x_coords[j]))

            # Y 方向重叠
            overlap_y = max(0,
                           min(y_coords[i] + problem.heights[i],
                               y_coords[j] + problem.heights[j]) -
                           max(y_coords[i], y_coords[j]))

            # 重叠面积
            total_overlap += overlap_x * overlap_y

    return total_overlap


def compute_cost(problem: Problem,
                 x_coords: List[float],
                 y_coords: List[float],
                 overlap_lambda: float = 0.0) -> Tuple[float, float, float, float]:
    """
    计算总代价

    Args:
        problem: 问题实例
        x_coords: 各矩形 x 坐标
        y_coords: 各矩形 y 坐标
        overlap_lambda: 重叠惩罚系数

    Returns:
        (total_cost, hpwl, area, overlap_penalty)
    """
    hpwl = compute_hpwl(problem, x_coords, y_coords)
    area = compute_area(problem, x_coords, y_coords)
    overlap = compute_overlap_penalty(problem, x_coords, y_coords)

    total_cost = 10 * hpwl + area + overlap_lambda * overlap

    return total_cost, hpwl, area, overlap


def count_overlaps(problem: Problem, x_coords: List[float], y_coords: List[float]) -> int:
    """统计重叠的矩形对数量"""
    n = problem.n
    count = 0

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

            if overlap_x > 0.001 and overlap_y > 0.001:
                count += 1

    return count
