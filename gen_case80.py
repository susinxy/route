#!/usr/bin/env python3
"""生成 case80: 80 boxes, 全覆盖约束, 多约束交叉, 精心预期布局"""
import json, sys, math
sys.path.insert(0, '/home/susinxy/.openclaw/workspace-coding/route')
from route import Problem, ConstraintSystem, solve

# ============================================================
# 1. 定义尺寸 (80 boxes)
# ============================================================
sizes = [
    # --- X 对称组 1: boxes 1-8 ---
    [10, 5],   # 1  (pair 左)  [交叉: align_right]
    [10, 5],   # 2  (pair 右)
    [8, 4],    # 3  (pair 左)
    [8, 4],    # 4  (pair 右)
    [6, 6],    # 5  (pair 左)
    [6, 6],    # 6  (pair 右)
    [12, 8],   # 7  (self_sym) [交叉: align_bottom]
    [12, 8],   # 8  (self_sym)

    # --- X 对称组 2: boxes 9-14 ---
    [10, 5],   # 9  (pair 左)  [交叉: align_right]
    [10, 5],   # 10 (pair 右)
    [5, 10],   # 11 (pair 左)
    [5, 10],   # 12 (pair 右)
    [8, 4],    # 13 (pair 左)
    [8, 4],    # 14 (pair 右)

    # --- Y 对称组 1: boxes 15-22 ---
    [10, 5],   # 15 (pair 下)  [交叉: nets 多约束]
    [10, 5],   # 16 (pair 上)
    [12, 8],   # 17 (pair 下)
    [12, 8],   # 18 (pair 上)
    [6, 6],    # 19 (pair 下)
    [6, 6],    # 20 (pair 上)
    [4, 12],   # 21 (pair 下)
    [4, 12],   # 22 (pair 上)

    # --- Y 对称组 2: boxes 23-28 ---
    [8, 4],    # 23 (pair 下)
    [8, 4],    # 24 (pair 上)
    [6, 6],    # 25 (pair 下)
    [6, 6],    # 26 (pair 上)
    [10, 5],   # 27 (self_sym) [交叉: align_bottom]
    [10, 5],   # 28 (self_sym)

    # --- 对齐组: boxes 29-42 ---
    [15, 6],   # 29 align_left
    [8, 4],    # 30 align_left
    [12, 3],   # 31 align_left
    [6, 8],    # 32 align_left
    [10, 5],   # 33 align_right
    [8, 4],    # 34 align_right
    [6, 6],    # 35 align_right
    [5, 10],   # 36 align_top
    [5, 10],   # 37 align_top
    [5, 10],   # 38 align_top
    [5, 10],   # 39 align_top
    [12, 8],   # 40 align_bottom
    [8, 4],    # 41 align_bottom
    [6, 6],    # 42 align_bottom

    # --- 重复组 1: boxes 43-48 (3×2) ---
    [10, 5],   # 43 template [交叉: align_left]
    [8, 4],    # 44 template
    [6, 6],    # 45 template
    [10, 5],   # 46 copy    [交叉: align_left]
    [8, 4],    # 47 copy
    [6, 6],    # 48 copy

    # --- 重复组 2: boxes 49-56 (4×2) ---
    [12, 8],   # 49 template
    [10, 5],   # 50 template
    [8, 4],    # 51 template
    [6, 6],    # 52 template
    [12, 8],   # 53 copy
    [10, 5],   # 54 copy
    [8, 4],    # 55 copy
    [6, 6],    # 56 copy

    # --- 重复组 3: boxes 57-65 (3×3) ---
    [5, 10],   # 57 template [交叉: align_top]
    [4, 12],   # 58 template
    [6, 6],    # 59 template
    [5, 10],   # 60 copy1   [交叉: align_top]
    [4, 12],   # 61 copy1
    [6, 6],    # 62 copy1
    [5, 10],   # 63 copy2   [交叉: align_top]
    [4, 12],   # 64 copy2
    [6, 6],    # 65 copy2

    # --- 额外 boxes: 66-80 ---
    [15, 6],   # 66
    [8, 4],    # 67
    [12, 3],   # 68
    [6, 8],    # 69
    [10, 5],   # 70
    [5, 10],   # 71
    [5, 10],   # 72
    [5, 10],   # 73
    [12, 8],   # 74
    [8, 4],    # 75
    [6, 6],    # 76
    [6, 6],    # 77
    [10, 5],   # 78
    [8, 4],    # 79
    [6, 6],    # 80
]

assert len(sizes) == 80, f"Expected 80 boxes, got {len(sizes)}"

# ============================================================
# 2. 定义约束
# ============================================================

input_data = {
    "box_size": sizes,
    "symmetry_x": [
        {
            "symmetry_pair": [[1, 2], [3, 4], [5, 6]],
            "self_symmetry": [7, 8]
        },
        {
            "symmetry_pair": [[9, 10], [11, 12], [13, 14]],
            "self_symmetry": []
        }
    ],
    "symmetry_y": [
        {
            "symmetry_pair": [[15, 16], [17, 18], [19, 20], [21, 22]],
            "self_symmetry": []
        },
        {
            "symmetry_pair": [[23, 24], [25, 26]],
            "self_symmetry": [27, 28]
        }
    ],
    "align": {
        "left": [
            [29, 30, 31, 32],
            [43, 46]           # ← 交叉: repeat_group_1 的 template[0] 和 copy[0]
        ],
        "right": [
            [33, 34, 35],
            [1, 9]             # ← 交叉: sym_x_1 + sym_x_2 的 pair 成员
        ],
        "top": [
            [36, 37, 38, 39],
            [57, 60, 63]       # ← 交叉: repeat_group_3 的三个 copy 的 box[0]
        ],
        "bottom": [
            [40, 41, 42],
            [7, 27]            # ← 交叉: sym_x self + sym_y self
        ]
    },
    "repeat_groups": [
        {"groups": [[43, 44, 45], [46, 47, 48]]},                    # 3×2
        {"groups": [[49, 50, 51, 52], [53, 54, 55, 56]]},            # 4×2
        {"groups": [[57, 58, 59], [60, 61, 62], [63, 64, 65]]}      # 3×3
    ],
    "nets": [
        [1, 3, 5, 9, 11, 13],           # X对称 pair 左成员
        [2, 4, 6, 10, 12, 14],          # X对称 pair 右成员
        [15, 17, 19, 21, 23, 25],       # Y对称 pair 下成员
        [16, 18, 20, 22, 24, 26],       # Y对称 pair 上成员
        [29, 33, 36, 40],               # 对齐组代表连接
        [43, 49],                        # RG1 ↔ RG2
        [57, 66, 67, 68],               # RG3 + 额外 box
        [7, 15, 27, 42],                # 多约束交叉连接
        [69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80]  # 额外 box 大连接
    ],
    "start_from_1": True
}

# ============================================================
# 3. 验证约束化简
# ============================================================
problem = Problem(input_data)
cs = ConstraintSystem(problem)
print(f"✅ 80 boxes → {len(cs.free_vars)} 独立变量")
print(f"   X 对称: {len(input_data['symmetry_x'])} 组")
print(f"   Y 对称: {len(input_data['symmetry_y'])} 组")
print(f"   对齐: L{len(input_data['align']['left'])} R{len(input_data['align']['right'])} T{len(input_data['align']['top'])} B{len(input_data['align']['bottom'])}")
print(f"   重复组: {len(input_data['repeat_groups'])} 组")
print(f"   Nets: {len(input_data['nets'])} 个")

# 交叉约束统计
box_constraints = {}
for i in range(1, 81):
    box_constraints[i] = []

for gi, g in enumerate(input_data['symmetry_x']):
    for p in g['symmetry_pair']:
        for b in p: box_constraints[b].append(f'sym_x_{gi}')
    for b in g.get('self_symmetry', []):
        box_constraints[b].append(f'sym_x_{gi}_self')
for gi, g in enumerate(input_data['symmetry_y']):
    for p in g['symmetry_pair']:
        for b in p: box_constraints[b].append(f'sym_y_{gi}')
    for b in g.get('self_symmetry', []):
        box_constraints[b].append(f'sym_y_{gi}_self')
for side, groups in input_data['align'].items():
    for gi, g in enumerate(groups):
        for b in g: box_constraints[b].append(f'align_{side}_{gi}')
for ri, rg in enumerate(input_data['repeat_groups']):
    for gi, g in enumerate(rg['groups']):
        for b in g: box_constraints[b].append(f'repeat_{ri}_g{gi}')

multi_constraint_boxes = [(b, c) for b, c in box_constraints.items() if len(c) >= 2]
all_constraint_boxes = [(b, c) for b, c in box_constraints.items() if len(c) >= 3]

print(f"\n多约束 box (≥2): {len(multi_constraint_boxes)} 个")
for b, c in sorted(multi_constraint_boxes, key=lambda x: -len(x[1]))[:10]:
    print(f"  box{b}: {len(c)} 约束 → {c}")

print(f"\n全约束 box (≥3): {len(all_constraint_boxes)} 个")
for b, c in sorted(all_constraint_boxes, key=lambda x: -len(x[1])):
    print(f"  box{b}: {len(c)} 约束 → {c}")

# ============================================================
# 4. 用算法求解高质量解作为预期
# ============================================================
print(f"\n🚀 求解高质量解 (240s)...")
x, y, elapsed = solve(problem, time_limit=240.0)

from route import compute_hpwl, compute_area, compute_overlap_penalty
hpwl = compute_hpwl(problem, x, y)
area = compute_area(problem, x, y)
overlap = compute_overlap_penalty(problem, x, y)
cost = 10 * hpwl + area
print(f"\n解: hpwl={hpwl:.1f}, area={area:.1f}, cost={cost:.1f}, overlap={overlap:.4f}")

expected_positions = [[round(x[i], 4), round(y[i], 4)] for i in range(80)]

# ============================================================
# 5. 输出 JSON
# ============================================================
output = {
    "description": "case80: 80 boxes, 全覆盖约束(X对称×2+Y对称×2+对齐×8+重复组×3), 多约束交叉, 大用例",
    "input": input_data,
    "expected_output": {
        "box_position": expected_positions
    }
}

outpath = 'test_cases/case80_full_coverage.json'
with open(outpath, 'w') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n✅ 用例已保存到 {outpath}")
print(f"   cost={cost:.1f}, overlap={overlap:.4f}")
