"""
主入口：读取输入 → 调用算法 → 输出结果

支持单用例和多用例批量执行。

用法:
  python main.py input.json [--time 115]          # 单用例
  python main.py case1.json case2.json [--time 115]  # 多用例
  python main.py -d ./cases [--time 115]          # 目录下所有 JSON
  python main.py -t [--time 30]                   # 内置测试用例
"""
import json
import sys
import os
import glob
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from route import Problem, solve


def main():
    time_limit = 115.0
    input_files = []

    # 解析命令行参数：过滤掉 --xxx 及其值
    args = []
    i = 1
    while i < len(sys.argv):
        if sys.argv[i].startswith("--"):
            i += 2  # 跳过 --key 和 value
        else:
            args.append(sys.argv[i])
            i += 1
    
    if "--time" in sys.argv:
        idx = sys.argv.index("--time")
        if idx + 1 < len(sys.argv):
            time_limit = float(sys.argv[idx + 1])

    # 处理输入源
    if "-t" in args:
        # 内置测试用例
        run_single_case(_builtin_test_case(), time_limit, "builtin_test")
        return
    
    if "-d" in args:
        # 目录下所有 JSON
        idx = args.index("-d")
        if idx + 1 < len(args):
            dir_path = args[idx + 1]
            input_files = sorted(glob.glob(os.path.join(dir_path, "*.json")))
            if not input_files:
                print(f"错误: 目录 {dir_path} 中没有找到 JSON 文件")
                sys.exit(1)
        else:
            print("错误: -d 需要指定目录路径")
            sys.exit(1)
    else:
        # 单个或多个文件
        input_files = args

    if not input_files:
        print("Usage:")
        print("  python main.py input.json [--time 115]")
        print("  python main.py case1.json case2.json [--time 115]")
        print("  python main.py -d ./cases [--time 115]")
        print("  python main.py -t [--time 30]")
        sys.exit(1)

    # 批量执行
    if len(input_files) == 1:
        # 单用例模式
        with open(input_files[0], "r") as f:
            data = json.load(f)
        run_single_case(data, time_limit, os.path.basename(input_files[0]))
    else:
        # 多用例模式
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
                result = run_single_case(data, time_limit, os.path.basename(filepath))
                results.append((os.path.basename(filepath), result))
            except Exception as e:
                print(f"❌ 执行失败: {e}")
                results.append((os.path.basename(filepath), None))

        # 汇总报告
        print_summary(results)


def run_single_case(data: Dict[str, Any], time_limit: float, case_name: str) -> Dict[str, Any]:
    """执行单个用例，返回结果"""
    print(f"输入: {len(data['box_size'])} 个矩形, "
          f"{len(data.get('nets', []))} 个网络")
    print(f"时间限制: {time_limit}s\n")

    problem = Problem(data)
    result = solve(problem, time_limit=time_limit)

    print(f"\n=== 结果 ===")
    print(f"Cost:    {result['cost']}")
    print(f"  HPWL:  {result['hpwl']}")
    print(f"  Area:  {result['area']}")
    print(f"  Overlap: {result['overlap']}")
    print(f"耗时:    {result['elapsed_seconds']}s")

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


def print_summary(results: List[tuple]):
    """打印批量执行汇总"""
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


if __name__ == "__main__":
    main()
