# -*- coding: utf-8 -*-
"""
===========================================================
Run All: 阶段一走读测试 — 统一运行入口
===========================================================

本脚本按顺序执行阶段一的 4 个 Demo 模块：
  1. demo_01_ray_init    — ray.init() 初始化流程
  2. demo_02_remote_task — @ray.remote 与 Task 提交执行
  3. demo_03_object_store— ray.get() / ray.put() 对象存储
  4. demo_04_actor       — Actor 基本使用

使用方式：
  cd ray/phase1_demos
  python run_all.py

或从项目根目录：
  python -m phase1_demos.run_all
"""

import sys
import time
import traceback

import ray


# 定义要执行的 Demo 模块及其描述
DEMOS = [
    {
        "name": "Demo 01: ray.init()",
        "module": "demo_01_ray_init",
        "description": "演示 Ray 集群初始化流程、参数配置、集群信息获取",
    },
    {
        "name": "Demo 02: @ray.remote Task",
        "module": "demo_02_remote_task",
        "description": "演示 Remote Task 定义、提交、并行执行、.options()、嵌套 Task",
    },
    {
        "name": "Demo 03: Object Store",
        "module": "demo_03_object_store",
        "description": "演示 ray.put()/ray.get() 对象存储、大对象共享、ray.wait()",
    },
    {
        "name": "Demo 04: Actor",
        "module": "demo_04_actor",
        "description": "演示 Actor 创建、方法调用、状态管理、多 Actor 并行、Handle 传递",
    },
]


def run_demo(demo_info):
    """
    动态导入并执行单个 Demo 模块的 main() 函数。

    返回：
        (bool, str) — (是否成功, 错误信息或空字符串)
    """
    module_name = demo_info["module"]
    try:
        # 动态导入模块（支持从 phase1_demos 包内运行和直接运行两种方式）
        try:
            mod = __import__(f"phase1_demos.{module_name}", fromlist=[module_name])
        except ImportError:
            mod = __import__(module_name)

        # 调用 main() 函数
        mod.main()
        return True, ""
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        return False, error_msg


def main():
    total_start = time.time()

    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + "  🚀 Ray 阶段一走读测试 — 统一运行入口".center(46) + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    print(f"  即将按顺序执行 {len(DEMOS)} 个 Demo 模块：")
    for i, demo in enumerate(DEMOS):
        print(f"    {i + 1}. {demo['name']} — {demo['description']}")
    print()

    # 记录每个模块的执行结果
    results = []

    for i, demo in enumerate(DEMOS):
        # 确保每个 Demo 开始前 Ray 是关闭的（每个 Demo 自行管理 init/shutdown）
        if ray.is_initialized():
            ray.shutdown()

        print()
        print("┌" + "─" * 58 + "┐")
        print(f"│  📖 第 {i + 1}/{len(DEMOS)} 章: {demo['name']}")
        print(f"│  {demo['description']}")
        print("└" + "─" * 58 + "┘")
        print()

        demo_start = time.time()
        success, error_msg = run_demo(demo)
        demo_time = time.time() - demo_start

        results.append({
            "name": demo["name"],
            "success": success,
            "time": demo_time,
            "error": error_msg,
        })

        if success:
            print(f"\n  ✅ {demo['name']} 执行成功 (耗时 {demo_time:.1f}s)")
        else:
            print(f"\n  ❌ {demo['name']} 执行失败 (耗时 {demo_time:.1f}s)")
            print(f"  错误信息：")
            for line in error_msg.strip().split("\n"):
                print(f"    {line}")
            print(f"\n  ⚠️  继续执行下一个模块...")

    # 确保最终 Ray 被关闭
    if ray.is_initialized():
        ray.shutdown()

    total_time = time.time() - total_start

    # ===========================================================
    # 打印总结表格
    # ===========================================================
    print()
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + "  📊 执行总结".center(50) + "║")
    print("╠" + "═" * 58 + "╣")

    for r in results:
        status = "✅ 成功" if r["success"] else "❌ 失败"
        line = f"  {status}  {r['name']:<35s} ({r['time']:.1f}s)"
        print(f"║{line:<58s}║")

    print("╠" + "═" * 58 + "╣")

    success_count = sum(1 for r in results if r["success"])
    fail_count = len(results) - success_count
    summary = f"  总计: {success_count} 成功, {fail_count} 失败, 总耗时 {total_time:.1f}s"
    print(f"║{summary:<58s}║")
    print("╚" + "═" * 58 + "╝")
    print()

    if fail_count > 0:
        print("  ⚠️  部分模块执行失败，请检查上方错误信息。")
    else:
        print("  🎉 所有模块执行成功！阶段一走读测试全部通过。")
    print()


if __name__ == "__main__":
    main()
