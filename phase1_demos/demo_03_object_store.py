# -*- coding: utf-8 -*-
"""
===========================================================
Demo 03: ray.get() / ray.put() — 对象存储与获取
===========================================================

本脚本演示 Ray 分布式对象存储的核心操作，帮助理解以下内容：
  1. ray.put() 存储 Python 对象到 Object Store，返回 ObjectRef
  2. ObjectRef 的含义（它是引用/指针，不是值本身）
  3. ray.get() 单个和批量获取对象
  4. 大对象共享传递 vs 直接传参的性能对比
  5. ray.wait() 非阻塞等待和批量处理模式

源码调用链（走读参考）：

  ray.put():
    python/ray/worker.py::put()
    → python/ray/_raylet.pyx::CoreWorker.put()
      → src/ray/core_worker/core_worker.cc::CoreWorker::Put()
        → src/ray/object_manager/memory_store.cc::MemoryStore::Put()
        → （大对象）src/ray/object_manager/plasma/store.cc::PlasmaStore

  ray.get():
    python/ray/worker.py::get()
    → python/ray/_raylet.pyx::CoreWorker.get()
      → src/ray/core_worker/core_worker.cc::CoreWorker::Get()
        → 本地查找 MemoryStore / PlasmaStore
        → （远程）src/ray/object_manager/object_manager.cc::ObjectManager::Pull()

  ray.wait():
    python/ray/worker.py::wait()
    → CoreWorker::Wait()
      → 检查对象就绪状态，返回 ready/not_ready 列表

可选依赖说明：
  本 Demo 中的"大对象传递"示例使用了 numpy 库来创建大型数组。
  如果环境中未安装 numpy，相关示例会优雅跳过并给出提示。
  安装方式：pip install numpy
"""

import ray
import time
import sys

# 尝试导入 numpy（可选依赖）
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@ray.remote
def process_data(data_ref, index):
    """
    处理共享数据的 Task。
    参数 data_ref 是一个已经在 Object Store 中的数据的 ObjectRef。

    源码流程：
    1. Worker 接收到 Task 时，参数中的 ObjectRef 会被自动解引用
       → CoreWorker::ExecuteTask() 中调用 GetArgs()
       → 如果对象在本地 Object Store 中，零拷贝读取
       → 如果对象不在本地，从远端 Pull 过来
    2. 对于 numpy 数组等大对象，使用共享内存（Plasma），实现零拷贝
    """
    # data_ref 在 Task 内部已经被自动解引用为实际的数据
    if HAS_NUMPY:
        return float(data_ref.sum()) + index
    else:
        return sum(data_ref) + index


@ray.remote
def process_direct(data, index):
    """
    直接传参的 Task（对比用）。
    data 会在每次调用时被序列化和传输，不走 Object Store 共享。
    """
    if HAS_NUMPY:
        return float(data.sum()) + index
    else:
        return sum(data) + index


@ray.remote
def slow_task(i, sleep_time):
    """模拟不同耗时的任务，用于演示 ray.wait()。"""
    time.sleep(sleep_time)
    return f"任务 {i} 完成 (耗时 {sleep_time:.1f}s)"


def print_section(title):
    """打印分隔线和标题。"""
    print(f"\n{'─' * 60}")
    print(f"  📌 {title}")
    print(f"{'─' * 60}")


def main():
    print("=" * 60)
    print("  Demo 03: ray.get() / ray.put() — 对象存储与获取")
    print("=" * 60)

    # 初始化 Ray
    if not ray.is_initialized():
        ray.init(num_cpus=4)

    # ===========================================================
    # 示例 1：ray.put() 存储对象并返回 ObjectRef
    # ===========================================================
    print_section("示例 1：ray.put() 存储对象并返回 ObjectRef")

    # ray.put() 将 Python 对象序列化后存入当前节点的 Object Store
    # 返回 ObjectRef，它是一个全局唯一的引用 ID
    #
    # 源码流程：
    # 1. python/ray/worker.py::put(value)
    # 2. → 序列化 value（默认使用 pickle + Apache Arrow）
    # 3. → CoreWorker::Put(serialized_data)
    # 4. → 小对象（< 100KB）: 存入 MemoryStore（进程内内存）
    #    → 大对象（>= 100KB）: 存入 PlasmaStore（共享内存）
    # 5. → 返回 ObjectRef（包含 ObjectID + 所属节点信息）
    data_list = [1, 2, 3, 4, 5]
    ref = ray.put(data_list) # 返回的引用对象可以作为一个索引，这个索引是记录了数据在ray里面存储的位置，
    # 在通过这个索引来获取数据到本地


    print(f"  存入的数据: {data_list}")
    print(f"  返回的 ObjectRef: {ref}")
    print(f"  ObjectRef 类型: {type(ref)}")
    print()
    print(f"  💡 ObjectRef 是什么？")
    print(f"     - 它是一个引用（指针），不是数据本身")
    print(f"     - 它包含 ObjectID（全局唯一标识）和位置信息")
    print(f"     - 通过 ObjectRef 可以在任何节点上获取到对应的数据")
    print(f"     - 多个 Task 可以共享同一个 ObjectRef，避免数据重复传输")

    # 存储不同类型的对象
    ref_str = ray.put("Hello, Ray Object Store!")
    ref_dict = ray.put({"name": "Ray", "version": "2.x", "type": "分布式计算框架"})

    print(f"\n  存储字符串 → ObjectRef: {ref_str}")
    print(f"  存储字典   → ObjectRef: {ref_dict}")

    # ===========================================================
    # 示例 2：ray.get() 单个和批量获取
    # ===========================================================
    print_section("示例 2：ray.get() 单个和批量获取")

    # 单个获取
    # ray.get(ref) 会阻塞当前进程，直到对象可用
    # 源码：CoreWorker::Get()
    #   → 先在本地 MemoryStore/PlasmaStore 查找
    #   → 如果不在本地，向 ObjectManager 请求从远端 Pull
    #   → 反序列化后返回 Python 对象
    val_list = ray.get(ref)
    val_str = ray.get(ref_str)
    val_dict = ray.get(ref_dict)

    print(f"  获取列表: {val_list}")
    print(f"  获取字符串: {val_str}")
    print(f"  获取字典: {val_dict}")

    # 批量获取
    # ray.get([ref1, ref2, ...]) 可以同时等待多个对象
    # 内部会并行地从 Object Store 获取，比逐个 get 更高效
    print(f"\n  批量获取（一次 ray.get 传入列表）：")
    results = ray.get([ref, ref_str, ref_dict])
    for i, r in enumerate(results):
        print(f"    [{i}] {r}")

    # ===========================================================
    # 示例 3：大对象共享传递 vs 直接传参
    # ===========================================================
    print_section("示例 3：大对象共享传递 vs 直接传参（性能对比）")

    if not HAS_NUMPY:
        print(f"  ⚠️  未安装 numpy，跳过大对象性能对比示例。")
        print(f"      安装方式：pip install numpy")
    else:
        # 创建一个较大的 numpy 数组（约 8MB）
        big_array = np.random.rand(1_000_000)
        array_size_mb = big_array.nbytes / 1024 / 1024
        print(f"  创建了 numpy 数组: shape={big_array.shape}, 大小≈{array_size_mb:.1f}MB")

        num_tasks = 4

        # 方式 A：先 ray.put() 存入 Object Store，再将 ObjectRef 传给多个 Task
        # 优势：数据只序列化和传输一次，多个 Worker 通过共享内存零拷贝访问
        print(f"\n  方式 A：ray.put() + ObjectRef 共享传递（{num_tasks} 个 Task）...")
        start = time.time()
        big_ref = ray.put(big_array)  # 只存一次
        refs_a = [process_data.remote(big_ref, i) for i in range(num_tasks)]
        results_a = ray.get(refs_a)
        time_a = time.time() - start
        print(f"    结果: {results_a}")
        print(f"    耗时: {time_a:.3f} 秒")

        # 方式 B：直接将大数据作为参数传入每个 Task
        # 劣势：每次调用都会序列化一份数据，内存占用和传输开销成倍增加
        print(f"\n  方式 B：直接传参（{num_tasks} 个 Task）...")
        start = time.time()
        # process_direct.remote 其实是创建一个actor来执行任务
        refs_b = [process_direct.remote(big_array, i) for i in range(num_tasks)]
        results_b = ray.get(refs_b)
        time_b = time.time() - start
        print(f"    结果: {results_b}")
        print(f"    耗时: {time_b:.3f} 秒")

        if time_b > 0:
            print(f"\n  📊 方式 A vs 方式 B 耗时比: {time_a:.3f}s vs {time_b:.3f}s")
            if time_a < time_b:
                print(f"  ✅ ray.put() 共享传递更快！节省了 {time_b - time_a:.3f} 秒")
            else:
                print(f"  ℹ️  数据量较小时差异不明显，数据量越大优势越明显")

    # ===========================================================
    # 示例 4：ray.wait() 非阻塞等待和批量处理
    # ===========================================================
    print_section("示例 4：ray.wait() 非阻塞等待和批量处理")

    # ray.wait() 与 ray.get() 的区别：
    #   - ray.get() 阻塞直到所有对象就绪
    #   - ray.wait() 返回已就绪和未就绪的 ObjectRef 列表，不阻塞等待全部完成
    #
    # 源码：python/ray/worker.py::wait(refs, num_returns, timeout)
    #   → CoreWorker::Wait()
    #     → 轮询检查每个 ObjectRef 对应的对象是否已在 Object Store 中就绪
    #     → 返回 (ready_refs, remaining_refs)
    #
    # 应用场景：
    #   - 流式处理：哪个 Task 先完成就先处理，不用等全部完成
    #   - 超时控制：设置 timeout 避免无限等待
    #   - 批量消费：每次取 N 个结果处理

    # 提交多个不同耗时的任务
    sleep_times = [2.0, 0.5, 1.5, 0.2, 1.0]
    print(f"  提交 {len(sleep_times)} 个任务，耗时分别为: {sleep_times} 秒")
    refs = [slow_task.remote(i, t) for i, t in enumerate(sleep_times)]

    # 使用 ray.wait() 逐个获取已完成的结果（流式处理模式）
    print(f"\n  使用 ray.wait() 流式获取结果（先完成先处理）：")
    remaining = list(refs)
    completed_order = []
    while remaining:
        # num_returns=1 表示每次只要有 1 个就绪就返回
        # timeout=None 表示一直等到至少有 1 个就绪
        ready, remaining = ray.wait(remaining, num_returns=1, timeout=None)
        result = ray.get(ready[0])
        completed_order.append(result)
        print(f"    ✅ {result}  (剩余 {len(remaining)} 个)")

    print(f"\n  完成顺序（按实际完成时间排列）：")
    for i, r in enumerate(completed_order):
        print(f"    {i + 1}. {r}")

    # 演示 ray.wait() 的 timeout 参数
    print(f"\n  演示 ray.wait() 超时控制：")
    refs2 = [slow_task.remote(i, 3.0) for i in range(3)]
    ready, not_ready = ray.wait(refs2, num_returns=3, timeout=0.5)
    print(f"    提交了 3 个耗时 3 秒的任务，设置 timeout=0.5 秒")
    print(f"    0.5 秒后：{len(ready)} 个已就绪，{len(not_ready)} 个未就绪")
    # 清理：等待所有任务完成避免打印混乱
    ray.get(not_ready)

    # ===========================================================
    # 清理
    # ===========================================================
    ray.shutdown()

    print("\n" + "=" * 60)
    print("  Demo 03 执行完毕！")
    print("=" * 60)


if __name__ == "__main__":
    main()
