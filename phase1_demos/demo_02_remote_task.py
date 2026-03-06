# -*- coding: utf-8 -*-
"""
===========================================================
Demo 02: @ray.remote 与 Task 提交执行
===========================================================

本脚本演示 Ray Remote Task 的核心用法，帮助理解以下内容：
  1. @ray.remote 装饰器将普通函数转为 RemoteFunction
  2. .remote() 调用提交 Task 并返回 ObjectRef（Future）
  3. 资源参数配置（num_cpus 等）
  4. 并行 vs 串行耗时对比
  5. .options() 方法覆盖默认参数
  6. 嵌套 Task（在 remote 函数中调用另一个 remote 函数）

源码调用链（走读参考）：
  @ray.remote 装饰器：
    python/ray/remote_function.py::RemoteFunction.__init__()
    → 将普通 Python 函数包装为 RemoteFunction 对象

  .remote() 调用：
    python/ray/remote_function.py::RemoteFunction._remote()
    → python/ray/_raylet.pyx::CoreWorker.submit_task()
      → src/ray/core_worker/core_worker.cc::CoreWorker::SubmitTask()
        → src/ray/core_worker/transport/normal_task_submitter.cc::NormalTaskSubmitter::SubmitTask()
"""

import ray
import time


# ===========================================================
# 定义 Remote 函数
# ===========================================================

# @ray.remote 将普通 Python 函数转化为 RemoteFunction 对象。
# 此时函数并没有被执行，只是被注册为一个可远程调用的 Task 模板。
# 源码：python/ray/remote_function.py::RemoteFunction
#   → 内部保存了函数体、资源需求、重试策略等元数据
@ray.remote
def slow_square(x):
    """模拟一个耗时的计算任务：休眠 1 秒后返回 x 的平方。"""
    time.sleep(1)
    return x * x


@ray.remote
def add(a, b):
    """简单的加法函数，用于演示基本调用。"""
    return a + b


# 指定资源需求的 remote 函数
# @ray.remote(num_cpus=2) 表示该 Task 需要 2 个 CPU 资源单位才能被调度执行。
# 源码：ray.remote() 接收 num_cpus 参数 → 存储在 RemoteFunction._default_options 中
#   → SubmitTask() 时携带 ResourceMap{"CPU": 2} → Raylet Scheduler 根据此需求分配 Worker
@ray.remote(num_cpus=2)
def heavy_computation(x):
    """模拟一个需要较多 CPU 的计算任务。"""
    time.sleep(0.5)
    return x ** 3


# 用于嵌套 Task 演示的辅助函数
@ray.remote
def double(x):
    """将输入值翻倍。"""
    return x * 2


@ray.remote
def nested_task(x):
    """
    嵌套 Task 示例：在一个 remote 函数内部调用另一个 remote 函数。

    嵌套调用的源码流程：
    1. nested_task 在某个 Worker 进程上执行
    2. 内部调用 double.remote(x) 再次提交一个新的 Task
       → CoreWorker::SubmitTask() （该 Worker 自身就是一个 CoreWorker）
    3. 新 Task 被 Raylet 调度到（可能是另一个）Worker 上执行
    4. 通过 ray.get() 在当前 Worker 上阻塞等待结果
    """
    ref = double.remote(x)
    doubled = ray.get(ref)
    return doubled + 1  # 2*x + 1


def print_section(title):
    """打印分隔线和标题。"""
    print(f"\n{'─' * 60}")
    print(f"  📌 {title}")
    print(f"{'─' * 60}")


def main():
    print("=" * 60)
    print("  Demo 02: @ray.remote — Task 提交与执行")
    print("=" * 60)

    # 初始化 Ray（如果尚未初始化）
    if not ray.is_initialized():
        ray.init(num_cpus=4)

    # ===========================================================
    # 示例 1：基本的 @ray.remote 函数调用
    # ===========================================================
    print_section("示例 1：基本的 @ray.remote 函数调用")

    # .remote() 调用不会立即执行函数，而是：
    # 1. 将函数和参数序列化
    # 2. 通过 CoreWorker 向 Raylet 提交一个 Task
    # 3. 立即返回一个 ObjectRef（Future），不阻塞当前进程 这是一个异步操作
    #
    # 源码：RemoteFunction._remote()
    #   → CoreWorker.submit_task(function_descriptor, args, resources)
    #     → Raylet 将 Task 放入调度队列
    #       → 选择合适的 Worker 执行
    ref = add.remote(3, 5)
    print(f"  add.remote(3, 5) 返回 ObjectRef: {ref}")
    print(f"  （注意：此时 Task 可能还在执行中，ObjectRef 只是一个引用/Future）")

    # ray.get() 会阻塞等待 Task 完成并返回结果
    # 源码：python/ray/worker.py::get()
    #   → CoreWorker::Get() → 如果对象不在本地，向 Object Store 拉取
    result = ray.get(ref)
    print(f"  ray.get(ref) = {result}")  # 期望输出 8

    # ===========================================================
    # 示例 2：资源参数配置
    # ===========================================================
    print_section("示例 2：@ray.remote(num_cpus=2) 资源参数配置")

    print(f"  heavy_computation 需要 2 个 CPU 资源单位")
    print(f"  当前集群总 CPU: {ray.cluster_resources().get('CPU', 'N/A')}")

    ref = heavy_computation.remote(5)
    result = ray.get(ref)
    print(f"  heavy_computation.remote(5) = {result}")  # 5^3 = 125

    # ===========================================================
    # 示例 3：并行 vs 串行耗时对比
    # ===========================================================
    print_section("示例 3：并行 vs 串行耗时对比")

    num_tasks = 4

    # 串行执行
    print(f"\n  串行执行 {num_tasks} 个 slow_square 任务（每个耗时约 1 秒）...")
    start = time.time()
    serial_results = []
    for i in range(num_tasks):
        # 注意：这里故意串行调用 ray.get()，每次都阻塞等待
        serial_results.append(ray.get(slow_square.remote(i)))
    serial_time = time.time() - start
    print(f"  串行结果: {serial_results}")
    print(f"  串行耗时: {serial_time:.2f} 秒")

    # 并行执行
    # 关键区别：先提交所有 Task（返回 ObjectRef 列表），再统一 ray.get()
    # 这样所有 Task 会被 Raylet 并行调度到不同 Worker 上同时执行
    print(f"\n  并行执行 {num_tasks} 个 slow_square 任务...")
    start = time.time()
    refs = [slow_square.remote(i) for i in range(num_tasks)]  # 全部提交，不阻塞
    parallel_results = ray.get(refs)  # 统一等待所有结果
    parallel_time = time.time() - start
    print(f"  并行结果: {parallel_results}")
    print(f"  并行耗时: {parallel_time:.2f} 秒")

    speedup = serial_time / parallel_time if parallel_time > 0 else float('inf')
    print(f"\n  ⚡ 加速比: {speedup:.2f}x （理想值约为 {num_tasks}x）")

    # ===========================================================
    # 示例 4：.options() 方法覆盖默认参数
    # ===========================================================
    print_section("示例 4：.options() 覆盖默认参数")

    # .options() 返回一个新的 RemoteFunction 副本，使用新的参数配置。
    # 这不会修改原始的 @ray.remote 定义。
    # 源码：python/ray/remote_function.py::RemoteFunction.options()
    #   → 创建新的 RemoteFunction，覆盖 _default_options 中的指定字段
    #
    # 应用场景：
    #   - 动态调整 Task 的资源需求
    #   - 为特定调用设置重试次数
    #   - 指定 Task 名称便于 Dashboard 中追踪
    slow_square_named = slow_square.options(
        name="my_square_task",     # 在 Ray Dashboard 中可见的 Task 名称
        num_cpus=1,                # 覆盖默认 CPU 需求
        max_retries=3,             # 失败自动重试 3 次
    )

    print(f"  通过 .options() 创建了带自定义参数的 Task：")
    print(f"    name='my_square_task', num_cpus=1, max_retries=3")
    ref = slow_square_named.remote(10)
    result = ray.get(ref)
    print(f"  slow_square_named.remote(10) = {result}")  # 100

    # ===========================================================
    # 示例 5：嵌套 Task
    # ===========================================================
    print_section("示例 5：嵌套 Task（remote 函数中调用 remote 函数）")

    print(f"  nested_task(x) 内部调用 double(x)，返回 2*x + 1")
    refs = [nested_task.remote(i) for i in range(5)]
    results = ray.get(refs)
    print(f"  输入: {list(range(5))}")
    print(f"  结果: {results}")  # [1, 3, 5, 7, 9]
    print(f"  验证: 2*0+1=1, 2*1+1=3, 2*2+1=5, 2*3+1=7, 2*4+1=9 ✅")

    # ===========================================================
    # 清理
    # ===========================================================
    ray.shutdown()

    print("\n" + "=" * 60)
    print("  Demo 02 执行完毕！")
    print("=" * 60)


if __name__ == "__main__":
    main()
