# -*- coding: utf-8 -*-
"""
===========================================================
Demo 04: Actor 基本使用
===========================================================

本脚本演示 Ray Actor 的核心用法，帮助理解以下内容：
  1. @ray.remote 定义 Actor 类 → 通过 .remote() 创建 Actor 实例
  2. actor.method.remote() 调用方法并获取返回值
  3. Actor 内部状态累积（有状态计算）
  4. 多 Actor 并行工作
  5. Actor 资源配置（num_cpus）
  6. ActorHandle 传递给其他 Task 或 Actor

源码调用链（走读参考）：

  @ray.remote 定义 Actor 类：
    python/ray/actor.py::ActorClass._make()
    → 将普通 Python 类包装为 ActorClass 对象

  .remote() 创建 Actor 实例：
    python/ray/actor.py::ActorClass._remote()
    → python/ray/_raylet.pyx::CoreWorker.create_actor()
      → src/ray/core_worker/core_worker.cc::CoreWorker::CreateActor()
        → 向 GCS 注册 Actor 信息
          → src/ray/gcs/gcs_server/gcs_actor_manager.cc::GcsActorManager::RegisterActor()
        → GCS Scheduler 调度 Actor 到合适的节点
        → 目标节点的 Raylet 启动一个专用 Worker 进程来运行 Actor

  actor.method.remote() 调用方法：
    python/ray/actor.py::ActorMethod._remote()
    → CoreWorker::SubmitActorTask()
      → 将 Task 发送到 Actor 所在的 Worker（点对点直连，不经过 Raylet 调度）
      → Actor Worker 按 FIFO 顺序串行执行 Task（保证状态一致性）
"""

import ray
import time


# ===========================================================
# 定义 Actor 类
# ===========================================================

@ray.remote
class Counter:
    """
    简单计数器 Actor — 演示有状态的分布式对象。

    Actor 与普通 Remote Task 的核心区别：
    - Task 是无状态的：每次调用都在一个新的（或复用的）Worker 上执行，不保留状态
    - Actor 是有状态的：拥有专属 Worker 进程，所有方法调用在同一进程上串行执行，
      内部状态（self.xxx）在整个生命周期内持续存在

    源码：Actor Worker 启动后，CoreWorker 会维护一个 Actor 实例
      → 每个 method.remote() 调用对应一个 ActorTask
      → ActorTask 按 FIFO 顺序在该 Worker 上执行
      → self 指向同一个 Python 对象，因此状态得以保持
    """

    def __init__(self, name, initial_value=0):
        """
        Actor 的 __init__ 在 .remote() 调用时执行（在远端 Worker 上）。
        源码：ActorClass._remote() 时，__init__ 的参数会被序列化并发送到目标 Worker
        """
        self.name = name
        self.value = initial_value
        print(f"  [Actor '{self.name}'] 已创建，初始值 = {self.value}")

    def increment(self, delta=1):
        """累加计数器，返回新值。"""
        self.value += delta
        return self.value

    def get_value(self):
        """获取当前计数器的值。"""
        return self.value

    def get_name(self):
        """获取 Actor 名称。"""
        return self.name


# 指定资源需求的 Actor
# 该 Actor 需要 0.5 个 CPU 资源单位（可以是小数）
# 源码：ActorClass._remote() 时将 num_cpus 传入 → GCS Scheduler 根据资源需求选择节点
@ray.remote(num_cpus=0.5)
class LightweightWorker:
    """轻量级 Worker Actor，只需少量 CPU 资源。"""

    def __init__(self, worker_id):
        self.worker_id = worker_id
        self.tasks_done = 0

    def do_work(self, data):
        """处理一个数据项。"""
        time.sleep(0.1)  # 模拟轻量工作
        self.tasks_done += 1
        return f"Worker-{self.worker_id} 处理了数据 '{data}' (累计 {self.tasks_done} 个)"

    def get_stats(self):
        """返回工作统计。"""
        return {"worker_id": self.worker_id, "tasks_done": self.tasks_done}


# 用于演示 ActorHandle 传递的 Actor
@ray.remote
class Logger:
    """日志收集 Actor — 接收来自其他 Task/Actor 的日志。"""

    def __init__(self):
        self.logs = []

    def log(self, message):
        """记录一条日志。"""
        self.logs.append(message)
        return len(self.logs)

    def get_logs(self):
        """获取所有日志。"""
        return self.logs


@ray.remote
def task_with_actor_handle(actor_handle, task_id):
    """
    演示在普通 Task 中使用 ActorHandle。

    ActorHandle 是对远端 Actor 实例的引用，可以在任何地方（Driver、Task、其他 Actor）
    通过 handle.method.remote() 调用 Actor 的方法。

    源码流程：
    1. ActorHandle 被序列化后传入 Task（序列化的是 Actor 的 ID 和地址，不是 Actor 对象本身）
    2. Task 内部反序列化得到 ActorHandle
    3. 通过 handle.method.remote() 向 Actor 发送 ActorTask
       → CoreWorker::SubmitActorTask() → 直连到 Actor Worker
    """
    # 向 Logger Actor 写入日志
    ray.get(actor_handle.log.remote(f"Task-{task_id} 开始执行"))
    time.sleep(0.5)  # 模拟工作
    ray.get(actor_handle.log.remote(f"Task-{task_id} 执行完毕"))
    return f"Task-{task_id} done"


def print_section(title):
    """打印分隔线和标题。"""
    print(f"\n{'─' * 60}")
    print(f"  📌 {title}")
    print(f"{'─' * 60}")


def main():
    print("=" * 60)
    print("  Demo 04: Actor — 有状态的分布式计算")
    print("=" * 60)

    # 初始化 Ray
    if not ray.is_initialized():
        ray.init(num_cpus=4)

    # ===========================================================
    # 示例 1：创建 Actor 实例
    # ===========================================================
    print_section("示例 1：@ray.remote 定义 Actor 类并创建实例")

    # .remote() 调用的作用：
    # 1. 序列化 __init__ 参数
    # 2. 向 GCS 注册一个新的 Actor
    #    → GcsActorManager::RegisterActor() 分配 ActorID
    # 3. GCS Scheduler 选择一个节点，通知其 Raylet 启动一个 Worker 进程
    # 4. 新 Worker 进程中创建 Actor 对象（执行 __init__）
    # 5. 返回 ActorHandle（包含 ActorID、地址等信息）
    counter = Counter.remote("demo_counter", initial_value=10)

    print(f"  创建了 Counter Actor")
    print(f"  ActorHandle: {counter}")
    print(f"  ActorHandle 类型: {type(counter)}")

    # ===========================================================
    # 示例 2：调用 Actor 方法
    # ===========================================================
    print_section("示例 2：actor.method.remote() 调用方法")

    # actor.method.remote() 的调用流程：
    # 1. 序列化方法名和参数
    # 2. CoreWorker::SubmitActorTask() → 直连到 Actor 所在 Worker
    #    （注意：ActorTask 不经过 Raylet 调度，是 Driver/Worker 直连 Actor Worker）
    # 3. Actor Worker 将 Task 放入 FIFO 队列，按顺序执行
    # 4. 返回 ObjectRef（Future）
    name_ref = counter.get_name.remote()
    value_ref = counter.get_value.remote()

    # ray.get() 等待 Actor 方法执行完毕并返回结果
    name = ray.get(name_ref)
    value = ray.get(value_ref)
    print(f"  Actor 名称: {name}")
    print(f"  当前值: {value}")

    # 连续调用 increment
    for i in range(5):
        new_val = ray.get(counter.increment.remote(delta=i + 1))
        print(f"  increment({i + 1}) → 新值: {new_val}")

    final_value = ray.get(counter.get_value.remote())
    print(f"\n  最终值: {final_value}  (期望: 10 + 1+2+3+4+5 = 25)")

    # ===========================================================
    # 示例 3：Actor 内部状态累积（有状态证明）
    # ===========================================================
    print_section("示例 3：Actor 内部状态累积 — 证明 Actor 是有状态的")

    # 创建一个新的 Counter Actor，从 0 开始
    stateful_counter = Counter.remote("stateful_demo", initial_value=0)

    # 连续调用 10 次 increment(1)
    # 如果 Actor 是无状态的，每次调用都会从 0 开始 → 结果应该全是 1
    # 但因为 Actor 是有状态的，值会不断累积 → 结果是 1, 2, 3, ..., 10
    refs = []
    for i in range(10):
        refs.append(stateful_counter.increment.remote(1))
    results = ray.get(refs)

    print(f"  连续 10 次 increment(1) 的返回值: {results}")
    print(f"  期望: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]")
    assert results == list(range(1, 11)), "状态累积验证失败！"
    print(f"  ✅ 验证通过！Actor 内部状态在多次调用间保持并累积。")

    # ===========================================================
    # 示例 4：多 Actor 并行工作
    # ===========================================================
    print_section("示例 4：多 Actor 并行工作 — 分布式有状态计算")

    # 创建多个 Counter Actor 实例
    # 每个实例是独立的远程对象，拥有独立的状态和独立的 Worker 进程
    num_actors = 3
    counters = [Counter.remote(f"worker_{i}", initial_value=0) for i in range(num_actors)]
    print(f"  创建了 {num_actors} 个 Counter Actor")

    # 向每个 Actor 发送不同数量的 increment 任务
    all_refs = []
    for idx, c in enumerate(counters):
        count = (idx + 1) * 3  # Actor 0: 3次, Actor 1: 6次, Actor 2: 9次
        refs = [c.increment.remote(1) for _ in range(count)]
        all_refs.extend(refs)
        print(f"  Actor worker_{idx}: 提交了 {count} 次 increment")

    # 等待所有任务完成
    ray.get(all_refs)

    # 获取每个 Actor 的最终状态
    print(f"\n  各 Actor 最终状态：")
    for idx, c in enumerate(counters):
        final_val = ray.get(c.get_value.remote())
        name = ray.get(c.get_name.remote())
        expected = (idx + 1) * 3
        status = "✅" if final_val == expected else "❌"
        print(f"    {status} {name}: value = {final_val} (期望 {expected})")

    # ===========================================================
    # 示例 5：Actor 资源配置
    # ===========================================================
    print_section("示例 5：@ray.remote(num_cpus=0.5) 为 Actor 指定资源")

    # LightweightWorker 只需 0.5 个 CPU，因此在 4 CPU 集群上可以同时运行 8 个
    # 源码：ActorClass._remote() 传入 num_cpus=0.5
    #   → GcsActorManager 注册 Actor 时记录资源需求
    #   → GCS Scheduler 选择有足够资源的节点
    print(f"  LightweightWorker 只需 0.5 CPU，集群有 4 CPU")
    print(f"  理论上可同时运行 8 个 LightweightWorker")

    workers = [LightweightWorker.remote(i) for i in range(4)]
    print(f"  创建了 {len(workers)} 个 LightweightWorker")

    # 向各 Worker 分配任务
    results_refs = []
    data_items = ["apple", "banana", "cherry", "date", "elderberry", "fig", "grape", "honeydew"]
    for i, item in enumerate(data_items):
        worker = workers[i % len(workers)]  # 轮询分配
        results_refs.append(worker.do_work.remote(item))

    results = ray.get(results_refs)
    print(f"\n  处理结果：")
    for r in results:
        print(f"    {r}")

    # 获取各 Worker 统计
    print(f"\n  各 Worker 统计：")
    for w in workers:
        stats = ray.get(w.get_stats.remote())
        print(f"    Worker-{stats['worker_id']}: 完成了 {stats['tasks_done']} 个任务")

    # ===========================================================
    # 示例 6：ActorHandle 传递给其他 Task
    # ===========================================================
    print_section("示例 6：将 ActorHandle 传递给其他 Task 或 Actor")

    # 创建一个 Logger Actor 作为集中式日志收集器
    logger = Logger.remote()
    print(f"  创建了 Logger Actor 用于集中收集日志")

    # 将 Logger 的 ActorHandle 传递给多个独立的 Task
    # ActorHandle 在序列化时只传递 Actor 的 ID 和地址信息
    # 多个 Task 可以共享同一个 Actor，所有日志都汇集到同一个 Logger 实例
    task_refs = [task_with_actor_handle.remote(logger, i) for i in range(4)]
    task_results = ray.get(task_refs)
    print(f"\n  Task 执行结果: {task_results}")

    # 从 Logger 获取所有收集到的日志
    logs = ray.get(logger.get_logs.remote())
    print(f"\n  Logger 收集的日志 ({len(logs)} 条)：")
    for log in logs:
        print(f"    📝 {log}")

    print(f"\n  💡 说明：4 个 Task 都通过 ActorHandle 向同一个 Logger Actor 写入日志，")
    print(f"         Logger 的内部 self.logs 列表汇集了所有日志，证明 Handle 传递正常工作。")

    # ===========================================================
    # 清理
    # ===========================================================
    ray.shutdown()

    print("\n" + "=" * 60)
    print("  Demo 04 执行完毕！")
    print("=" * 60)


if __name__ == "__main__":
    main()
