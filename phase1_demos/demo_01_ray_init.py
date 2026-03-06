# -*- coding: utf-8 -*-
"""
===========================================================
Demo 01: ray.init() 初始化流程
===========================================================

本脚本演示 Ray 集群的初始化与连接机制，帮助理解以下内容：
  1. ray.init() 启动本地集群
  2. 集群关键信息获取（Dashboard URL、节点数量、可用资源）
  3. 常用初始化参数说明
  4. ray.is_initialized() 检测集群状态
  5. ray.shutdown() 正确关闭集群

源码调用链（走读参考）：
  python/ray/worker.py::init()
    → python/ray/node.py::Node.__init__()
      → python/ray/_raylet.pyx::CoreWorkerProcess
        → src/ray/core_worker/core_worker.cc::CoreWorker()
"""

import ray
import sys


def main():
    print("=" * 60)
    print("  Demo 01: ray.init() — Ray 集群初始化流程")
    print("=" * 60)

    # -------------------------------------------------------
    # 步骤 1：检测是否已有运行中的 Ray 集群
    # -------------------------------------------------------
    # ray.is_initialized() 会检查当前进程是否已连接到某个 Ray 集群。
    # 源码位置：python/ray/worker.py::is_initialized()
    #   → 实际检查 global_worker.connected 标志
    if ray.is_initialized():
        print("[提示] 检测到当前进程已连接到一个 Ray 集群。")
        print("       将先关闭已有连接，再重新初始化。")
        ray.shutdown()

    # -------------------------------------------------------
    # 步骤 2：调用 ray.init() 启动本地集群
    # -------------------------------------------------------
    # ray.init() 是 Ray 最核心的入口函数，其内部流程如下：
    #
    # 1) 解析参数，创建 RayParams 配置对象
    #    源码：python/ray/worker.py::init() → _init_ray_params()
    #
    # 2) 启动本地 Raylet 节点（包含 GCS、Object Store、Raylet 进程）
    #    源码：python/ray/node.py::Node.__init__()
    #      → Node._start_ray_processes()
    #        → 启动 GCS Server (gcs_server)
    #        → 启动 Raylet (raylet)
    #        → 启动 Plasma Object Store
    #
    # 3) 初始化当前进程的 CoreWorker（C++ 层，通过 Cython 绑定）
    #    源码：python/ray/_raylet.pyx::CoreWorkerProcess.__init__()
    #      → src/ray/core_worker/core_worker.cc::CoreWorker()
    #
    # 常用参数说明：
    #   num_cpus       - 覆盖自动检测的 CPU 核数（仅影响资源调度，不限制物理核使用）
    #   num_gpus       - 覆盖自动检测的 GPU 数量
    #   logging_level  - 日志级别，可选 logging.DEBUG / INFO / WARNING 等
    #   namespace      - 命名空间，不同 namespace 下的 Actor 互相隔离
    #   runtime_env    - 运行时环境配置（pip 包、环境变量、工作目录等）
    #   address        - 连接已有集群的地址，如 "auto" 或 "ray://<head-ip>:10001"
    #   _temp_dir      - 自定义临时文件目录（日志、socket 等）
    #
    print("\n[步骤 2] 正在调用 ray.init() 启动本地 Ray 集群...")
    context = ray.init(
        num_cpus=4,              # 示例：限制为 4 个 CPU 资源单位
        # num_gpus=0,            # 如有 GPU 可取消注释
        # logging_level="INFO",  # 日志级别
        # namespace="demo",      # 命名空间
    )

    print(f"  ✅ Ray 集群启动成功！")
    print(f"  ✅ ray.is_initialized() = {ray.is_initialized()}")

    # -------------------------------------------------------
    # 步骤 3：打印集群关键信息
    # -------------------------------------------------------
    # ray.init() 返回一个 RayContext 对象，包含集群的元数据。
    # 源码位置：python/ray/worker.py::RayContext
    print("\n[步骤 3] 集群关键信息：")
    print(f"  Dashboard URL     : {context.dashboard_url}")

    # 通过 ray.nodes() 获取集群节点列表
    # 源码位置：python/ray/worker.py::nodes()
    #   → 内部调用 GCS Client 从 GCS Server 拉取节点信息
    nodes = ray.nodes()
    alive_nodes = [n for n in nodes if n.get("Alive", False)]
    print(f"  节点总数          : {len(nodes)}")
    print(f"  存活节点数        : {len(alive_nodes)}")

    # 通过 ray.cluster_resources() 获取集群总资源
    # 通过 ray.available_resources() 获取集群当前可用资源
    # 源码：python/ray/worker.py::cluster_resources() / available_resources()
    #   → GCS Client::GetAllResourceUsage()
    cluster_res = ray.cluster_resources()
    available_res = ray.available_resources()
    print(f"\n  集群总资源：")
    for key, val in sorted(cluster_res.items()):
        print(f"    {key:20s}: {val}")
    print(f"\n  当前可用资源：")
    for key, val in sorted(available_res.items()):
        print(f"    {key:20s}: {val}")

    # -------------------------------------------------------
    # 步骤 4：打印节点详情
    # -------------------------------------------------------
    print("\n[步骤 4] 节点详情：")
    for i, node in enumerate(alive_nodes):
        print(f"  节点 {i + 1}:")
        print(f"    NodeID          : {node.get('NodeID', 'N/A')}")
        print(f"    NodeManagerAddress: {node.get('NodeManagerAddress', 'N/A')}")
        print(f"    ObjectStoreSocket: {node.get('ObjectStoreSocketName', 'N/A')}")
        print(f"    Resources       : {node.get('Resources', {})}")

    # -------------------------------------------------------
    # 步骤 5：关闭集群
    # -------------------------------------------------------
    # ray.shutdown() 会断开当前进程与 Ray 集群的连接。
    # 如果是 ray.init() 启动的本地集群，还会终止所有 Ray 后台进程。
    # 源码位置：python/ray/worker.py::shutdown()
    #   → CoreWorkerProcess.shutdown()
    #     → 通知 GCS 进行清理
    print("\n[步骤 5] 正在调用 ray.shutdown() 关闭集群...")
    ray.shutdown()
    print(f"  ✅ 集群已关闭。ray.is_initialized() = {ray.is_initialized()}")

    print("\n" + "=" * 60)
    print("  Demo 01 执行完毕！")
    print("=" * 60)


if __name__ == "__main__":
    main()
