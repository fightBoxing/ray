# -*- coding: utf-8 -*-
"""
===========================================================
Demo 01: XGBoost 在 Ray 上的分布式训练与调优
===========================================================

本脚本演示机器学习算法 XGBoost 在 Ray 生态上的完整使用场景，帮助理解：
  1. Ray Data  — 分布式数据加载与预处理
  2. Ray Train — 使用 XGBoostTrainer 进行分布式训练
  3. Ray Tune  — 对 XGBoost 超参数进行自动调优
  4. 模型保存与加载 — 通过 Checkpoint 机制持久化模型
  5. 模型推理预测 — 加载训练好的模型执行推理

源码走读参考（对应走读指南 §4）：
  Ray Data:
    python/ray/data/read_api.py          → ray.data.from_pandas() 等数据读取 API
    python/ray/data/dataset.py           → Dataset 类（map_batches, train_test_split 等）
    python/ray/data/_internal/execution/streaming_executor.py → 流式执行引擎

  Ray Train:
    python/ray/train/xgboost/__init__.py → XGBoostTrainer, RayTrainReportCallback
    python/ray/train/xgboost/xgboost_trainer.py → XGBoostTrainer 实现
    python/ray/train/xgboost/_xgboost_utils.py  → RayTrainReportCallback 回调
    python/ray/train/data_parallel_trainer.py    → DataParallelTrainer 基类
    python/ray/train/_internal/backend_executor.py → Worker 编排与执行
    python/ray/train/_internal/worker_group.py     → WorkerGroup 管理

  Ray Tune:
    python/ray/tune/tuner.py             → Tuner（用户面入口）
    python/ray/tune/execution/tune_controller.py → TuneController 控制器
    python/ray/tune/search/searcher.py   → Searcher 搜索算法基类

依赖安装（如未安装）：
  pip install xgboost scikit-learn
"""

import os
import sys
import time
import tempfile
import traceback

import numpy as np


# =====================================================
# 辅助函数：检查依赖是否安装
# =====================================================
def _check_dependencies():
    """
    检查运行本 Demo 所需的 Python 依赖包。
    必须安装：ray, xgboost, scikit-learn, pandas
    """
    missing = []
    for pkg_name, import_name in [
        ("ray", "ray"),
        ("xgboost", "xgboost"),
        ("scikit-learn", "sklearn"),
        ("pandas", "pandas"),
    ]:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg_name)
    if missing:
        print(f"  ❌ 缺少依赖包: {', '.join(missing)}")
        print(f"     请执行: pip install {' '.join(missing)}")
        return False
    return True


# =====================================================
# 场景一：使用 Ray Data + 原生 XGBoost 进行单机训练
# =====================================================
def demo_ray_data_xgboost_basic():
    """
    演示如何使用 Ray Data 进行数据处理，然后用 XGBoost 进行训练。

    走读要点：
      - ray.data.from_pandas()：将 Pandas DataFrame 转换为 Ray Dataset
        源码：python/ray/data/read_api.py :: from_pandas()
      - Dataset.train_test_split()：将数据集划分为训练集和测试集
        源码：python/ray/data/dataset.py :: train_test_split()
      - Dataset.to_pandas()：将 Ray Dataset 转回 Pandas DataFrame
        源码：python/ray/data/dataset.py :: to_pandas()
    """
    import ray
    import pandas as pd
    import xgboost
    from sklearn.datasets import load_iris
    from sklearn.metrics import accuracy_score

    print("\n" + "─" * 60)
    print("  📌 场景一：Ray Data + 原生 XGBoost 单机训练")
    print("─" * 60)

    # -------------------------------------------------------
    # 步骤 1：使用 sklearn 加载经典 Iris 数据集
    # -------------------------------------------------------
    print("\n  [步骤 1] 加载 Iris 数据集...")
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target
    print(f"    数据集形状: {df.shape}")
    print(f"    特征列: {iris.feature_names}")
    print(f"    类别数: {len(set(iris.target))}")

    # -------------------------------------------------------
    # 步骤 2：将 Pandas DataFrame 转为 Ray Dataset
    # -------------------------------------------------------
    # ray.data.from_pandas() 会将 DataFrame 包装为一个 Ray Dataset 对象。
    # 内部流程：
    #   python/ray/data/read_api.py :: from_pandas()
    #     → 创建一个 MaterializedDataset
    #     → 底层以 Arrow Table 格式存储
    #
    # Ray Dataset 支持惰性执行、分片处理和分布式操作。
    print("\n  [步骤 2] 将 DataFrame 转换为 Ray Dataset...")
    dataset = ray.data.from_pandas(df)
    print(f"    Ray Dataset: {dataset}")
    print(f"    数据块(Block)数量: {dataset.num_blocks()}")

    # -------------------------------------------------------
    # 步骤 3：使用 Ray Data 进行数据预处理
    # -------------------------------------------------------
    # Dataset.map_batches() 是 Ray Data 最核心的算子之一。
    # 它对每个数据块执行用户定义的函数，支持批量向量化处理。
    # 源码：python/ray/data/dataset.py :: map_batches()
    #   → 生成 MapBatches 逻辑算子
    #   → 经过优化器后转为物理算子并执行
    print("\n  [步骤 3] 数据预处理 — 特征标准化（通过 map_batches）...")

    def standardize_features(batch: pd.DataFrame) -> pd.DataFrame:
        """对特征列进行 Z-score 标准化"""
        feature_cols = [c for c in batch.columns if c != "target"]
        for col in feature_cols:
            mean = batch[col].mean()
            std = batch[col].std()
            if std > 0:
                batch[col] = (batch[col] - mean) / std
        return batch

    dataset = dataset.map_batches(standardize_features, batch_format="pandas")
    print(f"    标准化后 Dataset: {dataset}")

    # -------------------------------------------------------
    # 步骤 4：划分训练集与测试集
    # -------------------------------------------------------
    # Dataset.train_test_split() 按比例将数据集分为两部分。
    # 源码：python/ray/data/dataset.py :: train_test_split()
    #   → 内部调用 split_proportionately()
    print("\n  [步骤 4] 划分训练集(80%) / 测试集(20%)...")
    train_ds, test_ds = dataset.train_test_split(test_size=0.2, seed=42)
    print(f"    训练集: {train_ds.count()} 条")
    print(f"    测试集: {test_ds.count()} 条")

    # -------------------------------------------------------
    # 步骤 5：转回 Pandas 并用原生 XGBoost 训练
    # -------------------------------------------------------
    print("\n  [步骤 5] 使用原生 XGBoost 训练分类模型...")

    train_df = train_ds.to_pandas()
    test_df = test_ds.to_pandas()

    train_X = train_df.drop("target", axis=1)
    train_y = train_df["target"]
    test_X = test_df.drop("target", axis=1)
    test_y = test_df["target"]

    dtrain = xgboost.DMatrix(train_X, label=train_y)
    dtest = xgboost.DMatrix(test_X, label=test_y)

    # XGBoost 训练参数
    params = {
        "objective": "multi:softprob",   # 多分类概率输出
        "num_class": 3,                  # Iris 有 3 个类别
        "eval_metric": "mlogloss",       # 评估指标：多类对数损失
        "max_depth": 4,                  # 树的最大深度
        "eta": 0.3,                      # 学习率
        "tree_method": "hist",           # 使用直方图近似，速度更快
        "seed": 42,
    }

    start_time = time.time()
    bst = xgboost.train(
        params,
        dtrain=dtrain,
        evals=[(dtrain, "train"), (dtest, "test")],
        num_boost_round=50,
        verbose_eval=10,  # 每 10 轮打印一次
    )
    train_time = time.time() - start_time

    # 评估
    preds = bst.predict(dtest)
    pred_labels = np.argmax(preds, axis=1)
    accuracy = accuracy_score(test_y, pred_labels)

    print(f"\n    ✅ 训练完成！耗时: {train_time:.2f}s")
    print(f"    ✅ 测试集准确率: {accuracy:.4f} ({accuracy * 100:.1f}%)")
    print(f"    ✅ 特征重要性 (gain):")
    importance = bst.get_score(importance_type="gain")
    for feat, score in sorted(importance.items(), key=lambda x: -x[1]):
        print(f"       {feat}: {score:.4f}")

    return bst


# =====================================================
# 场景二：使用 Ray Train XGBoostTrainer 分布式训练
# =====================================================
def demo_ray_train_xgboost():
    """
    演示使用 Ray Train 的 XGBoostTrainer 进行分布式数据并行训练。

    走读要点：
      - XGBoostTrainer 继承自 DataParallelTrainer
        源码：python/ray/train/xgboost/xgboost_trainer.py :: XGBoostTrainer
      - 训练流程：
        Trainer.fit()
          → DataParallelTrainer._validate()
            → BackendExecutor.start()
              → 创建 WorkerGroup（一组 Ray Actor）
              → 设置分布式 XGBoost 环境（rabit 通信后端）
              → 在每个 Worker 上执行 train_func()
      - RayTrainReportCallback 在每轮训练后上报 metrics + checkpoint
        源码：python/ray/train/xgboost/_xgboost_utils.py :: RayTrainReportCallback
      - ray.train.get_dataset_shard() 获取当前 Worker 的数据分片
        源码：python/ray/train/_internal/session.py :: get_dataset_shard()
    """
    import ray
    import ray.train
    import pandas as pd
    import xgboost
    from sklearn.datasets import load_iris
    from sklearn.metrics import accuracy_score
    from ray.train.xgboost import XGBoostTrainer, RayTrainReportCallback
    from ray.train import ScalingConfig, RunConfig

    print("\n" + "─" * 60)
    print("  📌 场景二：Ray Train XGBoostTrainer 分布式训练")
    print("─" * 60)

    # -------------------------------------------------------
    # 步骤 1：准备 Ray Dataset
    # -------------------------------------------------------
    print("\n  [步骤 1] 准备训练数据为 Ray Dataset...")
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target

    dataset = ray.data.from_pandas(df)
    train_ds, eval_ds = dataset.train_test_split(test_size=0.2, seed=42)
    print(f"    训练集: {train_ds.count()} 条")
    print(f"    验证集: {eval_ds.count()} 条")

    # -------------------------------------------------------
    # 步骤 2：定义分布式训练函数 (train_func)
    # -------------------------------------------------------
    # 该函数在每个 Worker 上独立执行。
    # Ray Train 会自动：
    #   1. 将数据集按 Worker 数量均匀分片
    #   2. 设置 XGBoost 的 rabit 通信后端
    #   3. 在训练结束后收集和合并结果
    #
    # 关键 API：
    #   ray.train.get_dataset_shard("train") — 获取当前 Worker 的数据分片
    #   ray.train.report(metrics, checkpoint) — 上报训练指标和检查点
    #
    # 内部调用链：
    #   XGBoostTrainer.fit()
    #     → DataParallelTrainer.training_loop()
    #       → BackendExecutor.start_training()
    #         → 在每个 Worker Actor 中调用 train_func()
    print("\n  [步骤 2] 定义分布式训练函数...")

    def train_func():
        """每个分布式 Worker 执行的训练函数"""
        # 获取当前 Worker 的数据分片
        # 源码：python/ray/train/_internal/session.py :: get_dataset_shard()
        train_shard = ray.train.get_dataset_shard("train")
        eval_shard = ray.train.get_dataset_shard("eval")

        # 将 Ray Dataset 分片转为 Pandas DataFrame
        train_df = train_shard.materialize().to_pandas()
        eval_df = eval_shard.materialize().to_pandas()

        train_X = train_df.drop("target", axis=1)
        train_y = train_df["target"]
        eval_X = eval_df.drop("target", axis=1)
        eval_y = eval_df["target"]

        dtrain = xgboost.DMatrix(train_X, label=train_y)
        deval = xgboost.DMatrix(eval_X, label=eval_y)

        # XGBoost 训练参数
        params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "max_depth": 4,
            "eta": 0.3,
            "tree_method": "hist",
            "seed": 42,
        }

        # 使用 RayTrainReportCallback 在每轮训练后自动上报 metrics 和 checkpoint
        # 源码：python/ray/train/xgboost/_xgboost_utils.py
        #   RayTrainReportCallback.after_iteration()
        #     → ray.train.report(metrics, checkpoint)
        #       → Session._report()
        bst = xgboost.train(
            params,
            dtrain=dtrain,
            evals=[(deval, "eval")],
            num_boost_round=30,
            callbacks=[
                RayTrainReportCallback(
                    metrics={"eval_mlogloss": "eval-mlogloss"},
                    frequency=5,            # 每 5 轮保存一次 checkpoint
                    checkpoint_at_end=True,  # 训练结束时保存最终 checkpoint
                )
            ],
        )

    # -------------------------------------------------------
    # 步骤 3：配置 ScalingConfig（分布式扩展参数）
    # -------------------------------------------------------
    # ScalingConfig 定义了分布式训练的规模：
    #   num_workers  — 训练 Worker 数量（每个是一个 Ray Actor）
    #   use_gpu      — 是否使用 GPU
    #   resources_per_worker — 每个 Worker 的资源配额
    #
    # 源码：python/ray/train/scaling_config.py :: ScalingConfig
    print("\n  [步骤 3] 配置训练规模（2 个 Worker，每个 1 CPU）...")
    scaling_config = ScalingConfig(
        num_workers=2,
        use_gpu=False,
        resources_per_worker={"CPU": 1},
    )

    # -------------------------------------------------------
    # 步骤 4：创建 XGBoostTrainer 并启动训练
    # -------------------------------------------------------
    # XGBoostTrainer 继承自 DataParallelTrainer → BaseTrainer
    # 源码：python/ray/train/xgboost/xgboost_trainer.py :: XGBoostTrainer
    #
    # 训练内部流程：
    #   trainer.fit()
    #     → BaseTrainer.fit()
    #       → DataParallelTrainer._validate()
    #       → BackendExecutor.start()
    #         → 创建 WorkerGroup
    #         → XGBoostConfig.backend_cls().on_training_start()
    #           → 设置 rabit tracker（分布式通信协调器）
    #       → BackendExecutor.start_training(train_func)
    #         → 在 Worker 上执行 train_func()
    #       → 收集结果 → 返回 Result 对象
    print("\n  [步骤 4] 启动 XGBoostTrainer 分布式训练...")

    trainer = XGBoostTrainer(
        train_func,
        scaling_config=scaling_config,
        datasets={"train": train_ds, "eval": eval_ds},
        run_config=RunConfig(
            name="xgboost_iris_demo",
            # 本地演示无需配置远程存储
        ),
    )

    start_time = time.time()
    result = trainer.fit()
    train_time = time.time() - start_time

    print(f"\n    ✅ 分布式训练完成！耗时: {train_time:.2f}s")
    print(f"    ✅ 训练结果路径: {result.path}")
    print(f"    ✅ 最终指标: {result.metrics}")

    # -------------------------------------------------------
    # 步骤 5：从 Checkpoint 加载模型并推理
    # -------------------------------------------------------
    # RayTrainReportCallback.get_model() 从 Checkpoint 中恢复 XGBoost Booster 模型
    # 源码：python/ray/train/xgboost/_xgboost_utils.py :: get_model()
    #   → 解压 checkpoint 目录
    #   → Booster().load_model(path)
    print("\n  [步骤 5] 从 Checkpoint 加载模型并推理...")

    if result.checkpoint:
        booster = RayTrainReportCallback.get_model(result.checkpoint)
        print(f"    ✅ 模型加载成功！Booster 已训练 {booster.num_boosted_rounds()} 轮")

        # 用测试数据推理
        test_df = eval_ds.to_pandas()
        test_X = test_df.drop("target", axis=1)
        test_y = test_df["target"]
        dtest = xgboost.DMatrix(test_X)

        preds = booster.predict(dtest)
        pred_labels = np.argmax(preds, axis=1)
        accuracy = accuracy_score(test_y, pred_labels)
        print(f"    ✅ 推理测试集准确率: {accuracy:.4f} ({accuracy * 100:.1f}%)")
    else:
        print("    ⚠️  未找到 Checkpoint，跳过模型加载")

    return result


# =====================================================
# 场景三：使用 Ray Tune 对 XGBoost 进行超参数调优
# =====================================================
def demo_ray_tune_xgboost():
    """
    演示使用 Ray Tune 对 XGBoost 进行超参数搜索。

    走读要点：
      - Tuner 是 Ray Tune 的用户面入口
        源码：python/ray/tune/tuner.py :: Tuner
      - Tuner.fit() 内部调用 TuneController.run() 执行搜索
        源码：python/ray/tune/execution/tune_controller.py :: TuneController
      - 搜索流程：
        Tuner.fit()
          → TuneController.run()
            → Searcher.suggest()    — 生成超参配置
            → 创建 Trial            — 启动训练 Actor
            → Trial 执行            — 训练 + 上报结果
            → Scheduler.on_trial_result() — 决定 Trial 状态
            → 循环直到完成
      - tune.choice() / tune.uniform() 等定义搜索空间
        源码：python/ray/tune/search/sample.py
    """
    import ray
    import pandas as pd
    import xgboost
    from sklearn.datasets import load_iris
    from ray import tune
    from ray.tune import Tuner, TuneConfig, RunConfig

    print("\n" + "─" * 60)
    print("  📌 场景三：Ray Tune + XGBoost 超参数调优")
    print("─" * 60)

    # -------------------------------------------------------
    # 步骤 1：准备数据
    # -------------------------------------------------------
    print("\n  [步骤 1] 准备 Iris 数据集...")
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target

    # 用 ray.put() 将数据放入 Object Store，供所有 Trial 共享
    # 避免每个 Trial 重复加载数据
    # 源码：python/ray/_private/worker.py :: put()
    #   → CoreWorker::Put()
    #     → 小对象 → MemoryStore / 大对象 → PlasmaStore
    df_ref = ray.put(df)
    print(f"    数据已放入 Object Store, ObjectRef: {df_ref}")

    # -------------------------------------------------------
    # 步骤 2：定义可调参训练函数
    # -------------------------------------------------------
    # 训练函数接收一个 config 字典，包含由 Ray Tune 搜索器生成的超参数。
    # 函数内部使用 ray.train.report() 上报每轮的指标。
    print("\n  [步骤 2] 定义可调参训练函数...")

    def tune_train_func(config):
        """
        用于 Ray Tune 的 XGBoost 训练函数。
        config 参数由 Tune 的搜索空间自动填充。
        """
        # 从 Object Store 获取共享数据
        df = ray.get(df_ref)
        from sklearn.model_selection import train_test_split as sk_split

        train_df, eval_df = sk_split(df, test_size=0.2, random_state=42)

        train_X = train_df.drop("target", axis=1)
        train_y = train_df["target"]
        eval_X = eval_df.drop("target", axis=1)
        eval_y = eval_df["target"]

        dtrain = xgboost.DMatrix(train_X, label=train_y)
        deval = xgboost.DMatrix(eval_X, label=eval_y)

        # 使用 Tune 提供的超参数
        params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "max_depth": config["max_depth"],
            "eta": config["eta"],
            "subsample": config["subsample"],
            "colsample_bytree": config["colsample_bytree"],
            "tree_method": "hist",
            "seed": 42,
        }

        # 自定义 XGBoost Callback，通过 ray.tune.report() 上报指标
        # 注意：在 Tune 上下文中不能使用 RayTrainReportCallback（它依赖 ray.train.get_context()），
        # 因此我们自定义一个简单的 callback 来上报。
        class TuneReportCallback(xgboost.callback.TrainingCallback):
            """在每轮训练后调用 ray.tune.report() 上报 eval 指标"""
            def after_iteration(self, model, epoch, evals_log):
                # evals_log 格式：{"eval": {"mlogloss": [val1, val2, ...]}}
                if "eval" in evals_log and "mlogloss" in evals_log["eval"]:
                    mlogloss = evals_log["eval"]["mlogloss"][-1]
                    ray.tune.report({"eval_mlogloss": mlogloss})
                return False  # 返回 False 表示不终止训练

        bst = xgboost.train(
            params,
            dtrain=dtrain,
            evals=[(deval, "eval")],
            num_boost_round=config["num_boost_round"],
            callbacks=[TuneReportCallback()],
        )

    # -------------------------------------------------------
    # 步骤 3：定义搜索空间
    # -------------------------------------------------------
    # tune.choice()   — 从列表中随机选择一个值
    # tune.uniform()  — 从均匀分布中采样
    # tune.loguniform() — 从对数均匀分布中采样（适合学习率等跨数量级的参数）
    # 源码：python/ray/tune/search/sample.py
    print("\n  [步骤 3] 定义超参数搜索空间...")

    search_space = {
        "max_depth": tune.choice([3, 4, 5, 6]),
        "eta": tune.loguniform(1e-3, 0.3),
        "subsample": tune.uniform(0.6, 1.0),
        "colsample_bytree": tune.uniform(0.6, 1.0),
        "num_boost_round": tune.choice([20, 50, 100]),
    }

    for k, v in search_space.items():
        print(f"    {k}: {v}")

    # -------------------------------------------------------
    # 步骤 4：创建 Tuner 并执行调优
    # -------------------------------------------------------
    # Tuner 是 Ray Tune 的核心入口：
    #   Tuner.fit()
    #     → TuneController.run()
    #       → 循环：Searcher.suggest() → 创建 Trial → 执行 → 收集结果
    #
    # TuneConfig:
    #   num_samples  — 总共尝试多少组超参组合
    #   metric       — 优化目标指标名
    #   mode         — "min" (最小化) 或 "max" (最大化)
    #
    # 源码：python/ray/tune/tuner.py :: Tuner.fit()
    print("\n  [步骤 4] 启动 Ray Tune 超参数搜索（4 组 Trial）...")

    tuner = Tuner(
        tune_train_func,
        param_space=search_space,
        tune_config=TuneConfig(
            num_samples=4,           # 尝试 4 组超参组合
            metric="eval_mlogloss",  # 优化目标：验证集多类对数损失
            mode="min",              # 最小化损失
        ),
        run_config=RunConfig(
            name="xgboost_tune_demo",
            verbose=0,               # 关闭冗余输出，避免版本兼容问题
        ),
    )

    start_time = time.time()
    tune_results = tuner.fit()
    tune_time = time.time() - start_time

    # -------------------------------------------------------
    # 步骤 5：分析调优结果
    # -------------------------------------------------------
    print(f"\n    ✅ 超参数调优完成！耗时: {tune_time:.2f}s")
    print(f"    ✅ 总 Trial 数: {len(tune_results)}")

    # 获取最佳结果
    best_result = tune_results.get_best_result(metric="eval_mlogloss", mode="min")
    print(f"\n    🏆 最佳 Trial:")
    print(f"       eval_mlogloss: {best_result.metrics.get('eval_mlogloss', 'N/A')}")
    print(f"       最佳超参数:")
    best_config = best_result.config
    for k, v in best_config.items():
        if isinstance(v, float):
            print(f"         {k}: {v:.6f}")
        else:
            print(f"         {k}: {v}")

    # 打印所有 Trial 的结果对比
    print(f"\n    📊 所有 Trial 结果对比:")
    print(f"    {'Trial':>8s}  {'eval_mlogloss':>15s}  {'max_depth':>10s}  {'eta':>10s}  {'subsample':>10s}")
    print(f"    {'─' * 8}  {'─' * 15}  {'─' * 10}  {'─' * 10}  {'─' * 10}")
    for i, result in enumerate(tune_results):
        metrics = result.metrics or {}
        config = result.config or {}
        mlogloss = metrics.get("eval_mlogloss", "N/A")
        if isinstance(mlogloss, float):
            mlogloss_str = f"{mlogloss:.6f}"
        else:
            mlogloss_str = str(mlogloss)
        print(
            f"    {i + 1:>8d}  {mlogloss_str:>15s}  "
            f"{str(config.get('max_depth', 'N/A')):>10s}  "
            f"{config.get('eta', 0):.6f}  "
            f"{config.get('subsample', 0):.4f}"
        )

    return tune_results


# =====================================================
# 主入口
# =====================================================
def main():
    print("=" * 60)
    print("  Demo 01: XGBoost 在 Ray 上的分布式训练与调优")
    print("  （阶段二 — 进阶走读测试）")
    print("=" * 60)

    # 检查依赖
    if not _check_dependencies():
        sys.exit(1)

    import ray

    # 初始化 Ray（统一管理生命周期）
    if ray.is_initialized():
        ray.shutdown()

    print("\n[初始化] 启动 Ray 集群...")
    ray.init(num_cpus=4)
    print(f"  ✅ Ray 已启动，Dashboard: {ray.get_runtime_context().worker.current_cluster_and_job}")

    results = {}
    total_start = time.time()

    # --------------------------------------------------
    # 场景一：Ray Data + 原生 XGBoost
    # --------------------------------------------------
    try:
        print("\n\n" + "█" * 60)
        print("  ▶ 运行场景一：Ray Data + 原生 XGBoost 单机训练")
        print("█" * 60)
        bst = demo_ray_data_xgboost_basic()
        results["场景一"] = "✅ 成功"
    except Exception as e:
        print(f"\n  ❌ 场景一执行失败: {e}")
        traceback.print_exc()
        results["场景一"] = f"❌ 失败: {e}"

    # --------------------------------------------------
    # 场景二：Ray Train XGBoostTrainer
    # --------------------------------------------------
    try:
        print("\n\n" + "█" * 60)
        print("  ▶ 运行场景二：Ray Train XGBoostTrainer 分布式训练")
        print("█" * 60)
        train_result = demo_ray_train_xgboost()
        results["场景二"] = "✅ 成功"
    except Exception as e:
        print(f"\n  ❌ 场景二执行失败: {e}")
        traceback.print_exc()
        results["场景二"] = f"❌ 失败: {e}"

    # --------------------------------------------------
    # 场景三：Ray Tune + XGBoost
    # --------------------------------------------------
    try:
        print("\n\n" + "█" * 60)
        print("  ▶ 运行场景三：Ray Tune + XGBoost 超参数调优")
        print("█" * 60)
        tune_results = demo_ray_tune_xgboost()
        results["场景三"] = "✅ 成功"
    except Exception as e:
        print(f"\n  ❌ 场景三执行失败: {e}")
        traceback.print_exc()
        results["场景三"] = f"❌ 失败: {e}"

    # --------------------------------------------------
    # 总结
    # --------------------------------------------------
    total_time = time.time() - total_start

    print("\n\n" + "=" * 60)
    print("  📊 执行总结")
    print("=" * 60)
    for scenario, status in results.items():
        print(f"  {scenario}: {status}")
    print(f"\n  总耗时: {total_time:.1f}s")

    # 关闭 Ray
    ray.shutdown()
    print(f"\n  ✅ Ray 已关闭。ray.is_initialized() = {ray.is_initialized()}")

    print("\n" + "=" * 60)
    print("  Demo 01 (阶段二) 执行完毕！")
    print("=" * 60)


if __name__ == "__main__":
    main()
