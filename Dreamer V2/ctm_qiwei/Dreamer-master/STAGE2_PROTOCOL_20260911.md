# 第二阶段：保持配置不变，从 10 万步续至 30 万步

用户在确认训练已停止后要求“重新开始”。本次按从现有 checkpoint 续训理解，而非从零重训；已明确告知用户本阶段三个种子各续到累计 300,000 步，不自动扩大至全量。

## 固定条件

- 原始阶段：WSL `/home/madao/ctm-runs/dreamerv2_stage1_20260910`，保留不覆盖。
- 新阶段：WSL `/home/madao/ctm-runs/dreamerv2_stage2_20260911`。
- 从 seed 0/1/2 的 100,050 / 100,094 / 100,004 步 checkpoint 分别继续，约新增 20 万步/种子；单 GPU 串行运行。
- 训练源码沿用第一阶段的全部 15 个冻结文件，SHA-256 不变；原训练命令只改 `--steps` 和 `--logdir`。不改模型、学习率、BC/imagination 权重、batch、replay 或动作语义。
- 仍为 MOVE / TURN 两动作、范围内自动取货；不是显式 CATCH，也不是未经修改的 DreamerV2。
- 复制原 replay、64 条永久 demonstrations、模型/优化器/warmup 状态、历史 metrics 和 provenance；检查 checkpoint 散列、有限性、步数与 replay 一致性，并逐文件校验复制的 replay。
- warmup 已完成，续训不会重新执行 1000 次模型预热或 3000 次 Actor 预热。RNG、环境、采样器从种子重建，不承诺与不间断训练逐位一致；续训后的在线测试序列也会重启，跨阶段解读在线曲线时必须注明。

## checkpoint 与最终评测

- 每约 1000 步保存最新与上一份 checkpoint；每跨万步及阶段终点保留快照；额外保留此次 10 万步起点快照。
- 继续每 5000 步做 5 局在线测试。三个种子全部训练完成后，由程序自动评测，无需持续调用语言模型。
- 每个模型在 60000–60099 场景上做 100 局：这是与第一阶段的同场景配对比较，不能再标作新 holdout。
- 每个模型另在预先指定的 70000–70099 新场景上做 100 局独立评测，文件为 `final_holdout.json`。
- 每个模型继续相同的有界世界模型诊断（32 条 replay、每类 8 条 heldout、16 次 MC）。不同阶段 replay 抽样集合会变化，不能当作严格配对模型误差比较。
- 阶段结束后核验所有 checkpoint 和 9 份评测/诊断结果，重点比较 10 万与 30 万步的送达率、只选 TURN 的比例、取货后送达率，以及世界模型和 Actor/Critic 曲线。Critic loss 目标在变化，不能仅凭 loss 上升认定没有学习。
- 此阶段只检验“更多步数是否改善”，不在运行中调参，不自动启动第三阶段。

## 运行与恢复

主状态为新目录内的 `stage_status.json`、`batch_status.json`；`continuation_provenance.json` 记录父 checkpoint、起点步数和准备脚本散列。

沿用的冻结 runner 支持载入保存的批次命令，因此用 `--resume` 启动新准备的续训批次。它的固定说明/完成打印仍可能写 Stage 1；真实阶段以 `stage_status.json` 中的 Stage 2 protocol 和 300000 步训练命令为准，未为修改文字而改变冻结代码。

```bash
/home/madao/.venvs/ctm-dreamer/bin/python -B -u \
  /home/madao/ctm-runs/dreamerv2_stage2_20260911/source/run_validation_stage.py \
  --outdir /home/madao/ctm-runs/dreamerv2_stage2_20260911 --resume
```

首次准备由工程 `prepare_stage2.py` 完成，已有目标目录会拒绝覆盖；中断后只需上述恢复命令，不重新准备。启动前须确认旧训练进程已退出；锁防止并发写入。机器保持接电与唤醒，Codex 保持运行用于低频完成跟进；不修改全局电源配置。
