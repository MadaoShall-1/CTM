# KL 修复后重启：seed 1 有界续训

2026-09-12 用户要求 KL 修复后重启训练。本轮先验证 seed 1，结束后停止，不自动启动其他 seed 或独立评测。

- 新目录：`/home/madao/ctm-runs/dreamerv2_klfix_seed1_20260912`。
- 从第一阶段 seed 1 的 100,094 步 checkpoint 续到累计 300,000 步（完整 episode 可能略超）。不是从零训练，也不是从旧版 30 万步继续增加预算。
- 新目录前 100,094 步的历史记录仍属于旧 KL；之后才使用修复后的 prior 0.8 / posterior 0.2 直接 KL 梯度权重。
- 相对旧冻结训练源码，仅 `models.py` 的 KL 混合系数修正；其余 14 个冻结文件散列一致。
- 保留模型、优化器、已完成的预热状态、replay 和 64 条永久示范。MOVE/TURN + 自动取货保持不变。
- 最新 checkpoint 约每 1,000 步更新，保留上一份；每跨 10,000 步及终点留快照，另保留 100,094 步起点。
- 每 5,000 步在线测试 5 局；单 seed 跑完后 batch runner 自动退出。没有阶段 runner，因此不会自动执行旧阶段的 9 项评测。
- 已再次通过 8 项 KL 梯度测试；准备脚本校验了父 checkpoint 的散列、步数、有限性、预热、replay 步数及复制后所有 replay 文件散列。
- 父 checkpoint SHA256：`50aeb9ba46946db5b1f7b8bfdae4f1ba3875e532422ba0efa07e85d17812b530`。
- 修复 `models.py` SHA256：`c95064dd88f7e6af4d327484b27754a4e15a9939717846c80fabacc4818b59b4`。

## 解读边界

这是相同 10 万步起点、其余参数不变的修复续训对照。起点权重已受旧 KL 影响，因此不能代表“从零就使用正确 KL”的效果；RNG、环境和采样器会重新初始化，不承诺逐位可复现。在线滑动成功率不代替固定场景独立评测。旧实验目录、checkpoint 和曲线均未覆盖。

## 状态与恢复

主状态：新目录下 `batch_status.json`；seed 日志：`seed_1/console.log` 和 `seed_1/metrics.jsonl`。`continuation_provenance.json` 记录本轮来源与源码散列。

仅在用户要求恢复、且确认没有运行中的同目录进程时使用：

```bash
/home/madao/.venvs/ctm-dreamer/bin/python -B -u \
  /home/madao/ctm-runs/dreamerv2_klfix_seed1_20260912/source/run_batch.py \
  --outdir /home/madao/ctm-runs/dreamerv2_klfix_seed1_20260912 --resume
```

旧自动跟进仍为暂停，本次没有启用定时模型调用。
