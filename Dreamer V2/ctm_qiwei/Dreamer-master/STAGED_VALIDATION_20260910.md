# 分阶段长跑验证：第一阶段

## 固定协议

- 训练种子 0、1、2，各 100,000 个累计环境步，单 GPU 串行；预算包含 demonstration 和 prefill，结束允许一个 episode 的小幅超出。
- 当前任务：MOVE / TURN 两动作，范围内自动取货；模型保持 RSSM 400、32×32，batch 50×20，imagination horizon 15。
- 保持当前的 exact-vector actor、64 条 controller demonstrations、BC warmup / 在线 BC；这是 UAV 适配版本，不应写成未经修改的 DreamerV2 复现。
- 每约 1,000 步原子保存 `variables.pkl`，保留 `variables.prev.pkl`；warmup 每 100 次更新保存进度。每跨过 10,000 步以及终点额外保留 `checkpoints/step_*.pkl`，文件名是实际 checkpoint 步数。
- 每 5,000 步做 5 局在线评测。三次训练全部完成后，每个最终模型在相同、预先固定且与训练 demonstrations 分离的 60000–60099 场景种子上评测 100 局。
- 最后每个模型做有界世界模型诊断：32 个 replay episodes、每类 8 个 heldout episodes、16 次 Monte Carlo；诊断场景 20000 起，仅作机制诊断，不充当新的独立泛化成绩。
- 训练、评测和诊断全部由本地程序执行，不调用语言模型；定时跟进仅低频读状态，异常或完成时才通知。第一阶段结束后分析，不自动扩大训练预算。

## 位置与执行

WSL Ubuntu 原生磁盘运行目录：

`/home/madao/ctm-runs/dreamerv2_stage1_20260910`

Windows 可通过 `\\wsl.localhost\Ubuntu\home\madao\ctm-runs\dreamerv2_stage1_20260910` 访问。

`source/` 是包含训练、评测、诊断、编排脚本的 SHA-256 冻结副本；运行后修改工程不会改变本批实验。

- `stage_status.json`：全流程状态（training / evaluating / completed / failed / interrupted）。
- `batch_status.json`：每个种子状态、PID、尝试次数、训练命令、源码散列。
- `seed_N/`：replay、固定 demonstrations、metrics、console、checkpoint、最终评测和诊断。
- `checkpoint_manifest.json`：全部完成后生成的 checkpoint 文件大小与 SHA-256 清单。
- 工程 `outputs/stage1_20260910/launcher.*.log`：Windows 启动进程输出；详细训练输出在每个 seed 的 console.log。

中断后先确认旧训练进程已退出，再使用冻结脚本恢复（不能并发启动）：

```bash
/home/madao/.venvs/ctm-dreamer/bin/python -B \
  /home/madao/ctm-runs/dreamerv2_stage1_20260910/source/run_validation_stage.py \
  --outdir /home/madao/ctm-runs/dreamerv2_stage1_20260910 --resume
```

恢复保留模型、优化器和 warmup 进度，但不承诺随机数、环境和 replay 采样器的逐位一致恢复。不自动忽略数值错误、不覆盖坏 checkpoint 后强行继续；先分析异常。

## 完成后的统一分析

检查三种子是否正常结束、训练和梯度是否有限、checkpoint 和 demonstrations 完整性、耗时和异常重启。报告每种子的最终 success / pickup / 越界 / timeout / 参数饱和率及种子间波动；结合在线曲线与世界模型动作响应，判断应继续第二阶段还是先修复。最终 holdout 不用于本轮调参。不能用短跑工程通过替代策略收敛结论。

机器需接电且保持唤醒；自动分析需要 Codex 应用运行。本轮不修改全局电源设置。
