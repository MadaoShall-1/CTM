# KL balancing 修复与验证（2026-09-12）

## 完成状态

- seed 1 已跑到 300,056 步；2026-09-12 12:07:47（北京时间）按用户要求暂停。
- seed 0 保持 300,083 步；seed 2 仍 pending，attempts=0，没有启动续训。
- 修复在确认训练进程退出后进行；未启动新的训练或九项最终评测。

## 最小修正

`models.py` 的 `RSSM.kl_loss` 原先直接用 `balance` 加权 lhs 梯度项。
默认 `forward=False` 时 lhs 是 posterior、rhs 是 prior，导致默认 0.8
落在 posterior 端，而非 prior 端。

现按 KL 方向转换混合系数：

```python
mix = float(balance) if forward else (1.0 - float(balance))
loss = mix * loss_lhs + (1.0 - mix) * loss_rhs
```

默认配置现在为 posterior 0.2、prior 0.8 的 KL 直接梯度权重。
未改配置数值、学习率、Actor/Critic、动作语义、free/valid 处理，或
balance=0.5 时原有的两端半权重梯度行为。

参考：[官方 DreamerV2 KL 实现](https://github.com/danijar/dreamerv2/blob/main/dreamerv2/common/nets.py)。
两项 stop-gradient KL 的前向值相同，因此仅测试 loss 数值不能发现权重接反；
必须对 posterior/prior 分别检查梯度。

## 测试

新增 `tests/test_kl_balance.py`，覆盖默认反向 KL、正向 KL、balance 0/1/0.5、
padding、全 padding、free 阈值及 `tf.function` 图模式梯度。

- 修复前：8 个测试方法，出现 5 个失败项（含子测试），正确暴露反向 KL 的梯度错误。
- 修复后：新增 8 项测试全部通过。
- 既有 `tests/test_engineering.py`：29 项全部通过，含小模型更新与 checkpoint 集成测试。
- 合计 37 项测试通过；`git diff --check -- models.py` 通过。
- 全部测试使用 CPU；未加载或更新实验模型 checkpoint。

复现（在工程的 WSL 路径下运行）：

```bash
CUDA_VISIBLE_DEVICES=-1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 \
  /home/madao/.venvs/ctm-dreamer/bin/python -B -m unittest discover -s tests -p test_kl_balance.py -v
CUDA_VISIBLE_DEVICES=-1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 \
  /home/madao/.venvs/ctm-dreamer/bin/python -B -m unittest discover -s tests -p test_engineering.py -v
```

## 实验保全及后续边界

第二阶段原始目录 `/home/madao/ctm-runs/dreamerv2_stage2_20260911` 中的
15 个冻结源文件全部通过原始散列核验。三个 seed 的最新 checkpoint
在修复前后散列相同；seed 1 checkpoint 的变量全部有限，step=300056，
且最终保留快照与最新 checkpoint 散列一致。旧 replay、日志、曲线未修改。

- seed 0 checkpoint SHA-256：`ede30821b1f927c26bd866837198d49ed6a43745109ef02aaa12cbaed6c10fb0`
- seed 1 checkpoint SHA-256：`64183cf1fe3740c1f915b3702f256f9bb9bc885d72a4a50de7b43a9d8bbfcf48`
- seed 2 checkpoint SHA-256：`651f8265f6395d11e2ed0efd2bd1ed3dedfba482e348df7d2ab9e858b0d0e48c`
- 旧冻结 `models.py` SHA-256：`1f3cdc7e4fd4dda009454293dfec3b8a810364a351bfa7dbfa12b984b4785870`
- 修复后工作区 `models.py` SHA-256：`c95064dd88f7e6af4d327484b27754a4e15a9939717846c80fabacc4818b59b4`

本次运行代码相对冻结版本仅包含上述 KL 混合系数修正和解释注释。
所有其他已有未提交修改保持不动。

旧实验仍代表旧实现，修复不会追溯改变已训练权重。后续验证必须使用新的源码快照
和运行目录；直接 `--resume` 旧冻结 runner 不会启用此修复。没有授权时不重启训练。
测试通过只证明实现与梯度方向正确，不代表已经证明任务成功率恢复或该问题是唯一根因。
