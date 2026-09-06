# UAV 复现实验结果分析与当前场景问题

整理日期：2026-09-06。范围：CTM 下的 `her_mpdqn_reproduction`，供后续基线修正及 CT-WM / CT-RSSM 实验设计使用。

## 1. 当前结论与证据范围

上一轮边界修正后的全量实验已经完成，但没有复现预期的算法排序：Direct 四种方法总体都能完成任务；Relay 只有 MP-DQN 的最终策略表现较好，P-DQN 与 HER 方法存在退化或失效。旧实现的观测接口、HER 语义及更新频率与当前重建版本有差异，所以这轮结果应保留为诊断数据，不能作为最终论文复现结论。

当前版本能够训练、评估和保存断点，但代码检查确认了 HER 目标与距离特征不一致、Relay 奖励重算接口不一致等问题。最新短跑展示了 Direct HER-MPDQN 的学习信号，也暴露了 Direct P-DQN 的严重数值不稳定；尚不适合直接启动下一轮全量并把结果当作正式基准。

此前“场景和 HER 语义未发现新问题”的表述过强。本次检查只支持：运行管线可执行、已检查的 CATCH 样本符合动作条件；不支持整个 GoalEnv / HER 接口已正确的结论。

证据分为三类：

- **已确认**：磁盘实验记录、代码路径或最小复现直接证明的事实。
- **设计风险**：实现行为明确，但是否适合目标基准需要明确任务定义或进行消融。
- **待验证**：可能影响结果，尚未通过针对性实验确定因果或影响范围。

本文记录的是整理时的工作区状态，其中包含未提交修改；不是某个已发布 Git 版本的报告。本次整理没有修改环境或训练逻辑。

## 2. 实验批次与统计口径

| 项目 | 旧版边界修正后全量 | 当前 v2 短跑 |
|---|---|---|
| 输出目录 | `outputs/full_reproduction_boundary_fixed` | `outputs/paper_fidelity_v2_30min` |
| 组合 | Direct / Relay × 4 算法 × seeds 0、1、2，共 24 组 | 同左，共 24 组 |
| 完成情况 | 24/24；Direct 每组 2,000 回合，Relay 每组 30,000 回合 | 24/24；每组 6,000 环境步 |
| 更新方式 | `update_every=1, gradient_steps=1`；日志确认扣除预热后约每环境步一次更新 | Direct `U=40`；Relay 动态 U，短跑实际每组 4,000 次更新 |
| Direct 更新总量 | 每组约 7.97万–9.71万 | 每组 20万 |
| 评估 | 每 seed 100 个确定性评估回合 | 每 seed 100 个确定性评估回合 |
| 主表 checkpoint | `last.pt` | `last.pt` |
| 边界 | `clip` | `clip` |
| 观测 | 旧版非 HER 基线也获得目标坐标，Relay 还获得显式 phase | 非 HER 无额外目标坐标；四种基线均去除显式 phase；HER 保留目标坐标 |

“全量”指旧配置的训练回合预算完成，不代表全部论文设置已精确复现。当前短跑为预计 30 分钟的固定步数预算，实际约 16 分 28 秒自然完成，使用 8 个 CPU 进程、每进程 1 个 PyTorch 线程。

以下成功率为三个 seed 的均值；带 ± 的数字为 seed 间总体标准差，不是置信区间。最终 Q loss / actor loss 为各 run 最后一条训练日志的值再跨 seed 汇总；日志内 loss 是回合更新均值，所谓峰值也只是这些回合均值的峰值，不是逐优化步峰值。

旧全量数据源：[聚合 CSV](her_mpdqn_reproduction/outputs/full_reproduction_boundary_fixed/aggregate_summary.csv)、[逐 run 记录](her_mpdqn_reproduction/outputs/full_reproduction_boundary_fixed/per_run.jsonl)、各 run 的 `config.yaml`、`metrics.jsonl`、`evaluation/summary.json` 与 `evaluation_best/summary.json`。

## 3. 旧全量结果

### 3.1 最终策略表现

| 环境 | 算法 | 最终成功率 | 平均回报 | 平均回合长度 | 中继指标 | 碰边回合占比 |
|---|---|---:|---:|---:|---:|---:|
| Direct | P-DQN | 99.67% ± 0.47% | -35.97 | 36.97 | — | 8.33% |
| Direct | MP-DQN | 99.00% ± 1.41% | -35.44 | 36.43 | — | 8.33% |
| Direct | HER-PDQN | 89.33% ± 6.02% | -43.83 | 44.72 | — | 9.33% |
| Direct | HER-MPDQN | 98.00% ± 1.41% | -36.64 | 37.62 | — | 4.00% |
| Relay | P-DQN | 0.00% | -100.00 | 100.00 | 0.00% | 10.00% |
| Relay | MP-DQN | 80.33% ± 9.53% | -71.00 | 71.81 | 94.33% | 20.67% |
| Relay | HER-PDQN | 0.67% ± 0.94% | -99.86 | 99.87 | 2.33% | 56.33% |
| Relay | HER-MPDQN | 0.00% | -100.00 | 100.00 | 0.00% | 47.00% |

中继指标沿用评估日志中的 `relay_reached_rate`。已核对评估实现按 `current_phase > 0` 累积，因此在本次要求 CATCH 的配置下，它表示至少完成一次合法拾取的回合占比。环境原始 info 中同名的 `relay_reached` 也可能在仅接触中继时为真，不能混用这两个口径；后续应分别命名接触与拾取指标。

所有组的越界终止率为 0。这是边界裁剪机制的结果，不能推导为策略没有碰边。

### 3.2 Direct 的含义

Direct 在旧观测接口下可学习，三个普通/多通道方法的最终成功率接近饱和。HER-PDQN 更低且跨 seed 波动较大，但本轮没有呈现预期的 `HER-MPDQN > HER-PDQN > MP-DQN > P-DQN` 排序。

不能依据这些饱和后的点估计断言 P-DQN 更优：评估量为每 seed 100 回合，观测接口与预期论文基线不同，且最终成功率无法代替样本效率曲线。旧 Direct P-DQN 的最终 Q loss 约 0.066，不能与当前短跑 P-DQN 的发散混写成同一次实验现象。

### 3.3 Relay 的主要异常：训练中成功不等于最终策略成功

| 算法 | 全程探索训练成功率均值 | 最终确定性成功率 | 最后一条 Q loss 均值 | 最后一条 actor loss 均值 |
|---|---:|---:|---:|---:|
| P-DQN | 5.32% | 0.00% | 0.0133 | 299.24 |
| MP-DQN | 76.27% | 80.33% | 0.400 | 91.67 |
| HER-PDQN | 15.16% | 0.67% | 1.444 | 20.11 |
| HER-MPDQN | 27.95% | 0.00% | 2.217e7 | -1.038e5 |

MP-DQN 能稳定完成相当比例的 Relay 任务，证明这套旧配置并非完全无法求解。其余方法的失败不能只归结为“任务太难”或“训练不足”。

旧版 `best.pt` 根据探索训练的滚动成功率选择，不是按独立验证集选择。对它进行事后确定性评估，得到：

| Relay 算法 | `last.pt` 成功率，seed 0 / 1 / 2 | 旧 `best.pt` 成功率，seed 0 / 1 / 2 |
|---|---|---|
| P-DQN | 0% / 0% / 0% | 7% / 85% / 0% |
| MP-DQN | 78% / 70% / 93% | 80% / 90% / 90% |
| HER-PDQN | 2% / 0% / 0% | 21% / 12% / 24% |
| HER-MPDQN | 0% / 0% / 0% | 4% / 2% / 3% |

P-DQN seed 1 的中间 checkpoint 曾达到 85%，最终下降到 0%，因此至少该 seed 存在显著后期退化，不能表述为“一直没有学会”。这些 `best.pt` 也不能宣称是全训练过程的确定性评估最优模型。

Relay HER-MPDQN seed 0 的最终 Q loss 只有约 0.54，但最终成功率仍为 0%；另外两个 seed 的 Q loss 则严重爆炸。因此数值发散不能解释所有 HER-MPDQN 失败，必须同时检查动作、目标与阶段语义。

### 3.4 数值退化证据

| 旧 Relay run | 回合 Q loss 均值峰值 | 最后一条 Q loss |
|---|---:|---:|
| P-DQN seed 0 | 3.385e5 | 0 |
| P-DQN seed 1 | 9.704e4 | 0.0399 |
| P-DQN seed 2 | 5.229e7 | 4.202e-7 |
| HER-PDQN seed 0 | 4.286e7 | 2.423 |
| HER-PDQN seed 1 | 9.061e4 | 1.604 |
| HER-PDQN seed 2 | 2.874e7 | 0.304 |
| HER-MPDQN seed 0 | 1.250 | 0.540 |
| HER-MPDQN seed 1 | 4.146e9 | 6.469e7 |
| HER-MPDQN seed 2 | 1.054e9 | 1.805e6 |

Q loss 最终很小并不保证策略有效。P-DQN 可在全失败轨迹上形成低 TD 误差；仅观察最后一条 loss 会漏掉前期爆炸和后期策略退化。actor loss 的正负和绝对大小也不能单独用作性能指标。

### 3.5 旧全量可以支持和不能支持的结论

- 可以支持：边界修正后的环境能被部分策略求解；Relay 比 Direct 更难；训练成功率与确定性评估之间可能有巨大差距；后期退化和数值爆炸实际存在。
- 不能支持：论文算法排序已复现；HER-MPDQN 原理无效；只增加训练量即可解决问题；普通基线与 HER 方法的差异完全来自算法。
- 已知混杂因素：旧非 HER 基线获得额外目标/阶段信息；Direct 更新频率未采用当前的 U=40；旧 Relay 的虚拟成功未要求 CATCH；当前仍存在的 HER 距离特征问题也需纳入解释。

上述因素存在，不等于已证明其中某一个就是退化的唯一原因。因果判断需要修正后的受控对照实验。

## 4. 当前 v2 短跑提供的新证据

| 环境 | 算法 | 最终成功率 | 最后一条 Q loss 均值 | 最后一条 actor loss 均值 | 每 seed HER 总数均值 |
|---|---|---:|---:|---:|---:|
| Direct | P-DQN | 2.67% ± 3.09% | 5.557e11 | -2.387e7 | 0 |
| Direct | MP-DQN | 5.67% ± 4.92% | 1.901 | 133.61 | 0 |
| Direct | HER-PDQN | 3.00% ± 2.16% | 0.984 | 12.81 | 23,608 |
| Direct | HER-MPDQN | 40.67% ± 10.87% | 0.502 | 9.69 | 23,580 |
| Relay | P-DQN | 0.00% | 0.128 | 86.82 | 0 |
| Relay | MP-DQN | 0.00% | 0.0473 | 83.73 | 0 |
| Relay | HER-PDQN | 0.00% | 0.201 | 23.62 | 23,640 |
| Relay | HER-MPDQN | 0.67% ± 0.94% | 0.127 | 21.99 | 23,640 |

数据源：[v2 短跑聚合 CSV](her_mpdqn_reproduction/outputs/paper_fidelity_v2_30min/aggregate_summary.csv)。

- Direct HER-MPDQN 三个 seed 的成功率为 56%、34%、32%，已出现学习信号，但受下文接口问题影响，仍不是最终基准成绩。
- Direct P-DQN seed 0 / 1 最后 Q loss 分别约 6.72e5 / 1.67e12，确认数值发散；高学习率与 U=40 是候选因素，尚未证明因果。
- Relay HER-MPDQN 仅 seed 1 完成 2/100 个评估回合；当前训练量不足以评判论文级 Relay 性能。
- 全部 HER 共生成 283,404 条重标记样本。前次只读 replay 审计记录：Relay phase-0 的 HER 成功样本共 36,743 条，非 CATCH 成功违规数为 0。该计数证明动作条件检查有效，不证明整条反事实转移的语义正确。
- Relay HER 训练轨迹没有进入 phase 1，跨阶段过滤计数为 0。该批没有实际覆盖跨阶段候选，不能以此证明跨阶段处理完整。
- 24 组均保存 `last.pt` 和 `best.pt`；抽查 `last.pt` 包含版本 2 的 replay、优化器和 RNG 状态。未达到周期验证间隔，没有 `best_eval.pt`，相关汇总字段为 null。
- 日志无 stderr 错误，已扫描数值无 NaN/Inf；有限数值仍可能严重发散。

v1 与 v2 同时改变了观测和更新频率，且训练预算不同，所以不计算二者的“性能提升百分比”，也不把旧 Direct 99% 与新 Direct 40.67% 当作退步幅度。

## 5. 当前问题清单

### P0-1：HER 目标与派生距离特征不一致——已确认

位置：[replay/goal_relabeling.py](her_mpdqn_reproduction/replay/goal_relabeling.py)、[envs/direct_navigation.py](her_mpdqn_reproduction/envs/direct_navigation.py)、[envs/multi_relay_navigation.py](her_mpdqn_reproduction/envs/multi_relay_navigation.py)。

重标记当前只修改前后观测的 `desired_goal`，没有重算 `observation` 内的目标距离。Direct 的第五个坐标明确定义为到任务目标的距离；若新目标替代了任务目标，该距离仍指向原目标，输入不再符合环境观测定义。

利用现有测试构造器复查得到：保存的距离为 113.137，重标记后真实目标距离为 20.0。

影响：虚拟经验同时包含新目标坐标与旧目标距离，可能干扰价值学习。若有意保留原任务距离作为辅助上下文，需要明确命名和任务语义，不能继续把它解释为重标记目标的距离。Relay 还涉及 final/relay 两种距离，不能简单把所有距离都替换为当前目标距离。

验收：环境提供统一的目标重建/观测重算接口；Direct、Relay 各阶段和 multi-relay 的前后观测都与所声明的虚拟任务一致；增加数值一致性测试。

### P0-2：Relay 真实奖励与 `compute_reward` 重算不一致——已确认

位置：[envs/relay_navigation.py](her_mpdqn_reproduction/envs/relay_navigation.py)、[envs/multi_relay_navigation.py](her_mpdqn_reproduction/envs/multi_relay_navigation.py)。

非 HER 调用目前落入纯空间距离奖励。Relay 在 phase 0 的 `desired_goal` 是中继位置，而真实任务只有完成配送才获得 0。

最小复查：UAV 位于中继、尚未拾取，执行 MOVE(0)；`step()` 返回 -1，随后使用其返回的 achieved goal、desired goal 和 info 调用 `compute_reward()` 却返回 0。

当前 trainer 原始转移保存 `step()` 的奖励，因此并非所有真实训练样本都被该接口错误污染。但该行为破坏 GoalEnv 奖励重算约定，并影响后续 HER 或世界模型数据处理。仅凭两个二维坐标不能决定“是否已配送”，必须使用明确的任务阶段/完成上下文。

另一个相关缺口：HER 特殊处理只支持单个 dict info，传入 list-of-info 会走空间奖励分支，无法维持逐样本 CATCH 语义。

验收：真实 reward 与重算结果在所有阶段、成功/失败动作和批量输入上保持一致；真实任务奖励与辅助阶段奖励如需并存，应使用可辨识接口。

### P0-3：Relay HER 的阶段结束与整任务结束混用——行为已确认，设计待统一

位置：[replay/goal_relabeling.py](her_mpdqn_reproduction/replay/goal_relabeling.py)。

真实中继拾取后 phase 0 → 1，奖励仍为 -1，episode 继续。HER 当前对虚拟拾取成功赋 0，并统一设 `terminated=True`；它还把前后 desired goal 都设为同一虚拟目标，却保留原始 next observation 的阶段字段。

这可以被设计成独立阶段辅助任务，但目前共享 replay / Q 更新需要明确其与整任务价值的关系。仅过滤候选所属阶段和检查 CATCH，不能保证替换中继后所有转移仍是同一个可实现任务的轨迹。

影响尚未定量证明：可能截断配送价值传播，或把拾取/配送混成不同 Bellman 目标。不能直接断言这是所有 Relay 失败的唯一原因。

验收：先定义完整任务 HER 或显式阶段辅助任务；用含成功 CATCH、阶段切换、最终配送的脚本轨迹覆盖 goal、phase、reward、terminated、next observation 的一致性。特别覆盖“源转移属于 phase 0、next state 已进入 phase 1”的边界情况。

### P1-1：可观测性与算法比较混杂——已确认的设计风险

位置：[envs/wrappers.py](her_mpdqn_reproduction/envs/wrappers.py)、[scripts/train.py](her_mpdqn_reproduction/scripts/train.py)。

当前非 HER Direct 输入为 `[x,y,v,heading,distance,step]`；heading 在动力学里是绝对航向。同一位置和同一目标距离可以对应多个目标方向，单帧输入不能唯一确定目标位置。当前四种 Relay 基线还都去掉显式 phase。

HER 额外获得目标坐标，所以方法之间既有 replay/网络差异，也有观测信息差异。历史“加入 phase 就使整个 Relay 观测 Markov”的文档描述也过强：phase 0 下 final goal 只有距离，未必能从单帧恢复完整任务状态。

验收：分别声明论文重建输入与等信息条件的算法消融输入；明确绝对航向/相对方位及 phase 可见性。后续 CT-WM / RSSM 对比使用同一观测协议，避免将信息优势误归因于记忆或世界模型。

### P1-2：超时 bootstrap 与有限时域任务定义——待确认

位置：[replay/replay_buffer.py](her_mpdqn_reproduction/replay/replay_buffer.py)、[scripts/train.py](her_mpdqn_reproduction/scripts/train.py)。

达到 100 步时环境设 `truncated=True`；replay 默认 `bootstrap_on_truncation=True`，trainer 使用该默认值。与此同时 step count 是网络输入。

如果 100 步仅是采样截断，bootstrap 可以合理；如果任务定义是“必须在 100 步内完成”，则到期后继续估计价值可能不符合有限时域目标。当前样本也不提供超过时限的真实后继数据。

验收：确定时限属于任务终点还是外部采样边界，再据此统一 Bellman done mask；单独区分人为短跑预算截断。该项是否参与 Q 发散需要对照验证。

### P1-3：边界裁剪消除了早撞获利，但仍改变运动行为——已确认的建模取舍

位置：[envs/direct_navigation.py](her_mpdqn_reproduction/envs/direct_navigation.py)。

旧的 -1 越界终止会让短命失败比持续 -1 的超时获得更高回报。现在位置被裁剪且继续计步，避免该提前结束收益；速度和航向仍保留，因而可以出现贴边滑行或持续顶墙。

当前短跑各组碰边回合占比约 15%–60.33%，不是每步碰边概率。该结果表明有必要观察边界行为，不证明存在新的奖励漏洞。

验收：增加碰边步数、连续顶墙长度与靠边成功轨迹诊断；明确裁剪是基准假设。是否采用其它边界模型应作单独版本/消融，不加入未声明的稠密奖励。

### P1-4：Relay 布局和长时域扩展的可达性——风险明确，比例未测

地图边长 2,000，速度上限 40，单回合 100 步。即使忽略加速和转向损失，路径长度也不超过 4,000；从零速出发实际更短。随机两段 Relay 路线可能超过这个预算，即便目标区域半径允许提前接触也不能保证所有布局可达。

当前采样设置最小点间距离，没有按动力学和时限筛选可达布局。multi-relay 的时限可能随阶段数缩放，不能直接套用单中继的 100 步结论，应按实际配置逐个检查。

验收：使用几何下界先标出必不可达布局，再用脚本控制器给出可求解比例估计；控制器失败不等同于理论不可达。保留原始布局分布结果，并另报可达子集；不要静默筛选任务来抬高成功率。

### P1-5：基线数值不稳定——已确认异常，根因未定位

当前 Direct P-DQN 2/3 seed 爆炸；旧 Relay 多个方法也出现峰值 loss 爆炸和策略退化。梯度范数裁剪不能保证 Q 值或 TD 目标有界。

高学习率、更新/数据比、联合参数干扰、目标网络、超时处理和不一致 HER 样本都是待检查因素。当前证据不支持直接指定某一参数为唯一原因。

验收：先修正确定的数据语义问题，再对失败 seed 做控制变量诊断；同时观察 Q/target Q 的分布、梯度、动作饱和比例和固定评估成功率。保留论文设置运行，稳定化实验明确标记为消融。

### P2：指标覆盖与文档准确性

- 将接触中继率、CATCH 成功率、配送率分别命名；当前评估的 `relay_reached_rate` 已按阶段切换计数，但环境同名 info 的语义更宽。
- 短跑应实际覆盖阶段切换样本与验证 checkpoint；当前“0 条跨阶段过滤”只是没有候选。
- `best.pt` 的训练选择标准和 `best_eval.pt` 的验证选择标准要明确，报告最终/验证选中结果，不能事后挑高分替代。
- 观测 Markov 性、批量 reward 支持、阶段虚拟任务等文档声明应随接口修正更新。
- 物理参数、成功半径、初始化间距、Relay U 函数等仍是重建假设；见 [RECONSTRUCTION_SPEC.md](her_mpdqn_reproduction/RECONSTRUCTION_SPEC.md)。

## 6. 建议处理顺序与重新全量的条件

1. 定义真实任务、阶段辅助任务和目标重标记的语义，修正 P0-1 / P0-2 / P0-3。
2. 补充针对性回归：目标距离、reward 重算、批量 info、CATCH 阶段边界、终止和截断；保留原有动力学与 MP-DQN 掩码测试。
3. 固定观测协议、时限定义与边界假设，检查 Relay / multi-relay 可达性。
4. 重新做覆盖 24 组的短跑，加入脚本生成的阶段切换审计轨迹；脚本轨迹是否用于训练需另外明确，默认仅用于验证。
5. 对 Direct P-DQN 和旧 Relay 退化问题完成小规模数值诊断，确定正式论文配置与稳定化消融配置。
6. 正式全量时保存配置和代码版本、固定评估 seeds、验证选中 checkpoint、逐步数成功率及各类诊断指标。

场景/奖励/观测语义变化后应启动新的输出目录和训练运行，并检查 checkpoint 兼容版本。旧 replay 中的错误标签或旧特征不能未经迁移验证直接续训成新版本结果。

## 7. 文件索引与复查方法

- 旧全量：[aggregate_summary.csv](her_mpdqn_reproduction/outputs/full_reproduction_boundary_fixed/aggregate_summary.csv)、[per_run.jsonl](her_mpdqn_reproduction/outputs/full_reproduction_boundary_fixed/per_run.jsonl)。
- 当前短跑：[aggregate_summary.csv](her_mpdqn_reproduction/outputs/paper_fidelity_v2_30min/aggregate_summary.csv)、[per_run.jsonl](her_mpdqn_reproduction/outputs/paper_fidelity_v2_30min/per_run.jsonl)。
- 早期小规模历史结果：[RESULTS.md](her_mpdqn_reproduction/RESULTS.md)。其中包含更早的越界终止版本，不能混入本文全量主表。
- 任务假设：[RECONSTRUCTION_SPEC.md](her_mpdqn_reproduction/RECONSTRUCTION_SPEC.md)。

本文两项接口问题可在项目目录下用以下只读示例复查；它使用现有测试构造器，不改动训练数据：

```python
import numpy as np
from tests.test_her import make_episode
from tests.test_goal_switching import make_env
from replay import FutureGoalRelabeler
from envs import DirectNavigationEnv
from envs.dynamics import MOVE

item = FutureGoalRelabeler(her_k=4, seed=2).relabel_episode(
    make_episode(), DirectNavigationEnv(goal_radius=0.1).compute_reward)[0]
obs = item.transition.observation
print(obs['observation'][4],
      np.linalg.norm(obs['achieved_goal'] - obs['desired_goal']))
# 整理时：113.137... 与 20.0

env = make_env()
obs, reward, _, _, info = env.step((MOVE, np.array([0.0], np.float32)))
print(reward, env.compute_reward(
    obs['achieved_goal'], obs['desired_goal'], info))
# 整理时：-1.0 与 0.0
```

输出目录受 Git 忽略规则影响。分享本文时，应同时提供相应聚合记录；仅推送 Markdown 不会自动上传原始实验数据。
