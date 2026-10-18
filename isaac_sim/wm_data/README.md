# 世界模型数据采集（第一步：离线验证 RSSM 的记忆）

目的：在 Isaac 中采集无人机飞行数据，只训练 DreamerV3 的世界模型，然后检验隐状态 h_t 能否还原
"见过但当前看不到"的障碍。

## 文件
- `scenegen.py`：随机场景（柱子、墙、U/C 形陷阱）、A* 规划、脚本化飞行策略、占据图与可见掩码、论文奖励。纯 numpy，不依赖 Isaac
- `dry_run.py`：用简化动力学试飞，调策略参数用
- `collect.py`：Isaac Sim 4.1 + OmniDrones 多机并行采集（每架无人机一个 24×24 m 场地，场地间隔 40 m）
- `viz_episode.py`：把一条 episode 渲染成视频，用于人工检查标签

## 每步数据（20 Hz，每条 episode 一个 npz）
| 键 | 形状 | 含义 |
|---|---|---|
| depth | (T,48,80) f16 | 渲染深度，截断到 [0.6, 6] m，超量程填 6 m |
| state | (T,15) | o^s = 目标方向(机体系,3) + d_xy + d_z + 机体速度(3) + 角速度(3) + 四元数 wxyz(4) |
| action | (T,5) | a_t = [v_body/Λ (3), cos ψ_cmd, sin ψ_cmd]，Λ = diag(1.5,1.5,0.5)，在看到 obs_t 后选择（DreamerV3 约定），最后一步为 0 |
| reward | (T,) | 10·r_prog + 5·r_safety + 2·r_height + r_goal(+100) + r_collision(−5)，与 sandbox 一致 |
| is_first / is_last / is_terminal | (T,) | episode 标志；超时和越界不算 terminal |
| occ / seen / vis | (T,64,64) bool | 以机体为中心的俯视图，0.15 m/格，第 0 行在正前方 4.8 m：障碍占据 / 至今见过 / 当前可见 |
| pos, yaw | (T,3), (T,) | 场地坐标系下的位置和偏航 |
另外存了 boxes (B,5: cx,cy,hx,hy,yaw)、start、goal、mode、outcome。

**记忆格** = `occ & seen & ~vis`：见过、但当前不在视野里的障碍格，是记忆探针的主要评测对象。

## 飞行策略（mode）
- goal（70%）：A* 规划到终点，纯追踪 + OU 动作噪声（靠近障碍时噪声和速度降低）
- explore（15%）：先访问 2–4 个随机点，偏航带随机偏移
- trap（15%）：先飞进 U/C 形陷阱腔体，原地转身观察 1.5 s，再飞向终点

障碍只用于渲染，碰撞按解析距离判断（< 0.3 m 即碰撞并终止），与沙盒一致。

## 校验
`python collect.py --num_envs 4 --episodes 8 --out $Y/data/wm_test --validate` 会把渲染深度与按场景几何解析计算的深度对比。
2026-10-18 的结果：中位绝对误差 0 m，95% 分位 0.03 m，说明障碍摆放、相机内参和读图时序都正确。

## 世界模型训练与记忆探针
- `train_wm.py train`（dreamer 环境）：用官方 DreamerV3 离线训练，随机截取 64 步窗口，每个窗口开头重置 RSSM 状态。深度量化为 uint8 图像，state 作为向量一起重建
- `train_wm.py extract`：把整条 episode 依次送入编码器和 RSSM（跨块保留状态），导出每步的 h_t（deter）和 z_t（stoch），同时做开环测试：前半段用观测、后半段只用动作想象，解码出深度。`--random_init` 导出未训练模型的特征作为对照
- `probe.py`（isaac41 环境）：用同一个 MLP 从特征还原 64×64 占据图，在验证集上按"当前可见 / 记忆 / 从未见过"三组分别统计 IoU 和召回率，并画出记忆随"离开视野的步数"衰减的曲线；对照组包括未训练模型和只看当前帧的 CNN

## 结果
- `pipeline.sh`：采集满 1000 条后停止，然后依次训练、导出特征（训练后模型和未训练对照）、跑探针
- `results/`：第一步的结果报告（`step1_results.md`）、方向一方案（`direction1_plan.md`）、图表和原始指标 json
