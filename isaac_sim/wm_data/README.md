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

## 第二步（离线）：地图辅助损失
- `train_wm.py train --map_scale λ`：在世界模型上加一个地图头，从 (h_t, z_t) 解码 64×64 占据图，只在"见过"的格子上计算加权交叉熵，梯度回传进 RSSM（`MapAgent` 类）。地图标签只在训练时使用
- `pipeline_step2.sh`：训练 λ = 1 和 λ = 10 两个模型，导出特征，用与第一步相同的探针和基线对比
- 结果见 `results/step2/step2_offline_results.md`：λ = 10 把记忆格 AP 从 0.118 提升到 0.145，深度重建和想象误差不变；λ = 1 基本无效

## 第二步补充实验（多种子）
- `pipeline_seeds.sh`：基线和 λ=10 各补 2 个种子，外加 λ=30，然后统一用探针评测；`plot_seeds.py` 画均值 ± 标准差图
- 结果见 `results/step2_seeds/step2_seeds_results.md`：记忆格 AP 基线 0.126 ± 0.006，λ=10 为 0.149 ± 0.002，λ=30 为 0.165

## 在线强化学习（进行中）
- `rl_server.py`（isaac41 环境）：Isaac 环境服务器，N 架无人机同步步进，观测/动作/奖励与采集一致，`--labels` 时附带地图标签
- `isaac_uav_env.py`（dreamer 环境）：DreamerV3 的环境客户端，每个环境进程占一个无人机槽位
- `train_rl.py`：在线 DreamerV3 训练入口，`--map_scale` 打开地图辅助损失（`wm_agents.py`）
- `run_rl.sh <logdir> <map_scale> [参数]`：先启动服务器，再启动训练
- `bench_server.py`：用随机动作测环境速度（GPU 空闲时 16 架约 164 步/秒，含地图标签）
- `scenegen.sample_test_scene`：固定的 U/C 形陷阱测试场景，用于最终评测
- `make_ppt_videos.py`：组会用的标签视频和"想象 vs 现实"视频

## 第三轮：长时记忆场景
- `scenegen.sample_memory_scene`：深死胡同（8–10 m 走廊，起点在最深处）和嵌套死胡同（走廊外再套一个口袋）
- `rl_server.py --trap_kinds`：指定训练时"陷阱内起飞"使用的场景类型
- `pipeline_mem_eval.sh`：用已训练模型零样本测试新场景；`pipeline_rl3.sh`：按规则自动决定是否重训并评测
- 完整结论见 `results/FINAL_REPORT.md`
