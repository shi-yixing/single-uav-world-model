# single-uav-world-model

[中文](#中文) | [English](#english)

## 中文

基于世界模型（World Model）的单无人机（Single UAV）自主导航研究。参考工作为 AirDreamer：以 48×80 深度图与 15 维本体状态作为观测，在 DreamerV3 世界模型的隐空间想象（latent imagination）中训练避障导航策略。

### 研究动机

DreamerV3 的 RSSM 以确定性循环状态 h_t 与随机状态 z_t 表征环境，h_t 承担对历史观测的记忆。深度相机视场有限（87°×58°，量程 0.6–6 m），障碍物离开视野后只能依赖 h_t 保留其信息；若该记忆不足，策略在 U 形、C 形等陷阱场景中容易陷入局部极小。

方向一围绕 RSSM 记忆的显式化展开：

1. **记忆评估**：冻结世界模型，训练探针从 (h_t, z_t) 解码以机体为中心的 64×64 俯视占据图，按"当前可见 / 已见但不可见（记忆格）/ 从未见过"分组评估，并统计记忆随离开视野时长的衰减。
2. **地图辅助损失**：在世界模型上增加占据图解码头，仅在已观测格子上计算交叉熵，梯度回传至 RSSM，检验能否增强对不可见障碍的记忆。

### 目前进展

| 阶段 | 内容 | 结论 |
|---|---|---|
| 演示沙盒 | 浏览器里的简化仿真，复现论文的任务设定 | 用于任务演示，不用于训练 |
| Isaac 环境 | Isaac Sim 4.1 + OmniDrones，论文同款深度相机与场景 | 已跑通并录制演示视频 |
| 方向一 · 第一步 | 采集约 1000 条飞行数据（约 52 万步），离线训练 DreamerV3 世界模型，用探针从 h_t 还原占据图 | h_t 保留一定记忆但衰减明显：记忆格 AP 0.117，只看当前帧 0.074，未训练 0.045；离开视野 1–2 s 后降至约 1/3 |
| 方向一 · 第二步（离线） | 加地图头作为辅助损失，权重 λ = 1 / 10 | λ = 10 把记忆格 AP 从 0.118 提到 0.145（+23%），深度重建和想象误差不变；λ = 1 基本无效 |

下一步（尚未开始）：多种子与更大权重的离线补充实验；在线 RL 训练，比较 U/C 形陷阱上的成功率。

### 目录结构

```
.
├── sandbox/             # three.js 网页演示沙盒，双击 index.html 即可离线运行
├── isaac_sim/           # Isaac Sim 4.1 + OmniDrones 安装脚本与演示场景
│   └── wm_data/         # 方向一：数据采集、世界模型训练、记忆探针、地图辅助损失
│       └── results/     # 第一、二步的结果报告、图表和指标
└── README.md
```

各目录的详细说明：

- [sandbox/README.md](sandbox/README.md)：沙盒的运行方式、代码结构和常见修改
- [isaac_sim/README.md](isaac_sim/README.md)：Isaac 环境安装、三处兼容性修补、演示视频录制
- [isaac_sim/wm_data/README.md](isaac_sim/wm_data/README.md)：数据格式、飞行策略、训练与探针流程
- 结果报告：[第一步](isaac_sim/wm_data/results/step1_results.md)、[第二步](isaac_sim/wm_data/results/step2/step2_offline_results.md)、[方向一方案](isaac_sim/wm_data/results/direction1_plan.md)

### 运行环境

- 一张 RTX 4090（24 GB，驱动 530 / CUDA 12.1，Ubuntu 22.04）
- 两个 conda 环境：`isaac41`（Isaac Sim 4.1 + OmniDrones，采数据、跑探针）和 `dreamer`（官方 JAX 版 DreamerV3，训练世界模型）
- JAX 默认会预占大部分显存，Isaac 采集和世界模型训练需要分开运行

### 说明

- 视频、深度序列 `.npy`、采集数据和模型权重体积较大，不放进仓库，保存在服务器上
- 演示视频中的飞行轨迹来自预先规划的样条加 Lee 位置控制器，用于展示环境和传感器，不是学习得到的策略

## English

Research on autonomous single-UAV navigation with world models. The reference work is AirDreamer: a 48×80 depth image and a 15-D proprioceptive state form the observation, and an obstacle-avoidance navigation policy is trained by latent imagination in a DreamerV3 world model.

### Motivation

The RSSM in DreamerV3 represents the environment with a deterministic recurrent state h_t and a stochastic state z_t; h_t carries the memory of past observations. The depth camera has a limited field of view (87°×58°, range 0.6–6 m), so once an obstacle leaves the view, its information survives only in h_t. If this memory is weak, the policy tends to get trapped in local minima such as U- and C-shaped traps.

Direction 1 targets making the RSSM memory explicit:

1. **Memory evaluation**: freeze the world model and train a probe that decodes a body-centred 64×64 top-down occupancy map from (h_t, z_t); evaluate separately on visible, seen-but-not-visible (memory) and never-seen cells, and measure how memory decays with time since an obstacle left the view.
2. **Map auxiliary loss**: add an occupancy decoder head to the world model, with cross-entropy computed only on observed cells and gradients flowing into the RSSM, to test whether it strengthens memory of occluded obstacles.

### Progress

| Stage | What | Finding |
|---|---|---|
| Demo sandbox | Simplified browser simulation reproducing the paper's task setup | For task illustration only, not for training |
| Isaac environment | Isaac Sim 4.1 + OmniDrones with the paper's depth camera and scenes | Running; demo video recorded |
| Direction 1 · Step 1 | ~1000 flight episodes (~520k steps), offline DreamerV3 world-model training, probe that decodes occupancy from h_t | h_t retains some memory but it decays: memory-cell AP 0.117 vs 0.074 (current frame only) vs 0.045 (untrained); drops to about 1/3 after 1–2 s out of view |
| Direction 1 · Step 2 (offline) | Map head as an auxiliary loss, weight λ = 1 / 10 | λ = 10 raises memory-cell AP from 0.118 to 0.145 (+23%) with unchanged depth reconstruction and imagination error; λ = 1 has little effect |

Next (not started yet): more seeds and larger weights offline; online RL comparing success rates in U/C traps.

### Layout

```
.
├── sandbox/             # three.js browser sandbox, open index.html offline
├── isaac_sim/           # Isaac Sim 4.1 + OmniDrones install scripts and demo scene
│   └── wm_data/         # Direction 1: data collection, world-model training, memory probe, map auxiliary loss
│       └── results/     # Step 1 and Step 2 reports, figures and metrics
└── README.md
```

Sub-directory docs (in Chinese):

- [sandbox/README.md](sandbox/README.md): how to run the sandbox, code structure, common edits
- [isaac_sim/README.md](isaac_sim/README.md): Isaac setup, three compatibility patches, recording the demo video
- [isaac_sim/wm_data/README.md](isaac_sim/wm_data/README.md): data format, flight policies, training and probe pipeline
- Reports: [Step 1](isaac_sim/wm_data/results/step1_results.md), [Step 2](isaac_sim/wm_data/results/step2/step2_offline_results.md), [Direction 1 plan](isaac_sim/wm_data/results/direction1_plan.md)

### Environment

- One RTX 4090 (24 GB, driver 530 / CUDA 12.1, Ubuntu 22.04)
- Two conda environments: `isaac41` (Isaac Sim 4.1 + OmniDrones, data collection and probes) and `dreamer` (official JAX DreamerV3, world-model training)
- JAX preallocates most GPU memory by default, so Isaac collection and world-model training run separately

### Notes

- Videos, depth `.npy` sequences, collected data and model weights are large and kept on the server, not in this repo
- The flight in the demo video follows a pre-planned spline with a Lee position controller to show the environment and sensors; it is not a learned policy
