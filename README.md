# single-uav-world-model

利用世界模型（World Model）实现单无人机（Single UAV）自主导航的研究与代码仓库。参考论文为 AirDreamer：无人机只靠一张 48×80 的深度图和自身状态，在障碍物之间飞到目标点，策略在 DreamerV3 世界模型的"想象"里训练。

## 大白话版

世界模型就是让无人机在脑子里学一个"模拟器"：给它看当前画面和要做的动作，它能预测接下来会看到什么。DreamerV3 用一个循环隐状态 h_t 当这份"记忆"。

问题在于深度相机视野有限，障碍一转到身后就看不见了。如果 h_t 记不住它，无人机就容易钻进 U 形、C 形这种"死胡同"里出不来。所以本仓库目前的工作（方向一）是：

1. **先测一测**：h_t 到底记不记得"见过但现在看不到"的障碍？
2. **再教一教**：训练时额外让世界模型画出一张俯视占据图（辅助损失），看记忆会不会变好。

## 目前进展

| 阶段 | 内容 | 结论 |
|---|---|---|
| 演示沙盒 | 浏览器里的简化仿真，复现论文的任务设定 | 用于讲解，不能训练 |
| Isaac 环境 | Isaac Sim 4.1 + OmniDrones，论文同款深度相机与场景 | 已跑通并录制演示视频 |
| 方向一 · 第一步 | 采集约 1000 条飞行数据（约 52 万步），离线训练 DreamerV3 世界模型，用探针从 h_t 还原占据图 | h_t 有记忆但不牢：记忆格 AP 0.117，只看当前帧 0.074，未训练 0.045；离开视野 1–2 s 后明显衰减 |
| 方向一 · 第二步（离线） | 加地图头作为辅助损失，权重 λ = 1 / 10 | λ = 10 把记忆格 AP 从 0.118 提到 0.145（+23%），深度重建和想象误差不变；λ = 1 基本无效 |

下一步（尚未开始）：多种子与更大权重的离线补充实验；在线 RL 训练，比较 U/C 形陷阱上的成功率。

## 目录结构

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

## 运行环境

- 一张 RTX 4090（24 GB，驱动 530 / CUDA 12.1，Ubuntu 22.04）
- 两个 conda 环境：`isaac41`（Isaac Sim 4.1 + OmniDrones，采数据、跑探针）和 `dreamer`（官方 JAX 版 DreamerV3，训练世界模型）
- JAX 默认会预占大部分显存，Isaac 采集和世界模型训练需要分开运行

## 说明

- 视频、深度序列 `.npy`、采集数据和模型权重体积较大，不放进仓库，保存在服务器上
- 演示视频中的飞行轨迹来自预先规划的样条加 Lee 位置控制器，用于展示环境和传感器，不是学习得到的策略
