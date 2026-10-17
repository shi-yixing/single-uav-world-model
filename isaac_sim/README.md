# AirDreamer 演示环境（nice / RTX 4090）

- Isaac Sim 4.1.0（pip 版）+ Isaac Lab v1.1.0 + OmniDrones（main），conda 环境：/media/gsk/disk4t/yixing/envs/isaac41
- 驱动 530.30.02 可用（Iray 离线渲染器不可用，不影响 RTX 实时渲染和深度相机）

## 运行
    source /media/gsk/disk4t/yixing/env.sh
    conda activate $Y/envs/isaac41
    cd $Y/demo && python airdreamer_demo.py --speed 1.5
输出在 $Y/outputs/demo/：airdreamer_demo.mp4、depth_48x80.npy（策略输入深度序列）、trajectory.npy、summary.json

## 已做的兼容性修补
1. boto3/botocore 固定为 1.34.68（与 Isaac Sim 自带 botocore 一致，否则 replicator 加载失败）
2. tensordict 固定为 0.3.2（torchrl 0.3.1 要求）
3. 按 OmniDrones troubleshooting：xform_prim_view.py 第 189 行去掉 usd=usd（原文件备份为 .orig）

## 拷回本地
    scp yixing:/media/gsk/disk4t/yixing/outputs/demo/airdreamer_demo.mp4 .

## 本目录文件
- `install_isaacsim.sh`：在大盘上装 Miniforge、建 conda 环境、pip 安装 Isaac Sim 4.1.0 和 torch 2.2.2（cu121）
- `install_isaaclab_omnidrones.sh`：安装 Isaac Lab v1.1.0 和 OmniDrones，安装后还需要做上面列出的三处修补
- `smoke_test.py`：headless 渲染自检，会输出一张 png
- `airdreamer_demo.py`：演示场景（柱林、大墙、U 形陷阱），挂载 48×80 深度相机（87° 视场，量程 [0.6, 6] m），录制 1920×1080 合成视频。飞行使用预先规划的样条加 Lee 位置控制器，不是学习得到的策略

视频和 .npy 数据不放进仓库，保存在服务器的 `outputs/demo/` 下。
