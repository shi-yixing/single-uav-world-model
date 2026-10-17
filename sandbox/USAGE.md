# AirDreamer 导航沙盒（本地版）

## 运行
双击 `index.html`，用 Chrome 或 Edge 打开即可，不需要安装任何东西，也不需要网络。

如果浏览器限制本地文件，可以在这个文件夹里启动一个本地服务器：
```
python -m http.server 8000
```
然后访问 http://localhost:8000

## 文件
- `index.html`：页面结构
- `style.css`：样式（顶部的 `:root` 是配色，含深色模式）
- `sim.js`：全部逻辑，按以下顺序组织
  - 论文常量：深度图 48×80、[0.6, 6] m、87°×58° 视场、20 Hz 控制 / 200 Hz 物理、Λ=diag(1.5,1.5,0.5)
  - `SCENES`：6 个场景，障碍物都是长方体 `box(x0,y0,x1,y1)`，ENU 坐标，单位 m
  - `rayHit` / `renderDepth`：逐像素射线求交生成深度图，并把中间两行融合进占据栅格
  - `CTRL`：控制器（贪心势场、建图 + A*、手动），输出机体系速度和世界系偏航指令
  - `controlStep`：一阶动力学 + 论文的奖励和终止条件
  - 后半部分：Three.js 3D 显示和右侧面板
- `lib/`：three.js r147 和 OrbitControls（MIT 许可）

## 常见修改
- 加场景：在 `SCENES` 里仿照已有条目写一个 `build()`，返回 `start`、`goal`、`boxes`
- 加控制器：在 `CTRL` 里加一项，实现 `act()`，返回 `{vb:[vx,vy,vz], yaw}`
- 改奖励：在 `controlStep` 里的 `tr` 和 `r=` 那两行

注意：这是 JavaScript 写的简化仿真，适合演示任务设定，不能用来训练策略。正式实验用 Isaac Sim + OmniDrones。
