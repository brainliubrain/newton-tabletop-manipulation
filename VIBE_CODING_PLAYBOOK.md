# 具身智能物理仿真 (Embodied Physics Sim) 面试实战: Vibe Coding 现场改造与即时答辩指南
> **岗位**: RSI for Robotics Simulation 算法工程师  
> **面试形式**: 边演示边讲解 + 现场代码探讨与即时改造 (Live Vibe Coding)  
> **核心环境**: Newton 1.7.0 + SolverMuJoCo (CUDA) + Python 3.12 (原生 Windows RTX 3060)

---

## 快速导航与应急启动
在屏幕共享演示时，所有操作均已制作为桌面一键脚本，双击即可无延迟执行：
- `01_Run_Task_Evaluation_10Ep.bat`: 运行 10 轮主任务评测 (耗时 ~25s/轮)
- `02_Run_Solver_SysID_RSI_Fast.bat`: **【王牌演示】** 求解器接触刚度 Real2Sim RSI 闭环 (耗时仅 1.5s，秒级收敛)
- `03_Run_Gripper_Adaptive_RSI.bat`: 任务层抓持预紧自适应 RSI 闭环 (含专职教师门禁认证)
- `04_Open_Interview_Showcase.bat`: 打开交互式全屏汇报答辩大屏 (`showcase.html`)
- `05_Run_Unit_Tests.bat`: 运行任务判定与几何契约独立单元测试 (`test_success.py`)
- `06_Launch_3D_Interactive_Viewer.bat`: 启动原生 60FPS 3D 轨迹回放窗口 (支持鼠标 360° 旋转缩放)

---

## 考官 5 大高频现场改造需求 (Live Vibe Coding) 应对预案

### 改造题 1: 反转抓取放置顺序 (先杯子，后碗)
- **考官提问**: *"当前代码是先抓碗后抓杯子，请现场改写规划，变成先抓杯子放进托盘，再抓碗放进托盘，并验证仿真稳定性。"*
- **修改文件**: `C:\Users\HHH\Desktop\newton_pick_place\agent.py`
- **定位行号**: 搜索 `self.plan = [` (约第 75~135 行)
- **修改方法**:
  在 `PickPlaceAgent.build_plan()` 中，直接将碗与杯子的 Waypoint 块顺序调换：
  ```python
  # 修改前:
  self.plan = bowl_approach + bowl_grasp + bowl_lift + bowl_transit + bowl_release + \
              cup_approach + cup_grasp + cup_lift + cup_transit + cup_release

  # 修改后:
  self.plan = cup_approach + cup_grasp + cup_lift + cup_transit + cup_release + \
              bowl_approach + bowl_grasp + bowl_lift + bowl_transit + bowl_release
  ```
- **技术亮点与考官对答**:
  - *"我们在几何规划时严格解耦了双槽坐标：杯子托盘槽位为 `(0.68, 0.02, 0.41)`，碗槽位为 `(0.56, 0.02, 0.41)`。"*
  - *"当杯子先放入托盘后，其杯口上沿高度为 $z \approx 0.476\,\text{m}$；后续运送碗时的巡航高度设定为 `CARRY_Z = 0.600\,\text{m}$，净空垂直余量达 $+12.4\,\text{cm}$，完全避免空中碰擦！"*
- **验证命令**:
  ```powershell
  python run_task.py --episodes 1
  ```

---

### 改造题 2: 修改环境几何尺寸或托盘位置 (空间平移鲁棒性)
- **考官提问**: *"如果我将托盘沿 X 轴向右外移 5cm (`tray_center_xy = (0.67, 0.02)`)，你的代码需要改哪些地方？"*
- **修改文件**:
  1. `env.py`: 第 180 行 `tray_center_xy: tuple = (0.67, 0.02)`
  2. `task.py`: 第 45 行 `tray_center_xy: tuple = (0.67, 0.02)`
  3. `agent.py`: 槽位自适应跟随 `cfg.tray_center_xy`
- **技术亮点与考官对答**:
  - *"我们的架构设计遵循了严格的**环境-判据解耦契约**：`task.py` 是纯数学/几何判据，零仿真器依赖，可直接作为强化学习 Reward 或无缝迁移至真机。"*
  - *"将托盘外移至 $x=0.67\,\text{m}$ 时，机械臂接近其工作空间极限（Franka 最大展长 $0.855\,\text{m}$），肩部关节由于重力力矩 $\tau_g = mg\ell$ 增大，需要确保位置伺服刚度 `ARM_KE >= 2500`，否则会引发悬臂重力下垂（Cantilever Droop）。我们的大臂刚度储备充裕，可直接稳定达到。"*

---

### 改造题 3: 严格化成功判据 (防止放宽条件的 Reward Hacking)
- **考官提问**: *"如果考官觉得判定太宽松，要求将静止判定时间加倍 (`settle_frames: 15 -> 30`)，并将最大倾角从 35° 收紧到 15°，请现场修改并验证。"*
- **修改文件**: `C:\Users\HHH\Desktop\newton_pick_place\task.py`
- **定位行号**: 第 40~60 行 `TaskConfig`
- **修改内容**:
  ```python
  # 修改前:
  settle_frames: int = 15       # 0.25 秒
  max_tilt_deg: float = 35.0    # 倾斜容差

  # 修改后:
  settle_frames: int = 30       # 0.50 秒长时间静止判定
  max_tilt_deg: float = 15.0    # 严苛姿态容差
  ```
- **技术亮点与考官对答**:
  - *"在我们的物理落地实现中，物体放入托盘后速度迅速耗散至 $< 0.002\,\text{m/s}$，碗底与托盘底面完美贴合（倾角 $0.00^\circ$）。即便延长至 60 帧（1整秒）也完全稳态满足。"*
  - *"可立即运行 `test_success.py`，证明判定逻辑在极端 corner case 下的边界严密性。"*
- **验证命令**:
  ```powershell
  python test_success.py
  ```

---

### 改造题 4: 考察 Solver Infra——现场演示接触刚度对穿模的影响与 RSI 自动修复
- **考官提问**: *"你说 Newton / MuJoCo 默认接触参数偏软，能现场复现一下软接触导致的穿模吗？再演示你是怎么用 RSI 自动修好的。"*
- **操作步骤**:
  1. 双击桌面 `02_Run_Solver_SysID_RSI_Fast.bat`。
  2. 结合控制台输出向考官讲解：
     - **Round 0 (初始库默认态)**: $k_e = 2.5\times 10^3\,\text{N/m}, k_d = 100$。陶瓷碗受重力下落触底，穿透量达 **$8.59\,\text{mm}$**（碗体陷入桌面近 $1\,\text{cm}$，严重违背物理真实）。
     - **RSI 自适应反馈更新**: 外层 Agent 依据穿透遥测误差，基于物理阻尼比配准公式：
       $$\zeta = \frac{k_d}{2\sqrt{d_r k_e}} = 1.58 \quad (\text{最优临界/略微过阻尼})$$
       自动在第 1 轮将刚度自适应推进至 $k_e = 2.0\times 10^6\,\text{N/m}, k_d = 4.24\times 10^3$。
     - **Round 1 (自适应收敛态)**: 稳态穿透瞬间缩减至 **$0.22\,\text{mm}$**（亚毫米工程级精度，穿透减少 **$97.4\%$**，且动能完全耗散无虚假高频微震）。
     - **专职教师门禁 (TeacherSolverGuard)**: 实时审计流形无 NaN/Inf、阻尼比在 $[0.2, 5.0]$ 安全域内，判定 `CONVERGED (PERFECT)`，全流程耗时仅 1.5 秒！

---

### 改造题 5: 负载扰动与电机刚度补偿 (Armature & Damping)
- **考官提问**: *"如果物体质量从 0.35kg 增大到 1.0kg（重载抓取），机械臂发生末端下垂，在控制器与求解器层面该如何处理？"*
- **修改文件**: `C:\Users\HHH\Desktop\newton_pick_place\env.py`
- **定位行号**: 第 35~45 行
- **修改方案**:
  ```python
  # 增强前三关节 (大臂承重关节) 的刚度与阻尼，并注入对角电枢惯量
  ARM_KE = 3500.0   # 从 2500 提升至 3500 N*m/rad
  ARM_KD = 450.0    # 从 300 提升至 450 N*m*s/rad (匹配临界阻尼)
  ARM_ARMATURE = 0.15 # 注入电枢惯量抑制数值奇异性
  ```
- **第一性原理解释**:
  - *"前伸位置伺服的稳态下垂满足 $\Delta q_{\text{droop}} \approx \frac{\tau_g}{k_e}$。通过刚度提升可以将重载下垂从 $12\,\text{mm}$ 压减至 $3\,\text{mm}$ 以内；同时注入对角电枢惯量矩阵 $M'(q) = M(q) + \text{diag}(I_{\text{armature}})$，可显著改善广义质量矩阵的条件数 $\kappa(M)$，避免小质量连杆在隐式解算时的高频数值爆炸。"*

---

## 答辩防翻车金律 (The Iron Rules)
1. **绝不在演示时使用未验证的一句话复杂命令行**: 一律使用写好的脚本或明确传参。
2. **遇报错第一反应先看日志与门禁**: 明确告诉考官报错属于哪个维度的契约审计拒绝，而非代码崩溃。
3. **把所有调优过程转化为“第一性原理公式 + 数据曲线”汇报**: 杜绝“我大概调了一下感觉可以”的业余回答，使用 $k_e, k_d, \zeta, F_N, \mu$ 说话。
