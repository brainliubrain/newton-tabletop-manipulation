# Isaac Lab 3.0 × Newton 1.7.0: 任务定义与成功标准规格说明书 (Task Specification)

> **对应飞书面试要求**：`1）构建出完整的 task 任务和成功标准；`  
> **代码实现路径**：`newton_pick_place/task.py`（单测：`newton_pick_place/test_success.py`）  
> **技术栈**：IsaacLab 3.0 规范同构 / Newton 1.7.0 (MuJoCo-Warp CUDA GPU 求解器)

---

## 1. 任务定义 (Task MDP Formulation)

### 1.1 任务目标描述 (Goal Description)
在高度 $z=0.400\,\text{m}$ 的水平橡木桌面上，初始随机摆放着一个**矩形托盘 (Tray)**、一个**宽口陶瓷碗 (Bowl)** 和一个**带柄咖啡杯 (Cup)**。  
控制一台固定基座的 **Franka Emika FR3 (7-DoF + 2指平行夹爪)** 机械臂，依次将桌面上的**碗**和**杯子**抓起并平稳运送放入托盘的左右两个分区槽位内，最后完全松开夹爪并撤离。

### 1.2 状态与观测空间 (Observation Space $\mathcal{S}$)
符合 Gym / IsaacLab `DirectRLEnv` 与 `ManagerBasedRLEnv` 规范，观测字典（或 36 维连续张量）：
- **机械臂关节状态**：$\mathbf{q} \in \mathbb{R}^7, \; \dot{\mathbf{q}} \in \mathbb{R}^7$
- **末端 TCP 位姿**：$\mathbf{p}_{\text{tcp}} \in \mathbb{R}^3, \; \mathbf{q}_{\text{tcp}} \in \mathbb{R}^4$
- **夹爪状态**：两指张开间距 $g \in [0.0, 0.08]\,\text{m}$
- **物体感知状态**：
  - 陶瓷碗位姿：$\mathbf{p}_{\text{bowl}} \in \mathbb{R}^3, \; \mathbf{q}_{\text{bowl}} \in \mathbb{R}^4$
  - 咖啡杯位姿：$\mathbf{p}_{\text{cup}} \in \mathbb{R}^3, \; \mathbf{q}_{\text{cup}} \in \mathbb{R}^4$
  - 托盘目标中心：$\mathbf{p}_{\text{tray}} \in \mathbb{R}^3$

### 1.3 动作空间 (Action Space $\mathcal{A}$)
- **关节空间目标**：$\mathbf{a} \in \mathbb{R}^8$
  - $\mathbf{a}_{0:7}$：7 个臂关节的期望位置目标（由 DLS-IK 或策略网络以 60Hz 给出，经底层 600Hz 10 个子步 Newton 刚体积分求解）
  - $\mathbf{a}_7$：平行指爪期望开合量（$0.04\,\text{m}$ 全开，$\le 0.015\,\text{m}$ 闭合预紧）

---

## 2. 四维联合成功判定标准 (Success Criteria)

为了杜绝大模型/策略网络“空中掠过误判”、“夹着悬停在托盘上方算成功”、“砸进托盘弹飞瞬间算成功”等常见的规避作弊漏洞，设计了**严密的四维联合门禁（`EpisodeJudge`）**。

回合被判定为 `SUCCESS`，当且仅当**碗 (Bowl)** 与 **杯子 (Cup)** 两个物体**同时满足**以下 4 项条件，且连续稳定保持 $K=20$ 个控制帧（$0.333\,\text{s}$）：

```
           ┌──────────────────────────────────────────────┐
           │        四维联合成功门禁 (EpisodeJudge)        │
           └──────────────────────┬───────────────────────┘
                                  ▼
      ┌────────────────────────────────────────────────────────┐
      │ ① 空间内腔包络 (Position Containment in Cavity)         │
      │    • |x - x_tray| <= hx - margin (0.125m)              │
      │    • |y - y_tray| <= hy - margin (0.085m)              │
      │    • z_floor_top - 5mm <= z <= z_walls_top + 35mm      │
      └───────────────────────────┬────────────────────────────┘
                                  ▼
      ┌────────────────────────────────────────────────────────┐
      │ ② 姿态直立防倾倒 (Upright Attitude Gate)               │
      │    • R_zz = 1 - 2*(qx^2 + qy^2) >= cos(30°) ≈ 0.866   │
      │    • 严禁碗侧扣、杯子横躺或翻滚                          │
      └───────────────────────────┬────────────────────────────┘
                                  ▼
      ┌────────────────────────────────────────────────────────┐
      │ ③ 动力学静止 (Kinematic & Dynamic Settle)               │
      │    • 线速度 ||v|| <= 0.04 m/s                          │
      │    • 连续保持 Streak >= 20 frames (0.33s)              │
      └───────────────────────────┬────────────────────────────┘
                                  ▼
      ┌────────────────────────────────────────────────────────┐
      │ ④ 夹爪离手防作弊 (Hands-off Disengagement)              │
      │    • TCP 高度 z_tcp > 0.500m (撤离托盘上方)              │
      │    • 确保物体处于纯重力/托盘接触静平衡，绝非机械臂吊挂     │
      └────────────────────────────────────────────────────────┘
```

### 具体数学与几何阈值对照表

| 序号 | 判定维度 | 数学定义与公差 | 工程物理防作弊意义 |
| :---: | :--- | :--- | :--- |
| **1** | **内腔水平包含** | $|x - 0.62| \le 0.125\,\text{m}$<br>$|y - 0.02| \le 0.085\,\text{m}$ | 确保质心整体落入托盘底板内，排除了骑跨在 1cm 宽托盘挡墙上的假置入。 |
| **2** | **内腔垂直高度** | $0.403\,\text{m} \le z \le 0.488\,\text{m}$ | 底线 $0.403\text{m}$ 允许 $5\text{mm}$ 接触弹性压缩，排除穿底掉出；上限排除悬空。 |
| **3** | **直立姿态余弦** | $R_{zz} \ge \cos(30^\circ) \approx 0.866$ | 倾角超过 $30^\circ$ 立即拒判，杜绝倾覆或倒扣杯底。 |
| **4** | **线速度静止** | $\|\mathbf{v}\|_2 \le 0.040\,\text{m/s}$ | 过滤高频碰撞未稳态。若物体在盘内晃动弹跳，计数器立即清零。 |
| **5** | **去抖持续帧数** | $\text{streak} \ge 20\,\text{frames}$ ($0.33\,\text{s}$) | 杜绝瞬时穿过托盘时“恰好一帧速度降到阈值”产生的伪阳性。 |
| **6** | **离手撤退** | $z_{\text{tcp}} > 0.500\,\text{m}$ | 强制机械臂完全完成两段式泄压张爪并上升撤离，确保纯外力解耦。 |

---

## 3. 失败判据与截断条件 (Termination & Truncation)

1. **早停失败终止 (Early Failure Termination - `is_failed`)**：
   - 物体坠落出界：物体质心高度 $z < 0.350\,\text{m}$（掉落桌面坠向地面）；
   - 越界撞飞：物体被击打至桌面边界以外（$x \notin [0.25, 0.95]$ 或 $y \notin [-0.50, 0.50]$）。
   - 触发即判定本回合 `FAILED` 并提前重置，返回大额惩罚（$R_{\text{penalty}} = -10.0$）。
2. **超时截断 (Truncation / Timeout)**：
   - 单回合步数上限 $T_{\max} = 1740\,\text{frames}$（$29.0\,\text{s}$）。超过该步数仍未触发双入盘则判为超时未完成。

---

## 4. 稠密奖励函数塑造 (IsaacLab-style Dense Shaping Reward)

供强化学习策略（如 PPO / SAC）训练使用的逐步奖励：
$$R_{\text{step}} = r_{\text{reach}} + r_{\text{lift}} + r_{\text{place}} + r_{\text{upright}} + r_{\text{task}} + r_{\text{penalty}}$$

- **接近项**：$r_{\text{reach}} = 1.0 - \tanh(5.0 \cdot \|\mathbf{p}_{\text{tcp}} - \mathbf{p}_{\text{obj}}\|)$
- **举升项**：$r_{\text{lift}} = 2.0 \cdot \min\left(1.0, \frac{\max(0, z_{\text{obj}} - z_{\text{spawn}})}{0.05}\right)$
- **运送项**：$r_{\text{place}} = 3.0 \cdot (1.0 - \tanh(3.0 \cdot \|\mathbf{p}_{\text{obj}} - \mathbf{p}_{\text{target}}\|))$
- **姿态项**：$r_{\text{upright}} = 1.0 \quad (\text{if } R_{zz} \ge \cos 30^\circ \text{ else } 0.0)$
- **里程碑项**：
  - 成功抓起离地：$+3.0$
  - 单物体入盘稳态：$+10.0$
  - 掉落桌底惩罚：$-10.0$

---

## 5. 单元测试验证报告 (`test_success.py`)

在无物理仿真引擎的独立环境中，运行全集 8 项边界单测：
```bash
python newton_pick_place/test_success.py
```
**测试输出记录**：
- `[PASS] test_object_in_tray`: 验证正常入盘、底面高差、墙沿碰撞、60°倾倒反例、20°轻微倾斜容忍。
- `[PASS] test_object_fallen_off`: 验证掉落桌底 ($z<0.35$) 及飞出桌面横向边界判定。
- `[PASS] test_at_rest`: 验证线速度 $0.01\,\text{m/s}$ 通过、$0.05\,\text{m/s}$ 拦截。
- `[PASS] test_judge_success`: 验证双物体双入盘连续 20 帧触发回合成功。
- `[PASS] test_judge_streak_reset`: 验证被外力碰撞离开托盘 1 帧即清零重计。
- `[PASS] test_judge_early_failure`: 验证掉落物体阻断成功标记。
- `[PASS] test_judge_grasp_metric`: 验证离地 5cm 抓取指标打标。
- `[PASS] test_reward_terms`: 验证奖励函数在接近、抓取、举起、入盘时的严格单调递增性。
- **结论**：**8/8 项单测全绿 (ALL PASS)**。
