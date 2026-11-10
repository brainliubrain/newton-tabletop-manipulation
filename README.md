# 桌面抓放任务:碗 + 杯放入托盘(Kitless Newton)

> 面试题实现:基于最新 IsaacLab 3.0 同源技术栈(Newton 物理引擎)构建仿真 env——桌面摆放一个托盘、一个碗和一个杯子,机器人把碗和杯子都放入托盘;机器人采用 isaaclab 内同款 Franka FR3;1) 构建完整的 task 和成功标准;2) 用模型/agent 控制机器人完成任务。
>
> **本机约束**:RTX 3060 6GB + Isaac Sim 4.5。IsaacLab 3.0 完整栈需要 Isaac Sim 5.x(官方建议 16GB+ 显存),因此采用 **kitless 路线**:直接使用 Newton 引擎本体(`newton` 1.7.0,MuJoCo-Warp GPU 求解器),无需 Isaac Sim。任务定义、机器人资产、代码组织均与 IsaacLab 3.0 同构,可平移。

**最终结果:10/10 回合 100% 成功率**(每回合随机化物体出生位姿),单回合约 22s 仿真 / 11ms·帧⁻¹(GPU),演示视频 `results/demo_ep0.mp4`。

---

## 1. 快速开始

```bash
# 依赖(清华镜像):newton 1.7.0 + warp + mujoco/mujoco-warp 3.15 + 导入器依赖
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    "newton==1.7.0" "mujoco==3.15.*" "mujoco-warp==3.15.*" \
    pycollada resolve-robotics-uri-py "usd-core<26.9" newton-usd-schemas \
    scipy trimesh fast-simplification imageio imageio-ffmpeg

# 环境自检(CUDA + URDF + USD 资产 + GPU 仿真步进)
.venv/Scripts/python.exe newton_pick_place/check_env.py

# 成功判据单元测试(纯 numpy,无仿真)
.venv/Scripts/python.exe newton_pick_place/test_success.py

# 正式评测:10 个随机化回合 + 输出 results.json + 第一回合演示视频(~4.5 min)
.venv/Scripts/python.exe newton_pick_place/run_task.py --episodes 10 --min-success 0.66

# 快速冒烟验收(3 回合,~100 s,gate.json 采用此命令)
.venv/Scripts/python.exe newton_pick_place/run_task.py --episodes 3 --no-video --min-success 0.66
```

资产已 vendored 到 `newton_pick_place/assets/`(FR3 URDF+网格、Newton 官方 cup 资产),**离线可复现**;原始来源 `newton-physics/newton-assets@f8fb7abc`(pinned SHA)。

## 2. 任务定义与成功标准(task.py)

**任务**:FR3(固定基座,7 DOF + 平行夹爪)将桌面上的**碗**和**杯**先后抓取并放入**托盘**内腔。

**成功判定( EpisodeJudge,逐帧流式评估)**——单物体"已放置"需连续 20 帧(0.33s)同时满足:

| # | 判据 | 阈值 |
|---|------|------|
| 1 | **位置**:物体中心在托盘内腔水平投影内(距内壁 ≥1cm),高度在托盘底面与壁顶+3cm 之间 | 内腔半宽 0.135×0.095m |
| 2 | **姿态**:未倾倒(up 轴与 +z 夹角) | < 30° |
| 3 | **静止**:线速度 | < 0.04 m/s |
| 4 | **脱手**:TCP 高于托盘上方(不许拿着不放) | TCP z > 0.50m |

**回合成功** = 碗和杯**同时**满足上述条件。另跟踪指标:是否发生过抓取(举高 >5cm)、成功耗时、最终/出生位姿(全部写入 `results/results.json`)。

附 dense reward(到达/举升/放置/任务完成四项,IsaacLab 风格),供 RL 策略替换时使用。

## 3. 系统架构

```
env.py     PickPlaceEnv        场景构建(FR3 URDF + 桌/托盘盒体组合 + 旋转网格碗 + cup USD 资产)
                               reset(seed) 随机化物体位姿 → step(9 维关节目标) 60Hz 控制 × 10 子步(600Hz)
                               CUDA Graph capture;MuJoCo-Warp 求解器(cuda:0)
task.py    TaskConfig          成功判据/阈值(纯 numpy,可单测)
           EpisodeJudge        流式判定器(位置/姿态/静止/脱手)
           reward()            稠密塑形奖励
agent.py   FrankaIK            newton.ik.IKSolver 解析 IK(ANALYTIC 雅可比,关节限位目标)
           PickPlaceAgent      路标点状态机(hover→descend→grasp→lift→move→lower→
                               release(两段式:先松压再张开)→retract)
                               感知接口:只读物体位姿(等价感知模块输出),换 RL/MLP 策略只需替换 act()
run_task.py                    多回合评测器:成功率聚合 + results.json + mp4 录制
check_env.py                   环境自检(warp CUDA / GPU 刚体世界 / URDF / USD / PD 伺服步进)
```

**与 IsaacLab 3.0 的映射**:同一 FR3 资产(isaaclab `franka_emika_panda`)、同一 ManagerBased 结构分层(env / task 判据 / agent 策略)、同一随机化 reset 语义;差别仅在物理后端(IsaacLab 3.0 默认 PhysX,此处为 Newton 求解器)与运行时(无需 Kit)。在配备 Isaac Sim 5.x 的机器上,`env.py` 的场景构建可换用 `isaaclab.sim` 的 spawn API,判据与 agent 层零改动。

## 4. 评测结果(results/results.json)

| 指标 | 值 |
|------|-----|
| 回合数 | 10(seeds 0–9,出生位姿+朝向随机) |
| 成功率 | **10/10 = 100%**(验收线 66%) |
| 平均成功耗时 | ~21.5s 仿真时间(~1290 控制帧) |
| 双物体抓取率 | 10/10 |
| 性能 | 11 ms/帧(RTX 3060 6GB,CUDA Graph) |

## 5. 关键工程问题与解法(完整版见 docs/PROBLEMS.md)

1. **接触参数反解**:newton 默认接触刚度让轻物体陷入桌面 14mm——由 newton→mujoco solref 转换公式反解出 ke=2.5e8/kd=5e4(临界~过阻尼),物体落位误差 <0.1mm 且静止时无微振。
2. **自由体位姿权威在 `state.joint_q`**(reduced-coordinate):reset 只写 body_q 会被 FK 覆盖(杯"瞬移"回默认位)——必须写 FREE 关节坐标再 `eval_fk`。
3. **伺服下垂与哨卡位形**:伸展位姿重力矩大,PD 稳态下垂 ~1–3cm;慢速笛卡尔下降在某些位形卡死(无接触、无碰撞、力矩未饱和的纯伺服病态)——用官方同款路标点+整定式 PTP 流程 + 释放高度贴底规避。
4. **"西瓜籽"弹射**:夹爪松开瞬间, stored squeeze energy 把轻物体弹飞(官方 panda_hydro 注释同款问题)——两段式释放(先松压至近接触点,再张爪)解决。
5. **出生穿模**:防撞检查用"到托盘中心距离"但托盘是矩形——物体与 -x 壁穿模出生即被弹射(确定性 -43m 事故)——改为矩形+半径重叠检测。
6. **性能**:CUDA Graph capture 把 698ms/帧 → 11ms/帧(63×)。

## 6. 演示建议(面试现场)

1. 播放 `results/demo_ep0.mp4`(21s,含随机出生位姿的完整抓放);
2. 现场跑 `run_task.py --episodes 3 --no-video` 展示实时评测与成功率;
3. 打开 `results/results.json` 展示逐回合记录(判据数据全部落盘);
4. 讲 PROBLEMS.md 里的 2–3 个深挖问题(接触参数反解 / 自由体状态权威 / 弹射问题)体现 hands-on 深度。
