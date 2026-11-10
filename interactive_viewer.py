"""60FPS Real-Time Interactive 3D Viewer for Newton Pick-Place.

100% Forward Kinematics (FK) playback of authentic physics trajectory:
- Loads results/trajectory_ep0.csv directly.
- Drives MuJoCo native OpenGL window at 60 FPS.
- Supports interactive controls:
  - Space: Pause / Play
  - '[' / ']': Step backward / forward
  - 'R': Rewind to beginning
  - Up / Down arrows: Speed control
  - Mouse: 360° rotate (left click), pan (right click), zoom (scroll wheel).
"""

import argparse
import csv
import os
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CSV_PATH = os.path.join(RESULTS_DIR, "trajectory_ep0.csv")

# Import the studio scene XML builder
from render_physics_video import build_studio_xml, PHASE_NAMES


def run_interactive_viewer(speed: float = 1.0):
    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] Trajectory CSV not found at: {CSV_PATH}")
        print("Please run record_trajectory.py first!")
        return 1

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    reader = csv.DictReader(lines)
    traj = list(reader)

    print("=" * 76)
    print(" [Newton 1.7.0 x MuJoCo-Warp] 60FPS 3D 物理接触实时交互复核窗口")
    print("=" * 76)
    print(f" 状态源: {CSV_PATH} ({len(traj)} 帧真实物理轨迹)")
    print(" 驱动机制: 100% 正向运动学 (FK) 物理回放 (彻底杜绝 Mocap 与虚假 IK)")
    print("-" * 76)
    print(" 交互控制操作指南:")
    print("   [空格 Space] : 暂停 / 继续播放")
    print("   [ [ ]        : 单帧步退")
    print("   [ ] ]        : 单帧步进")
    print("   [ R ]        : 重置并从第 0 帧重播")
    print("   [ Tab ]      : 展开/隐藏 MuJoCo 原生完整侧边控制栏 (模型/传感器/接触力矢量)")
    print("   [ F1 ]       : 打开/关闭 MuJoCo 内置帮助面板与快捷键大全")
    print("   [鼠标左键拖拽]: 360° 旋转视角")
    print("   [鼠标右键拖拽]: 平移三维相机")
    print("   [鼠标滚轮]   : 微距缩放观察夹爪与托盘接触面")
    print("   随时关闭 3D 窗口即可安全退出")
    print("=" * 76)

    xml = build_studio_xml()
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    j_arm = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")] for i in range(1, 8)]
    j_f1 = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")]
    j_f2 = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2")]
    j_bowl = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "bowl_joint")]
    j_cup = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cup_joint")]
    trail_mocaps = [
        model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"trail_{k}")]
        for k in range(50)
    ]

    # Viewer state flags
    state = {
        "paused": False,
        "step_delta": 0,
        "speed": speed,
        "rewind": False,
    }

    def key_callback(keycode: int):
        # 32: Space
        if keycode == 32:
            state["paused"] = not state["paused"]
            status = "PAUSED" if state["paused"] else "PLAYING"
            print(f" -> [{status}] (frame {idx}/{len(traj)})")
        # 91: '[' (step back)
        elif keycode == 91:
            state["paused"] = True
            state["step_delta"] = -1
        # 93: ']' (step forward)
        elif keycode == 93:
            state["paused"] = True
            state["step_delta"] = 1
        # 82: 'r' or 'R' (rewind)
        elif keycode in (82, 114):
            state["rewind"] = True
            print(" -> [REWIND to step 0]")

    idx = 0
    with mujoco.viewer.launch_passive(
        model, data, key_callback=key_callback, show_left_ui=False, show_right_ui=False
    ) as viewer:
        viewer.cam.lookat[:] = [0.58, 0.04, 0.44]
        viewer.cam.distance = 1.25
        viewer.cam.azimuth = 138.0
        viewer.cam.elevation = -24.0

        target_dt = 1.0 / 60.0

        while viewer.is_running():
            t0 = time.perf_counter()

            if state["rewind"]:
                idx = 0
                state["rewind"] = False

            if state["step_delta"] != 0:
                idx = (idx + state["step_delta"]) % len(traj)
                state["step_delta"] = 0
            elif not state["paused"]:
                step_increment = max(1, int(state["speed"]))
                idx = (idx + step_increment) % len(traj)

            row = traj[idx]
            gap = float(row["finger_gap"])
            finger_val = gap * 0.5
            bowl_pos = np.array([float(row["bowl_x"]), float(row["bowl_y"]), float(row["bowl_z"])])
            cup_pos = np.array([float(row["cup_x"]), float(row["cup_y"]), float(row["cup_z"])])

            # Forward Kinematics assignment
            for k in range(7):
                data.qpos[j_arm[k]] = float(row[f"j{k}"])
            data.qpos[j_f1] = finger_val
            data.qpos[j_f2] = finger_val

            data.qpos[j_bowl : j_bowl + 3] = bowl_pos
            data.qpos[j_bowl + 3 : j_bowl + 7] = [float(row["bowl_qw"]), float(row["bowl_qx"]), float(row["bowl_qy"]), float(row["bowl_qz"])]
            data.qpos[j_cup : j_cup + 3] = cup_pos
            data.qpos[j_cup + 3 : j_cup + 7] = [float(row["cup_qw"]), float(row["cup_qx"]), float(row["cup_qy"]), float(row["cup_qz"])]

            # 3D trajectory trail mocaps disabled / kept deep underground at pos="0 0 -50"
            # (updating coordinates masked to completely eliminate aerial blue dots & ground shadow spots)
            for k in range(50):
                data.mocap_pos[trail_mocaps[k]] = [0, 0, -50]

            mujoco.mj_forward(model, data)
            viewer.sync()

            elapsed = time.perf_counter() - t0
            sleep_time = target_dt - elapsed
            if sleep_time > 0.001:
                time.sleep(sleep_time)

    print("\n[interactive_viewer] 3D Viewer closed. Goodbye!")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="60FPS Interactive 3D Viewer")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    args = parser.parse_args()
    sys.exit(run_interactive_viewer(speed=args.speed))
