"""Gate check 3/3: run N evaluation episodes of the full pick-place task.

For each episode: reset with a fresh seed, run the scripted IK agent, judge
the task (bowl AND cup placed in the tray, upright, at rest, gripper clear).
Streams high-frequency microsecond physics telemetry through PhysicsTelemetryBridge,
writes results/results.json, and exports to CSV / Parquet / LeRobot v2.0 datasets.

Exit code 0 iff success_rate >= --min-success.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import warp as wp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent import PickPlaceAgent  # noqa: E402
from env import PickPlaceEnv  # noqa: E402
from export_lerobot import LeRobotDatasetExporter  # noqa: E402
from task import EpisodeJudge, TaskConfig  # noqa: E402
from telemetry_bridge import PhysicsTelemetryBridge  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def parse_phase_id(phase_name: str) -> int:
    """Map symbolic agent phase name to standard 0..7 integer ID."""
    if "home" in phase_name or "hover" in phase_name:
        return 0
    if "descend" in phase_name:
        return 1
    if "grasp" in phase_name:
        return 2
    if "lift" in phase_name:
        return 3
    if "move" in phase_name:
        return 4
    if "lower" in phase_name:
        return 5
    if "release" in phase_name or "open" in phase_name:
        return 6
    if "retract" in phase_name or "settle" in phase_name:
        return 7
    return 0


def run_episode(
    env: PickPlaceEnv,
    agent: PickPlaceAgent,
    judge: EpisodeJudge,
    cfg: TaskConfig,
    seed: int,
    bridge: PhysicsTelemetryBridge | None = None,
    collect_lerobot: bool = False,
):
    poses = env.reset(seed=seed)
    agent.reset_episode({k: np.array([v.p[0], v.p[1], v.p[2]]) for k, v in poses.items()})
    spawn_z = {name: float(env.body_poses()[b, 2]) for name, b in env.object_bodies.items()}

    if bridge is not None:
        bridge.reset()

    lerobot_buffers = {
        "states": [],
        "actions": [],
        "ee_poses": [],
        "objects": [],
        "subtasks": [],
        "phases": [],
        "timestamps": [],
    } if collect_lerobot else None

    bowl_b = env.object_bodies["bowl"]
    cup_b = env.object_bodies["cup"]
    dest_bowl = np.array([0.56, 0.02, 0.408], dtype=np.float32)
    dest_cup = np.array([0.68, 0.02, 0.408], dtype=np.float32)

    frames = 0
    outcome, success_frame = "timeout", None

    for i in range(cfg.episode_frames):
        action_target = agent.act()
        env.step(action_target)
        frames += 1

        q = env.body_poses()
        qd = env.body_vels()
        tcp_z = float(q[env.tcp_idx, 2])

        for name, b in env.object_bodies.items():
            judge.observe(
                name,
                q[b],
                float(np.linalg.norm(qd[b, :3])),
                float(np.linalg.norm(qd[b, 3:])),
                spawn_z[name],
                hands_off=bool(tcp_z > cfg.hands_off_tcp_z),
            )

        cur_phase_name = agent.phase_name
        phase_id = parse_phase_id(cur_phase_name)
        subtask_id = 1 if ("cup" in cur_phase_name or "settle" in cur_phase_name) else 0

        tcp_pos, tcp_rot = env.tcp_pose()
        cur_obj_xyz = q[cup_b, :3] if subtask_id == 1 else q[bowl_b, :3]
        cur_dest_xyz = dest_cup if subtask_id == 1 else dest_bowl

        is_success = judge.success()

        if bridge is not None:
            gates = (
                float(judge.was_grasped["bowl"]),
                float(judge.was_placed["bowl"]),
                float(judge.was_grasped["cup"]),
                float(judge.was_placed["cup"]),
            )
            bridge.record_step(
                step=i,
                t=float(env.sim_time),
                phase=phase_id,
                subtask=subtask_id,
                tip_xyz=tcp_pos,
                obj_xyz=cur_obj_xyz,
                dest_xyz=cur_dest_xyz,
                grip=float(action_target[7]),
                arm_j=env.arm_joint_q(),
                arm_jd=env.arm_joint_qd(),
                finger_gap=float(env.finger_gap()),
                tip_q=tcp_rot,
                bowl_q=q[bowl_b, 3:7],
                cup_q=q[cup_b, 3:7],
                gates=gates,
                success=float(is_success),
                bowl_xyz=q[bowl_b, :3],
                cup_xyz=q[cup_b, :3],
            )

        if collect_lerobot:
            lerobot_buffers["states"].append(np.hstack([env.arm_joint_q(), [env.finger_gap()]]))
            lerobot_buffers["actions"].append(action_target[:8].copy())
            lerobot_buffers["ee_poses"].append(np.hstack([tcp_pos, tcp_rot]))
            lerobot_buffers["objects"].append(np.hstack([q[bowl_b, :7], q[cup_b, :7]]))
            lerobot_buffers["subtasks"].append(subtask_id)
            lerobot_buffers["phases"].append(phase_id)
            lerobot_buffers["timestamps"].append(float(env.sim_time))

        if is_success:
            outcome, success_frame = "success", i
            break

    q = env.body_poses()
    record = {
        "seed": seed,
        "outcome": outcome,
        "success_frame": success_frame,
        "frames": frames,
        "grasped": dict(judge.was_grasped),
        "final_poses": {
            name: [round(float(v), 4) for v in q[b, :7]]
            for name, b in env.object_bodies.items()
        },
        "spawn_poses": {
            name: [round(float(v.p[0]), 4), round(float(v.p[1]), 4)] for name, v in poses.items()
        },
    }

    if collect_lerobot:
        record["lerobot_data"] = {
            k: np.array(v, dtype=np.float32 if k not in ("subtasks", "phases") else np.int64)
            for k, v in lerobot_buffers.items()
        }

    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--min-success", type=float, default=0.66)
    parser.add_argument("--video", dest="video", action="store_true", default=False)
    parser.add_argument("--no-video", dest="video", action="store_false")
    parser.add_argument("--dump-traj", type=str, default=None, help="Path to dump trajectory CSV or Parquet")
    parser.add_argument("--export-lerobot", type=str, default=None, help="Export dataset in LeRobot v2.0 format")
    args = parser.parse_args()

    wp.init()
    cfg = TaskConfig()
    env = PickPlaceEnv()
    agent = PickPlaceAgent(env)

    bridge = PhysicsTelemetryBridge(capacity=cfg.episode_frames)
    exporter = LeRobotDatasetExporter(args.export_lerobot) if args.export_lerobot else None

    records = []
    t0 = time.time()

    for ep in range(args.episodes):
        seed = args.seed_base + ep
        judge = EpisodeJudge(cfg)

        rec = run_episode(
            env=env,
            agent=agent,
            judge=judge,
            cfg=cfg,
            seed=seed,
            bridge=bridge,
            collect_lerobot=(exporter is not None),
        )

        records.append(rec)
        print(
            f"[ep {ep}] seed={seed} -> {rec['outcome']}@{rec['success_frame']} "
            f"grasped={rec['grasped']}",
            flush=True,
        )

        # Dump trajectory for episode 0 (or all if formatted with {ep})
        if args.dump_traj:
            if "{ep}" in args.dump_traj:
                traj_out = args.dump_traj.format(ep=ep)
            else:
                traj_out = args.dump_traj if ep == 0 else None

            if traj_out:
                if traj_out.endswith(".parquet"):
                    bridge.dump_parquet(traj_out)
                else:
                    bridge.dump_csv(traj_out)
                print(f"    trajectory dumped -> {traj_out} ({bridge.steps_recorded} frames)", flush=True)

        if exporter is not None:
            ld = rec.pop("lerobot_data")
            is_succ = (rec["outcome"] == "success")
            exporter.add_episode(
                episode_index=ep,
                states=ld["states"],
                actions=ld["actions"],
                ee_poses=ld["ee_poses"],
                objects=ld["objects"],
                subtasks=ld["subtasks"],
                phases=ld["phases"],
                timestamps=ld["timestamps"],
                success=is_succ,
            )

    if exporter is not None:
        exporter.finalize()

    n_ok = sum(1 for r in records if r["outcome"] == "success")
    rate = n_ok / len(records) if records else 0.0
    summary = {
        "task": "tabletop pick-place: bowl + cup into tray (kitless Newton, FR3, scripted IK agent)",
        "episodes": len(records),
        "successes": n_ok,
        "success_rate": round(rate, 3),
        "min_success_required": args.min_success,
        "passed": rate >= args.min_success,
        "wall_time_s": round(time.time() - t0, 1),
        "records": records,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print(f"SUCCESS RATE: {n_ok}/{len(records)} = {rate:.0%}  (required >= {args.min_success:.0%})")
    print(f"results -> {out_path}")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
