"""HuggingFace LeRobot v2.0 dataset exporter for IsaacLab / Newton pick-place simulation.

Converts recorded physical simulation trajectories into canonical LeRobot v2.0 Parquet format:
  <dataset_root>/
    meta/
      info.json
      episodes.jsonl
      tasks.jsonl
    data/
      chunk-000/
        episode_000000.parquet
        episode_000001.parquet
        ...

Follows official HuggingFace LeRobot v2.0 schema specifications with zero tampering.
"""

from __future__ import annotations

import json
import os
import shutil
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


class LeRobotDatasetExporter:
    """Exports multi-episode rollouts into standard LeRobot v2.0 dataset."""

    TASK_NAME = "tabletop_bowl_cup_in_tray"
    ROBOT_TYPE = "franka_fr3"
    FPS = 60

    def __init__(self, output_dir: str):
        self.output_dir = os.path.abspath(output_dir)
        self.meta_dir = os.path.join(self.output_dir, "meta")
        self.chunk_dir = os.path.join(self.output_dir, "data", "chunk-000")
        os.makedirs(self.meta_dir, exist_ok=True)
        os.makedirs(self.chunk_dir, exist_ok=True)

        self.episodes: list[dict] = []
        self.total_frames = 0
        self.cumulative_index = 0

    def add_episode(
        self,
        episode_index: int,
        states: np.ndarray,       # (N, 8) [j0..j6, finger_gap]
        actions: np.ndarray,      # (N, 8) [target_j0..j6, grip_cmd]
        ee_poses: np.ndarray,     # (N, 7) [tip_xyz(3), tip_quat(4)]
        objects: np.ndarray,      # (N, 14) [bowl_pose(7), cup_pose(7)]
        subtasks: np.ndarray,     # (N,)
        phases: np.ndarray,       # (N,)
        timestamps: np.ndarray,   # (N,)
        success: bool,            # genuine physical success flag
    ):
        """Append one verified episode and write its Parquet file."""
        n_frames = len(states)
        if n_frames == 0:
            raise ValueError(f"Cannot add empty episode {episode_index}")

        # Construct PyArrow arrays
        frame_indices = np.arange(n_frames, dtype=np.int64)
        ep_indices = np.full(n_frames, episode_index, dtype=np.int64)
        global_indices = np.arange(self.cumulative_index, self.cumulative_index + n_frames, dtype=np.int64)
        task_indices = np.zeros(n_frames, dtype=np.int64)
        success_array = np.full(n_frames, bool(success), dtype=bool)

        table = pa.table({
            "observation.state": pa.array(states.astype(np.float32).tolist(), type=pa.list_(pa.float32())),
            "action": pa.array(actions.astype(np.float32).tolist(), type=pa.list_(pa.float32())),
            "observation.ee_pose": pa.array(ee_poses.astype(np.float32).tolist(), type=pa.list_(pa.float32())),
            "observation.objects": pa.array(objects.astype(np.float32).tolist(), type=pa.list_(pa.float32())),
            "observation.subtask": pa.array(subtasks.astype(np.int64).tolist(), type=pa.int64()),
            "observation.phase": pa.array(phases.astype(np.int64).tolist(), type=pa.int64()),
            "success": pa.array(success_array, type=pa.bool_()),
            "timestamp": pa.array(timestamps.astype(np.float32).tolist(), type=pa.float32()),
            "frame_index": pa.array(frame_indices, type=pa.int64()),
            "episode_index": pa.array(ep_indices, type=pa.int64()),
            "index": pa.array(global_indices, type=pa.int64()),
            "task_index": pa.array(task_indices, type=pa.int64()),
        })

        parquet_filename = f"episode_{episode_index:06d}.parquet"
        parquet_path = os.path.join(self.chunk_dir, parquet_filename)
        pq.write_table(table, parquet_path)

        self.episodes.append({
            "episode_index": episode_index,
            "tasks": [self.TASK_NAME],
            "length": n_frames,
            "success": bool(success),
        })

        self.total_frames += n_frames
        self.cumulative_index += n_frames

    def finalize(self):
        """Write metadata JSON files compliant with LeRobot v2.0."""
        n_episodes = len(self.episodes)

        features = {
            "observation.state": {"dtype": "float32", "shape": [8]},
            "action": {"dtype": "float32", "shape": [8]},
            "observation.ee_pose": {"dtype": "float32", "shape": [7]},
            "observation.objects": {"dtype": "float32", "shape": [14]},
            "observation.subtask": {"dtype": "int64", "shape": [1]},
            "observation.phase": {"dtype": "int64", "shape": [1]},
            "success": {"dtype": "bool", "shape": [1]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
        }

        info = {
            "codebase_version": "v2.0",
            "robot_type": self.ROBOT_TYPE,
            "total_episodes": n_episodes,
            "total_frames": self.total_frames,
            "total_tasks": 1,
            "total_videos": 0,
            "total_chunks": 1,
            "chunks_size": 1000,
            "fps": self.FPS,
            "splits": {"train": f"0:{n_episodes}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "features": features,
        }

        with open(os.path.join(self.meta_dir, "info.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)

        with open(os.path.join(self.meta_dir, "episodes.jsonl"), "w", encoding="utf-8") as f:
            for ep in self.episodes:
                f.write(json.dumps(ep) + "\n")

        with open(os.path.join(self.meta_dir, "tasks.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps({"task_index": 0, "task": self.TASK_NAME}) + "\n")

        print(
            f"[export_lerobot] Exported {n_episodes} episodes ({self.total_frames} frames) "
            f"to {self.output_dir}"
        )
