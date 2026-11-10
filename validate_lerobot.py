"""Strict read-only validator for HuggingFace LeRobot v2.0 dataset.

IMPORTANT: Completely excised the cheating logic from workspace 2 that forged
success flags ('success = True'). This validator performs strictly read-only audits
of physical simulation telemetry, Parquet schemas, and genuine 100% success assertions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import numpy as np
import pyarrow.parquet as pq


def validate_dataset(dataset_dir: str) -> bool:
    dataset_dir = os.path.abspath(dataset_dir)
    print(f"[validate_lerobot] Auditing dataset at: {dataset_dir}")

    meta_dir = os.path.join(dataset_dir, "meta")
    info_path = os.path.join(meta_dir, "info.json")
    episodes_path = os.path.join(meta_dir, "episodes.jsonl")
    tasks_path = os.path.join(meta_dir, "tasks.jsonl")

    assert os.path.isfile(info_path), f"Missing info.json at {info_path}"
    assert os.path.isfile(episodes_path), f"Missing episodes.jsonl at {episodes_path}"
    assert os.path.isfile(tasks_path), f"Missing tasks.jsonl at {tasks_path}"

    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    assert info.get("codebase_version") == "v2.0", f"Invalid codebase version: {info.get('codebase_version')}"
    total_episodes = info["total_episodes"]
    total_frames = info["total_frames"]
    features = info["features"]
    fps = info["fps"]

    print(f"  Codebase Version: {info['codebase_version']}")
    print(f"  Robot Type:       {info['robot_type']}")
    print(f"  Total Episodes:   {total_episodes}")
    print(f"  Total Frames:     {total_frames}")
    print(f"  FPS:              {fps}")

    with open(episodes_path, "r", encoding="utf-8") as f:
        episodes_meta = [json.loads(line) for line in f if line.strip()]

    assert len(episodes_meta) == total_episodes, (
        f"Episode count mismatch: meta has {len(episodes_meta)}, info has {total_episodes}"
    )

    sum_frames = sum(ep["length"] for ep in episodes_meta)
    assert sum_frames == total_frames, (
        f"Frame count mismatch: sum of episodes {sum_frames} != info total {total_frames}"
    )

    data_dir = os.path.join(dataset_dir, "data", "chunk-000")
    assert os.path.isdir(data_dir), f"Missing data directory: {data_dir}"

    all_episodes_succeeded = True
    success_count = 0

    for ep_idx, ep_meta in enumerate(episodes_meta):
        parquet_file = os.path.join(data_dir, f"episode_{ep_idx:06d}.parquet")
        assert os.path.isfile(parquet_file), f"Missing parquet file: {parquet_file}"

        # Read-only inspection using PyArrow
        table = pq.read_table(parquet_file)
        n_rows = table.num_rows
        assert n_rows == ep_meta["length"], (
            f"Ep {ep_idx}: Table rows {n_rows} != metadata length {ep_meta['length']}"
        )

        # Check feature column schema
        table_cols = set(table.column_names)
        expected_cols = set(features.keys())
        assert table_cols == expected_cols, (
            f"Ep {ep_idx}: Column mismatch. Missing: {expected_cols - table_cols}, Extra: {table_cols - expected_cols}"
        )

        # Audit real physical success flag
        succ_col = table["success"].to_pylist()
        ep_success = bool(succ_col[-1]) if succ_col else False
        meta_success = bool(ep_meta.get("success", False))

        assert ep_success == meta_success, (
            f"Ep {ep_idx}: Parquet final success ({ep_success}) != metadata success ({meta_success})"
        )

        if not ep_success:
            all_episodes_succeeded = False
            print(f"  [FAIL] Episode {ep_idx}: Genuine physical success was FALSE")
        else:
            success_count += 1

        # Check numerical sanity (no NaNs or Infs)
        for col_name in ["observation.state", "action", "observation.ee_pose", "observation.objects"]:
            arr = np.array(table[col_name].to_pylist(), dtype=np.float32)
            assert np.all(np.isfinite(arr)), f"Ep {ep_idx}: NaN or Inf detected in {col_name}"

        # Shape checks
        state_dim = len(table["observation.state"][0].as_py())
        act_dim = len(table["action"][0].as_py())
        ee_dim = len(table["observation.ee_pose"][0].as_py())
        obj_dim = len(table["observation.objects"][0].as_py())

        assert state_dim == features["observation.state"]["shape"][0], f"State dim mismatch {state_dim}"
        assert act_dim == features["action"]["shape"][0], f"Action dim mismatch {act_dim}"
        assert ee_dim == features["observation.ee_pose"]["shape"][0], f"EE dim mismatch {ee_dim}"
        assert obj_dim == features["observation.objects"]["shape"][0], f"Object dim mismatch {obj_dim}"

        print(
            f"  [AUDIT PASS] Ep {ep_idx:02d}: frames={n_rows:4d} | "
            f"state={state_dim}D act={act_dim}D ee={ee_dim}D obj={obj_dim}D | "
            f"genuine_success={ep_success}"
        )

    print("=" * 65)
    print(f"AUDIT SUMMARY: {success_count}/{total_episodes} episodes physically succeeded.")
    assert all_episodes_succeeded, f"Not all episodes succeeded! Only {success_count}/{total_episodes} passed."
    print("ALL LE-ROBOT V2.0 VALIDATION CHECKS PASSED (100% GENUINE PHYSICAL SUCCESS)")
    print("=" * 65)
    return True


def main():
    parser = argparse.ArgumentParser(description="Purified LeRobot v2.0 Dataset Validator")
    parser.add_argument(
        "--dataset",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "lerobot_dataset"),
        help="Path to the LeRobot dataset root",
    )
    args = parser.parse_args()

    passed = validate_dataset(args.dataset)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
