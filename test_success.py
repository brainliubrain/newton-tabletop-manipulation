"""Gate check 2/3: unit tests for the task success & failure criteria (pure numpy, no sim).

Run: .venv/Scripts/python.exe newton_pick_place/test_success.py
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from task import (  # noqa: E402
    EpisodeJudge,
    TaskConfig,
    TaskSpec,
    at_rest,
    object_fallen_off,
    object_in_tray,
    reward,
)

CFG = TaskConfig()
TRAY = (0.62, 0.02)


def quat_identity() -> np.ndarray:
    return np.array([0.0, 0.0, 0.0, 1.0])


def quat_tilt(angle_deg: float) -> np.ndarray:
    """Quaternion of a rotation about +y by angle (tilts the up-axis)."""
    a = math.radians(angle_deg) / 2.0
    return np.array([0.0, math.sin(a), 0.0, math.cos(a)])


def pose(x=0.5, y=0.0, z=0.40, q=None) -> np.ndarray:
    return np.array([x, y, z, *(q if q is not None else quat_identity())])


def test_object_in_tray():
    # bowl resting on the table (its natural height) -> not in tray
    assert not object_in_tray(pose(z=0.400), CFG)
    # bowl standing on the tray floor (center = floor_top + half height 0.017)
    assert object_in_tray(pose(x=0.60, y=0.02, z=0.425), CFG), "bowl properly in tray must pass"
    # cup standing on the tray floor (center = floor_top + 0.042)
    assert object_in_tray(pose(x=0.68, y=0.0, z=0.450), CFG), "cup properly in tray must pass"
    # outside the cavity in x / y
    assert not object_in_tray(pose(x=0.85, y=0.02, z=0.425), CFG), "beyond +x wall"
    assert not object_in_tray(pose(x=0.60, y=0.15, z=0.425), CFG), "beyond +y wall"
    # inside xy but resting height = on the table, below the tray floor
    assert not object_in_tray(pose(x=0.60, y=0.02, z=0.400), CFG), "below tray floor"
    # inside xy but floating way above the walls
    assert not object_in_tray(pose(x=0.60, y=0.02, z=0.55), CFG), "above walls+allowance"
    # inside xy, correct z, but toppled 60 deg
    assert not object_in_tray(pose(x=0.60, y=0.02, z=0.425, q=quat_tilt(60)), CFG), "toppled bowl"
    # tilted 20 deg is acceptable
    assert object_in_tray(pose(x=0.60, y=0.02, z=0.425, q=quat_tilt(20)), CFG), "slight tilt ok"
    # margin: center exactly at the inner wall (dx = inner_half) must fail
    assert not object_in_tray(pose(x=TRAY[0] + CFG.tray_inner_half[0], y=TRAY[1], z=0.425), CFG), "on the wall"
    print("PASS object_in_tray")


def test_object_fallen_off():
    # Normal on table
    assert not object_fallen_off(pose(x=0.50, y=0.0, z=0.42), CFG)
    # Dropped off table to the floor
    assert object_fallen_off(pose(x=0.50, y=0.0, z=0.30), CFG), "below table height must trigger fall off"
    # Kicked out of table lateral boundaries
    assert object_fallen_off(pose(x=1.10, y=0.0, z=0.42), CFG), "out of table x bounds"
    assert object_fallen_off(pose(x=0.50, y=0.60, z=0.42), CFG), "out of table y bounds"
    print("PASS object_fallen_off")


def test_at_rest():
    assert at_rest(0.01, 0.1, CFG)
    assert not at_rest(0.05, 0.1, CFG), "too fast linear"
    assert at_rest(0.01, 0.5, CFG), "0.5 rad/s is the metric threshold"
    assert not at_rest(0.01, 0.6, CFG), "too fast angular"
    print("PASS at_rest")


def test_judge_success():
    judge = EpisodeJudge(CFG)
    bowl = pose(x=0.60, y=0.02, z=0.425)
    cup = pose(x=0.68, y=0.0, z=0.450)
    for _ in range(CFG.settle_frames + 5):
        judge.observe("bowl", bowl, 0.0, 0.0, spawn_z=0.40)
        judge.observe("cup", cup, 0.0, 0.0, spawn_z=0.44)
    assert judge.object_placed("bowl") and judge.object_placed("cup")
    assert judge.success()
    print("PASS judge: both placed -> success")


def test_judge_streak_reset():
    judge = EpisodeJudge(CFG)
    bowl = pose(x=0.60, y=0.02, z=0.425)
    for _ in range(CFG.settle_frames - 1):
        judge.observe("bowl", bowl, 0.0, 0.0, spawn_z=0.40)
    assert not judge.object_placed("bowl"), "streak shorter than settle_frames"
    # knocked out of the tray for one frame -> streak resets
    judge.observe("bowl", pose(x=0.30, y=0.0, z=0.40), 0.3, 0.0, spawn_z=0.40)
    assert judge.in_tray_streak["bowl"] == 0
    for _ in range(CFG.settle_frames):
        judge.observe("bowl", bowl, 0.0, 0.0, spawn_z=0.40)
    assert judge.object_placed("bowl")
    assert not judge.success(), "only one object observed so far"
    print("PASS judge: streak reset + single object incomplete")


def test_judge_early_failure():
    judge = EpisodeJudge(CFG)
    bowl = pose(x=0.60, y=0.02, z=0.425)
    cup = pose(x=0.68, y=0.0, z=0.20)  # dropped to floor
    for _ in range(CFG.settle_frames + 5):
        judge.observe("bowl", bowl, 0.0, 0.0, spawn_z=0.40)
        judge.observe("cup", cup, 0.0, 0.0, spawn_z=0.44)
    assert judge.failed, "judge must register early failure"
    assert not judge.success(), "failed episode cannot be success"
    print("PASS judge: early failure detection")


def test_judge_grasp_metric():
    judge = EpisodeJudge(CFG)
    judge.observe("bowl", pose(z=0.40), 0.0, 0.0, spawn_z=0.40)
    assert not judge.was_grasped.get("bowl", False)
    judge.observe("bowl", pose(x=0.45, z=0.50), 0.1, 0.0, spawn_z=0.40)  # lifted 10 cm
    assert judge.was_grasped["bowl"], "lift >5cm counts as grasped"
    print("PASS judge: grasp metric")


def test_reward_terms():
    cfg = CFG
    tcp = np.array([0.5, 0.0, 0.5])
    obj = pose(x=0.45, y=0.0, z=0.40)
    spawn = np.array([0.45, 0.0, 0.40])
    target = np.array([0.60, 0.02, 0.425])
    r_far = reward(tcp, obj, spawn, target, cfg, grasped=False, placed=False)
    r_near = reward(np.array([0.45, 0.0, 0.45]), obj, spawn, target, cfg, grasped=False, placed=False)
    r_lifted = reward(tcp, np.array([0.45, 0.0, 0.46]), spawn, target, cfg, grasped=True, placed=False)
    r_placed = reward(tcp, np.array([0.60, 0.02, 0.425]), spawn, target, cfg, grasped=True, placed=True)
    r_fail = reward(tcp, obj, spawn, target, cfg, grasped=False, placed=False, fallen_off=True)
    assert r_near > r_far, "closer reach must score higher"
    assert r_lifted > r_near, "lift+grasp must score higher"
    assert r_placed > r_lifted, "placed must score highest"
    assert r_fail < -5.0, "fall penalty must be negative"
    print("PASS reward ordering")


if __name__ == "__main__":
    test_object_in_tray()
    test_object_fallen_off()
    test_at_rest()
    test_judge_success()
    test_judge_streak_reset()
    test_judge_early_failure()
    test_judge_grasp_metric()
    test_reward_terms()
    print("TEST_SUCCESS: ALL PASS (8/8 TESTS GREEN)")
