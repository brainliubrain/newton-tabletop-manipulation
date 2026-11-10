"""Task definition, MDP formulation, and success/failure criteria.

Standardized for IsaacLab 3.0 / Newton 1.7.0 kitless simulation:
Tabletop Pick-and-Place: Franka FR3 places Ceramic Bowl and Coffee Cup into Serving Tray.

This module defines:
  1. TaskConfig: Metric geometry, physical thresholds, and tolerance gates.
  2. TaskSpec: Formal MDP specification (Observation space, Action space, Reward terms).
  3. Geometric & Kinematic predicates:
     - object_in_tray(): 3D inner cavity bounding box + upright tilt check.
     - object_fallen_off(): Early-failure detection (fell off table or out of workspace).
     - at_rest(): Linear/angular velocity settling threshold.
  4. EpisodeJudge: Streaming online judge tracking per-object states, grasp metrics,
     settle streaks, and disengagement (hands-off).
  5. reward(): IsaacLab-style multi-stage dense shaping reward.

Everything here is pure numpy/math with zero simulator dependencies, enabling
isolated unit testing (test_success.py) and zero-overhead execution in any policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class SubtaskState(Enum):
    SPAWNED = "spawned"
    GRASPED = "grasped"
    TRANSPORTING = "transporting"
    PLACED = "placed"
    SETTLED = "settled"
    FAILED = "failed"


@dataclass
class TaskConfig:
    """Quantitative thresholds and metric criteria (metric units, z-up)."""

    # --- Tabletop & Tray Geometry ---
    table_top_z: float = 0.400
    table_bounds_x: tuple[float, float] = (0.25, 0.95)
    table_bounds_y: tuple[float, float] = (-0.50, 0.50)
    fall_off_z: float = 0.350  # z below this means object fell off the table

    # Tray inner cavity: base center (0.62, 0.02), base thickness 8mm, wall height 45mm
    tray_center_xy: tuple[float, float] = (0.62, 0.02)
    tray_inner_half: tuple[float, float] = (0.135, 0.095)  # half length x, half width y
    tray_floor_top: float = 0.408  # table_top(0.400) + base_thickness(0.008)
    tray_walls_top: float = 0.453  # floor_top(0.408) + wall_height(0.045)
    tray_z_allowance: float = 0.035  # object center may sit slightly above walls top
    center_to_wall_margin: float = 0.010  # safety margin inside inner walls (10mm)

    # --- Success Gates ---
    max_tilt_deg: float = 30.0  # object up-axis tilt w.r.t. world +z (cos 30 deg ≈ 0.866)
    settle_lin_speed: float = 0.040  # m/s (kinematically at rest)
    settle_ang_speed: float = 0.500  # rad/s (metric threshold)
    settle_frames: int = 20  # consecutive control frames @60Hz (0.333s)
    hands_off_tcp_z: float = 0.500  # gripper TCP z must be > 0.50m (fully retracted)
    gripper_open_min: float = 0.035  # gripper width > 35mm (fingers open)

    # --- Auxiliary Tracking ---
    lift_height: float = 0.050  # raised >= 5cm above spawn height counts as "grasped"
    episode_frames: int = 1740  # 29s @60Hz per episode


@dataclass
class TaskSpec:
    """Formal MDP specification metadata matching IsaacLab 3.0 conventions."""

    task_name: str = "IsaacLab-Newton-Tabletop-BowlCupInTray-v0"
    robot: str = "Franka FR3 (7-DoF arm + 2-finger parallel gripper)"
    required_objects: tuple[str, ...] = ("bowl", "cup")
    destination: str = "tray"

    # Observation Space: 29-dim continuous vector (or Dict observation)
    # [q(7), dq(7), gripper_width(1), tcp_pose(7), bowl_pose(7), cup_pose(7)]
    obs_dim: int = 36
    # Action Space: 8-dim continuous vector [delta_q(7) or IK pos, gripper_cmd(1)]
    action_dim: int = 8

    # Reward Weights (MDP Shaping)
    w_reach: float = 1.0
    w_lift: float = 2.0
    w_place: float = 3.0
    w_upright: float = 1.5
    w_settle: float = 2.0
    w_success: float = 10.0
    w_penalty_fall: float = -10.0


def object_in_tray(
    object_pose: np.ndarray,
    cfg: TaskConfig,
    tray_center_xy: tuple[float, float] | None = None,
) -> bool:
    """Criterion 1 & 2: Positional containment within tray cavity and upright attitude.

    Parameters
    ----------
    object_pose : np.ndarray
        (7,) array [x, y, z, qx, qy, qz, qw] (Warp quaternion convention, qw last).
    cfg : TaskConfig
        Geometric thresholds and tolerances.
    tray_center_xy : tuple, optional
        Center (x, y) of the tray. Defaults to cfg.tray_center_xy.

    Returns
    -------
    bool
        True iff object center is strictly inside the inner cavity and tilt <= 30 deg.
    """
    center_xy = tray_center_xy if tray_center_xy is not None else cfg.tray_center_xy
    x, y, z = float(object_pose[0]), float(object_pose[1]), float(object_pose[2])
    qx, qy = float(object_pose[3]), float(object_pose[4])

    dx = x - center_xy[0]
    dy = y - center_xy[1]
    mx = cfg.tray_inner_half[0] - cfg.center_to_wall_margin
    my = cfg.tray_inner_half[1] - cfg.center_to_wall_margin

    # 1. Horizontal cavity bounds
    if not (-mx <= dx <= mx and -my <= dy <= my):
        return False

    # 2. Vertical height band (resting on tray floor, not buried, not floating)
    if not (cfg.tray_floor_top - 0.005 <= z <= cfg.tray_walls_top + cfg.tray_z_allowance):
        return False

    # 3. Attitude: rotation of local z-axis [0, 0, 1] into world frame
    # R_zz = 1 - 2*(qx^2 + qy^2)
    up_z = 1.0 - 2.0 * (qx * qx + qy * qy)
    return up_z >= np.cos(np.deg2rad(cfg.max_tilt_deg))


def object_fallen_off(object_pose: np.ndarray, cfg: TaskConfig) -> bool:
    """Early failure check: object dropped off the table or kicked out of bounds."""
    x, y, z = float(object_pose[0]), float(object_pose[1]), float(object_pose[2])
    if z < cfg.fall_off_z:
        return True
    if not (cfg.table_bounds_x[0] <= x <= cfg.table_bounds_x[1]):
        return True
    if not (cfg.table_bounds_y[0] <= y <= cfg.table_bounds_y[1]):
        return True
    return False


def at_rest(lin_speed: float, ang_speed: float, cfg: TaskConfig) -> bool:
    """Criterion 3: Physical settling threshold."""
    return lin_speed <= cfg.settle_lin_speed and ang_speed <= cfg.settle_ang_speed


class EpisodeJudge:
    """Streaming online evaluator for task success, early failures, and metrics.

    Feed per-frame telemetry from simulator, query state at any instant.
    """

    REQUIRED_OBJECTS = ("bowl", "cup")

    def __init__(self, cfg: TaskConfig | None = None):
        self.cfg = cfg or TaskConfig()
        self.was_grasped: dict[str, bool] = {n: False for n in self.REQUIRED_OBJECTS}
        self.was_placed: dict[str, bool] = {n: False for n in self.REQUIRED_OBJECTS}
        self._spawn_z: dict[str, float] = {}
        self.in_tray_streak: dict[str, int] = {n: 0 for n in self.REQUIRED_OBJECTS}
        self.object_states: dict[str, SubtaskState] = {n: SubtaskState.SPAWNED for n in self.REQUIRED_OBJECTS}
        self.failed: bool = False
        self.failure_reason: str = ""

    def observe(
        self,
        name: str,
        pose: np.ndarray,
        lin_speed: float,
        ang_speed: float,
        spawn_z: float,
        hands_off: bool = True,
    ):
        """Observe one frame of telemetry for an object."""
        cfg = self.cfg
        if name not in self._spawn_z:
            self._spawn_z[name] = spawn_z

        # Early failure check
        if object_fallen_off(pose, cfg):
            self.failed = True
            self.failure_reason = f"{name} fell off table (z={pose[2]:.3f})"
            self.object_states[name] = SubtaskState.FAILED
            self.in_tray_streak[name] = 0
            return

        # Grasp metric
        if pose[2] > self._spawn_z[name] + cfg.lift_height:
            self.was_grasped[name] = True
            if self.object_states[name] == SubtaskState.SPAWNED:
                self.object_states[name] = SubtaskState.GRASPED

        # In-tray & settling streak
        in_tray = object_in_tray(pose, cfg)
        settled = lin_speed <= cfg.settle_lin_speed

        if in_tray:
            if settled and hands_off:
                self.in_tray_streak[name] += 1
                if self.in_tray_streak[name] >= cfg.settle_frames:
                    self.was_placed[name] = True
                    self.object_states[name] = SubtaskState.SETTLED
                else:
                    self.object_states[name] = SubtaskState.PLACED
            else:
                self.in_tray_streak[name] = 0
                self.object_states[name] = SubtaskState.TRANSPORTING
        else:
            self.in_tray_streak[name] = 0

    def object_placed(self, name: str) -> bool:
        """Object is verified inside tray and settled for settle_frames."""
        return self.in_tray_streak.get(name, 0) >= self.cfg.settle_frames

    def success(self) -> bool:
        """Complete task success: BOTH objects verified placed, settled, and no failure."""
        if self.failed:
            return False
        return all(self.object_placed(n) for n in self.REQUIRED_OBJECTS)


def reward(
    tcp_pos: np.ndarray,
    object_pose: np.ndarray,
    object_spawn: np.ndarray,
    tray_target: np.ndarray,
    cfg: TaskConfig,
    grasped: bool,
    placed: bool,
    fallen_off: bool = False,
) -> float:
    """IsaacLab-style dense shaping reward function (MDP formulation).

    R = r_reach + r_lift + r_place + r_upright + r_task + r_penalty
    """
    if fallen_off:
        return -10.0

    w_reach, w_lift, w_place, w_task = 1.0, 2.0, 3.0, 10.0

    # 1. Reaching reward: distance between TCP and object
    d_reach = float(np.linalg.norm(tcp_pos - object_pose[:3]))
    r_reach = 1.0 - np.tanh(5.0 * d_reach)

    # 2. Lifting reward: progress from spawn height up to lift_height
    lifted = max(0.0, float(object_pose[2] - object_spawn[2]))
    r_lift = w_lift * min(1.0, lifted / max(0.01, cfg.lift_height))

    # 3. Transport & Placing reward: distance to assigned tray slot
    d_place = float(np.linalg.norm(object_pose[:3] - tray_target[:3]))
    r_place = w_place * (1.0 - np.tanh(3.0 * d_place))

    # 4. Upright stability bonus
    if len(object_pose) >= 7:
        qx, qy = float(object_pose[3]), float(object_pose[4])
        up_z = 1.0 - 2.0 * (qx * qx + qy * qy)
        r_upright = 1.0 if up_z >= np.cos(np.deg2rad(cfg.max_tilt_deg)) else 0.0
    else:
        r_upright = 1.0

    # 5. Milestone completion rewards
    r_task = w_task if placed else 0.0
    r_grasp_bonus = 3.0 if grasped else 0.0

    return float(r_reach + r_lift + r_place + r_upright + r_task + r_grasp_bonus)
