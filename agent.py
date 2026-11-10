"""Scripted pick-and-place agent: joint-space PTP waypoints + newton IK.

Design notes (why PTP, not Cartesian tracking):
  Tracking a straight-line Cartesian descent with position servos stalls a few
  cm above the target on this scene (the extended-arm pose needs large
  shoulder torque, and slow oblique descents get stuck at a servo equilibrium
  even with tcp feedback). A point-to-point policy — command the IK solution
  of each waypoint and let the PD servos swing through free space — reaches
  every waypoint reliably (verified by the static servo test), which is also
  how industrial pick-and-place programs are written. Object poses are read
  like a perception module would output them; swapping in an RL/MLP policy
  only replaces `act()`.

Phases per object (bowl first, then cup):
  hover above object -> descend to grasp -> close gripper -> lift ->
  traverse above the tray slot -> lower -> release -> retract -> (next)
"""

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.ik as ik

from env import (
    FRANKA_URDF,
    TCP_BODY,
    SceneConfig,
    solve_home_q,
    stripped_urdf_path,
)

# tcp heights (world z) per phase
HOVER_Z = 0.55
CARRY_Z = 0.60  # carry height: clears the other object (cup top ~0.476) with margin
# release heights: lower the object until it RESTS on the tray floor while
# still gripped, then relax+open — the tray (floor + 4 walls) contains any
# stored squeeze energy, so nothing can squirt out
PLACE_Z = {"bowl": 0.44, "cup": 0.455}
RETRACT_Z = 0.56
HOME_Z = 0.62

# tcp grasp heights per object. The servos sink until the fingertips touch the
# table (hard floor tcp~0.418), so grasp heights must keep the pads clear of
# the table: tcp 0.45 -> pads span ~0.43-0.45 (upper bowl wall / lower cup wall).
GRASP_Z = {"bowl": 0.45, "cup": 0.45}
# steady-state servo sag at the grasp poses (measured by z-scan: -7..+11 mm)
GRASP_SAG_BIAS = 0.005

# tray slots (tray center 0.62, 0.02; inner half x 0.135 -> slots at +-6 cm)
SLOT = {"bowl": (0.56, 0.02), "cup": (0.68, 0.02)}

GRIP_OPEN = 0.04  # finger target per side; gap = 2*q (8 cm fully open)
# closed targets per object: firm enough to hold through the carry swing
GRIP_CLOSED = {"bowl": 0.0125, "cup": 0.010}
# pre-release squeeze: relaxing to this BEFORE opening drains the stored
# squeeze energy — a tight pinch on a light object squirts it out like a
# watermelon seed when the fingers open (newton's panda_hydro notes the
# same effect with compliant finger drives)
GRIP_RELEASE = 0.027
# finger pre-shape during the bowl descent: straddles the bowl (6.2 cm) with
# clearance so the swing cannot press on the rim
GRIP_PRESHAPE = 0.035

# palm-down orientation, constant for the whole task (Rx(pi))
PALM_DOWN_QUAT = wp.mul(wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), np.pi), wp.quat_identity())


class FrankaIK:
    """Analytic IK for the vendored FR3+hand URDF using newton.ik."""

    RESEED_ERR = 0.03

    def __init__(self, init_joint_q: np.ndarray):
        self.home_q = np.asarray(init_joint_q, dtype=np.float32).reshape(-1)
        builder = newton.ModelBuilder()
        # IK only needs kinematics: use the visual-stripped URDF (3x faster build)
        builder.add_urdf(stripped_urdf_path(), floating=False, joint_damping_mode="passive")
        self.model = builder.finalize()
        self.pos_obj = ik.IKObjectivePosition(
            link_index=TCP_BODY,
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3),
        )
        q = PALM_DOWN_QUAT
        self.rot_obj = ik.IKObjectiveRotation(
            link_index=TCP_BODY,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([wp.vec4(q[0], q[1], q[2], q[3])], dtype=wp.vec4),
        )
        self.limits_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.model.joint_limit_lower,
            joint_limit_upper=self.model.joint_limit_upper,
        )
        self.joint_q = wp.array(self.home_q.reshape(1, -1).copy(), dtype=wp.float32)
        self.solver = ik.IKSolver(
            model=self.model,
            n_problems=1,
            objectives=[self.pos_obj, self.rot_obj, self.limits_obj],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

    def solve(self, pos, iterations: int = 400, cold: bool = True) -> np.ndarray:
        """Solve IK for a tcp position; cold-start from home by default (robust)."""
        if cold:
            self.joint_q.assign(self.home_q.reshape(1, -1))
        self.pos_obj.set_target_position(0, wp.vec3(*[float(v) for v in pos]))
        self.solver.step(self.joint_q, self.joint_q, iterations=iterations)
        return self.joint_q.numpy().reshape(-1).copy()


def quintic(t: float) -> float:
    """Smoothstep time scaling on [0, 1]."""
    t = min(1.0, max(0.0, t))
    return t * t * t * (10.0 - 15.0 * t + 6.0 * t * t)


class PickPlaceAgent:
    """Joint-space PTP policy over IK waypoints; PD servos track the targets."""

    def __init__(self, env, cfg: SceneConfig | None = None):
        self.env = env
        self.cfg = cfg or env.cfg
        self.ik = FrankaIK(np.asarray(env.home_q, dtype=np.float32))

    # ------------------------------------------------------------------ #
    def _plan(self, object_poses: dict[str, np.ndarray]) -> list[dict]:
        """Waypoint list (tcp positions + durations + gripper targets)."""
        wps: list[dict] = [
            {"name": "home", "pos": (0.45, 0.0, HOME_Z), "dur": 0.8, "grip": GRIP_OPEN},
        ]
        for name in ("bowl", "cup"):
            ox, oy = float(object_poses[name][0]), float(object_poses[name][1])
            gz = GRASP_Z[name] - GRASP_SAG_BIAS
            sx, sy = SLOT[name]
            if name == "bowl":
                # both objects spawn in the comfortable workspace; use the
                # proven hover -> vertical-descent approach for both
                pass
            wps += [
                {"name": f"{name}:hover", "pos": (ox, oy, HOVER_Z), "dur": 1.0, "grip": GRIP_OPEN},
                {"name": f"{name}:descend", "kind": "aim_descend", "obj": name, "dur": 1.2, "grip": GRIP_OPEN},
                {"name": f"{name}:grasp", "kind": "hold", "dur": 1.6, "grip": GRIP_CLOSED[name]},
            ]
            wps += [
                {"name": f"{name}:lift", "pos": (ox, oy, HOVER_Z), "dur": 1.0, "grip": GRIP_CLOSED[name]},
                {"name": f"{name}:move", "pos": (sx, sy, CARRY_Z), "dur": 1.8, "grip": GRIP_CLOSED[name]},
                {"name": f"{name}:lower", "pos": (sx, sy, PLACE_Z[name]), "dur": 1.2, "grip": GRIP_CLOSED[name]},
                # two-stage release: ease the squeeze first, then open
                {"name": f"{name}:release", "kind": "hold", "dur": 0.9, "grip": GRIP_RELEASE},
                {"name": f"{name}:open", "kind": "hold", "dur": 0.7, "grip": GRIP_OPEN},
                {"name": f"{name}:retract", "pos": (sx, sy, RETRACT_Z), "dur": 0.9, "grip": GRIP_OPEN},
            ]
        wps.append({"name": "settle", "pos": (0.45, 0.0, HOME_Z), "dur": 3.5, "grip": GRIP_OPEN})
        return wps

    # ------------------------------------------------------------------ #
    def reset_episode(self, object_poses: dict[str, np.ndarray]):
        """Start a new episode: solve all waypoint IKs once, rewind the clock."""
        self.plan = self._plan(object_poses)
        self.object_xy = {
            name: (float(xf[0]), float(xf[1])) for name, xf in object_poses.items()
        }
        # joint-space endpoints for every waypoint (cold IK, deterministic);
        # aim/hold phases are resolved at phase entry / from the previous phase
        self.plan_q: list[np.ndarray | None] = []
        for wp_ in self.plan:
            if wp_.get("kind") in ("aim_descend", "aim_ptp", "hold"):
                self.plan_q.append(None)
            else:
                self.plan_q.append(self.ik.solve(wp_["pos"], iterations=400, cold=True)[:7])
        self.phase_idx = 0
        self.phase_time = 0.0
        self.frame = 0
        self.q_start = np.asarray(self.env.joint_q()[:7], dtype=np.float64)
        self.current_grip = float(self.plan[0]["grip"])

    def act(self) -> np.ndarray:
        """One control frame: joint-space PTP toward the current waypoint."""
        dt = self.env.cfg.dt_frame
        self.phase_time += dt
        self.frame += 1

        cur = self.plan[self.phase_idx]
        idx = self.phase_idx

        # phase-entry hooks
        if cur.get("kind") in ("aim_descend", "aim_ptp") and self.plan_q[idx] is None:
            # re-aim the approach at the object's CURRENT pose (it may have
            # been nudged by earlier motion); solve once, then lerp there
            name = cur["obj"]
            b = self.env.object_bodies[name]
            p = self.env.body_poses()[b]
            gz = GRASP_Z[name] - GRASP_SAG_BIAS
            self.plan_q[idx] = self.ik.solve((p[0], p[1], gz), iterations=400, cold=True)[:7]

        if cur.get("kind") == "hold":
            q_end = self.q_start.copy()  # hold position, only actuate the gripper
        else:
            q_end = self.plan_q[idx]
        t = quintic(self.phase_time / cur["dur"])
        q_cmd = self.q_start * (1.0 - t) + q_end * t
        nxt_grip = float(cur["grip"])
        self.current_grip = self.current_grip * (1.0 - t) + nxt_grip * t

        if self.phase_time >= cur["dur"]:
            if self.phase_idx + 1 < len(self.plan):
                self.phase_idx += 1
                self.phase_time = 0.0
                self.q_start = q_end.copy()
                self.current_grip = float(cur["grip"])

        target = np.concatenate([q_cmd, [self.current_grip, self.current_grip]]).astype(np.float32)
        return target

    @property
    def phase_name(self) -> str:
        return self.plan[self.phase_idx]["name"]

    def finished(self) -> bool:
        return self.phase_idx >= len(self.plan) - 1 and self.phase_time >= self.plan[-1]["dur"]
