"""Tabletop pick-place environment on kitless Newton (Newton 1.7 + SolverMuJoCo).

Scene (per the interview task): a tabletop holding a tray, a bowl and a cup;
a Franka FR3 arm (fixed base) must place BOTH the bowl and the cup into the
tray. The task/success definition lives in task.py; a scripted IK agent in
agent.py.

Kitless mapping note: IsaacLab 3.0 ships the same FR3 articulation
(``franka_emika_panda`` asset); here we load the identical URDF through
newton.ModelBuilder.add_urdf and drive it with SolverMuJoCo (MuJoCo-Warp on
CUDA) instead of PhysX, so the task definition and controller transfer.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
import warp as wp

import newton

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
FRANKA_URDF = os.path.join(ASSETS, "franka_emika_panda", "urdf", "fr3_franka_hand.urdf")
CUP_USD = os.path.join(ASSETS, "manipulation_objects", "cup", "model.usda")

# Body indices inside the FR3 + hand URDF articulation (verified by check_env).
HAND_BODY = 10  # fr3_hand
TCP_BODY = 11  # fr3_hand_tcp (massless fixed frame at the fingertips' center)
FINGER_BODIES = (12, 13)  # fr3_leftfinger, fr3_rightfinger
N_ARM_DOFS = 7

# PD gains for the FR3 arm + hand. Higher arm stiffness than the newton
# example (650/100) because our table is tall: objects are grasped with the
# arm extended, where gravity torque is large and position error scales 1/ke.
ARM_KE, ARM_KD = [2500.0] * 7, [300.0] * 7
FINGER_KE, FINGER_KD = [100.0] * 2, [20.0] * 2
ARM_EFFORT, ARM_ARMATURE = [90.0, 180.0, 90.0, 180.0, 90.0, 90.0, 90.0], [0.1] * 7
FINGER_EFFORT, FINGER_ARMATURE = [5.0] * 2, [0.5] * 2

# Fallback FR3 posture (from newton example_robot_panda_hydro); used only as the
# IK seed. The actual HOME_Q is solved once at startup so the hand hovers clear
# of the table (see solve_home_q()).
HOME_Q = [
    -3.6802115e-03, 2.3901723e-02, 3.6804110e-03, -2.3683236e00,
    -1.2918962e-04, 2.3922248e00, 7.8549200e-01, 0.04, 0.04,
]

# Home hover pose: TCP well above the tallest object (bowl top ~0.434) so the
# idle arm can never touch the spawned objects; palm facing down.
HOME_TCP_TARGET = wp.transform(
    wp.vec3(0.45, 0.0, 0.62),
    wp.mul(wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), np.pi), wp.quat_identity()),
)


def stripped_urdf_path() -> str:
    """Path to a visual-stripped copy of the FR3 URDF (cached on disk).

    The IK models only need kinematics; parsing 25 MB of visual DAE meshes
    three times per process wastes ~15 s.
    """
    out = FRANKA_URDF + ".novisual.urdf"
    if not os.path.exists(out):
        import xml.etree.ElementTree as ET

        tree = ET.parse(FRANKA_URDF)
        parents = {c: p for p in tree.iter() for c in p}
        for vis in list(tree.iter("visual")):
            parents[vis].remove(vis)
        tree.write(out)
    return out


def solve_home_q() -> list[float]:
    """Solve the FR3 home posture via newton's IK so the TCP hovers above the table.

    The example's hardcoded home hangs the hand at z~0.33 which intersects our
    table (top at 0.40); solving IK for a safe hover pose keeps the arm clear.
    Deterministic for a fixed seed pose, so the solution is cached on disk and
    verified by forward kinematics on every load.
    """
    cache = os.path.join(ASSETS, "franka_home_q.json")
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            q_np = np.asarray(json.load(f)["q"], dtype=np.float32)
        # verify against FK: the cached pose must still hit the hover target
        import newton.ik as ik  # noqa: F401

        builder = newton.ModelBuilder()
        builder.add_urdf(stripped_urdf_path(), floating=False, joint_damping_mode="passive")
        model = builder.finalize()
        state = model.state()
        newton.eval_fk(
            model,
            wp.array(q_np, dtype=wp.float32),
            wp.array(np.zeros_like(q_np), dtype=wp.float32),
            state,
        )
        tcp_pos = state.body_q.numpy()[TCP_BODY, :3]
        err = float(np.linalg.norm(tcp_pos - np.asarray(HOME_TCP_TARGET.p)))
        assert err < 0.01, f"cached home pose invalid (TCP error {err * 1000:.1f} mm)"
        print(f"[env] home IK loaded from cache, TCP error {err * 1000:.2f} mm")
        return q_np.tolist()

    import newton.ik as ik

    builder = newton.ModelBuilder()
    builder.add_urdf(stripped_urdf_path(), floating=False, joint_damping_mode="passive")
    model = builder.finalize()

    pos_obj = ik.IKObjectivePosition(
        link_index=TCP_BODY,
        link_offset=wp.vec3(0.0, 0.0, 0.0),
        target_positions=wp.array([wp.transform_get_translation(HOME_TCP_TARGET)], dtype=wp.vec3),
    )
    rot_q = wp.transform_get_rotation(HOME_TCP_TARGET)
    rot_obj = ik.IKObjectiveRotation(
        link_index=TCP_BODY,
        link_offset_rotation=wp.quat_identity(),
        target_rotations=wp.array([wp.vec4(rot_q[0], rot_q[1], rot_q[2], rot_q[3])], dtype=wp.vec4),
    )
    limits = ik.IKObjectiveJointLimit(
        joint_limit_lower=model.joint_limit_lower,
        joint_limit_upper=model.joint_limit_upper,
    )
    joint_q = wp.array(np.asarray(HOME_Q, dtype=np.float32).reshape(1, 9), dtype=wp.float32)
    solver = ik.IKSolver(
        model=model,
        n_problems=1,
        objectives=[pos_obj, rot_obj, limits],
        lambda_initial=0.1,
        jacobian_mode=ik.IKJacobianType.ANALYTIC,
    )
    solver.step(joint_q, joint_q, iterations=200)

    # verify convergence: forward-kinematics the solution and measure TCP error
    state = model.state()
    q_np = joint_q.numpy().reshape(-1)
    q_np[7:9] = 0.04  # fingers fully open at home
    qd_np = np.zeros_like(q_np)
    newton.eval_fk(model, wp.array(q_np, dtype=wp.float32), wp.array(qd_np, dtype=wp.float32), state)
    tcp_pos = state.body_q.numpy()[TCP_BODY, :3]
    target_pos = np.asarray(HOME_TCP_TARGET.p)
    err = float(np.linalg.norm(tcp_pos - target_pos))
    assert err < 0.01, f"home IK did not converge: TCP error {err * 1000:.1f} mm"
    with open(cache, "w", encoding="utf-8") as f:
        json.dump({"q": [round(float(v), 6) for v in q_np]}, f)
    print(f"[env] home IK solved (cached), TCP error {err * 1000:.2f} mm")
    return q_np.tolist()


# Contact material: newton converts (ke, kd) to mujoco solref via
# timeconst = 2/(kd*width), dampratio = kd/2*sqrt(d_r/ke) with default
# solimp width=0.001, d_r=0.9. The library default (ke=2.5e3, kd=100) yields
# an ultra-soft contact that lets light objects rest ~14 mm INSIDE the table;
# ke=2.5e8/kd=5e4 gives timeconst=0.04 s (> 2*dt) and dampratio~1.5, so
# objects rest within ~0.1 mm of the true surface AND settle quietly
# (slightly under-damped contacts keep micro-bouncing, which trips
# velocity-based success criteria).
CONTACT_KE = 2.5e8
CONTACT_KD = 5.0e4


def contact_cfg(**kwargs) -> newton.ModelBuilder.ShapeConfig:
    return newton.ModelBuilder.ShapeConfig(ke=CONTACT_KE, kd=CONTACT_KD, **kwargs)


@dataclass
class SceneConfig:
    """Metric dimensions of the tabletop scene (z-up, arm base at origin)."""

    table_top_z: float = 0.40
    table_center: tuple = (0.65, 0.0, 0.38)
    table_half: tuple = (0.35, 0.50, 0.02)

    # Rectangular tray assembled from 5 boxes (base + 4 walls).
    tray_center_xy: tuple = (0.62, 0.02)
    tray_base_half: tuple = (0.145, 0.105, 0.004)
    tray_wall_half_z: float = 0.0225  # wall height 4.5 cm
    tray_thickness: float = 0.01

    # Procedural lathe bowl (open top, closed bottom). Slightly tapered
    # (narrower at the rim) so the parallel fingers WEDGE it while lifting —
    # form closure makes the grasp self-tightening and cancels the servo's
    # ~1 cm lateral drift. Collision is a single cylinder at the rim radius;
    # everything sized to fit the Franka hand (inner faces span ~6.9 cm open).
    bowl_height: float = 0.050
    bowl_base_r: float = 0.032
    bowl_rim_r: float = 0.0275
    bowl_density: float = 1500.0

    cup_density: float = 700.0

    # Randomized spawn region for the two objects on the table. Kept inside
    # the arm's comfortable workspace (the extended far side of the table
    # triggers a servo pathology: the descent stalls and the gripper wedges
    # open with zero contacts); the tray sits to the +x side.
    spawn_x_range: tuple = (0.40, 0.47)
    spawn_y_range: tuple = (-0.08, 0.28)
    spawn_min_separation: float = 0.12
    spawn_min_tray_dist: float = 0.15

    # Simulation cadence: 60 Hz control frames, 10 substeps of 1/600 s.
    control_hz: int = 60
    substeps: int = 10

    @property
    def dt_frame(self) -> float:
        return 1.0 / self.control_hz

    @property
    def dt_substep(self) -> float:
        return self.dt_frame / self.substeps


def lathe_mesh(profile: list[tuple[float, float]], segments: int = 24) -> newton.Mesh:
    """Revolve a closed (r, z) profile around the z axis into a triangle mesh."""
    n = len(profile)
    verts = []
    for seg in range(segments):
        a = 2.0 * np.pi * seg / segments
        c, s = np.cos(a), np.sin(a)
        for r, z in profile:
            verts.append((r * c, r * s, z))
    verts = np.asarray(verts, dtype=np.float32)
    idx = []
    for seg in range(segments):
        seg2 = (seg + 1) % segments
        for i in range(n - 1):
            a0, a1 = seg * n + i, seg * n + i + 1
            b0, b1 = seg2 * n + i, seg2 * n + i + 1
            idx += [a0, b0, b1, a0, b1, a1]
    return newton.Mesh(verts, np.asarray(idx, dtype=np.int32))


def bowl_profile(cfg: SceneConfig) -> list[tuple[float, float]]:
    """Closed (r, z) profile of the tapered bowl: bottom -> outer wall -> rim -> inner wall."""
    rb, rt, h = cfg.bowl_base_r, cfg.bowl_rim_r, cfg.bowl_height
    ri_b, ri_t = rt - 0.006, rt - 0.0065  # ~6 mm wall
    return [
        (0.0, 0.0),
        (rb * 0.96, 0.0),
        (rb, 0.005),
        (rt, h * 0.92),
        (rt, h),
        (ri_t, h * 0.97),
        (ri_b, 0.007),
        (0.0, 0.005),
    ]


def tray_wall_boxes(cfg: SceneConfig):
    """Yield (center, half_extents) for the tray's 5 constituent boxes."""
    bx, by = cfg.tray_center_xy
    hx, hy, hz = cfg.tray_base_half
    t = cfg.tray_thickness / 2.0
    z0 = cfg.table_top_z
    # base
    yield (bx, by, z0 + hz), (hx, hy, hz)
    zw = z0 + 2 * hz + cfg.tray_wall_half_z
    # +y / -y walls
    yield (bx, by + hy - t, zw), (hx, t, cfg.tray_wall_half_z)
    yield (bx, by - hy + t, zw), (hx, t, cfg.tray_wall_half_z)
    # +x / -x walls
    yield (bx + hx - t, by, zw), (t, hy - 2 * t, cfg.tray_wall_half_z)
    yield (bx - hx + t, by, zw), (t, hy - 2 * t, cfg.tray_wall_half_z)


def load_cup_mesh(cfg: SceneConfig) -> newton.Mesh:
    """Load the vendored Newton 'cup' asset, baking its USD scale into vertices."""
    from pxr import Usd

    stage = Usd.Stage.Open(CUP_USD)
    prim = stage.GetPrimAtPath("/root/Model/Model")
    mesh = newton.usd.get_mesh(prim, load_normals=True, face_varying_normal_conversion="vertex_splitting")
    scale = np.asarray(newton.usd.get_scale(stage.GetPrimAtPath("/root/Model")), dtype=np.float32)
    if not np.allclose(scale, 1.0):
        mesh = mesh.copy(vertices=mesh.vertices * scale, recompute_inertia=True)
    return mesh


class PickPlaceEnv:
    """Single-world tabletop pick-place env: FR3 arm + table + tray + bowl + cup.

    Control contract: every call to :meth:`step` applies one 60 Hz control
    frame (``joint_target`` = 7 arm position targets + 2 finger targets)
    and advances ``cfg.substeps`` physics substeps.
    """

    def __init__(self, cfg: SceneConfig | None = None):
        self.cfg = cfg or SceneConfig()
        # solve the hover home once; the module-level HOME_Q stays as the IK seed
        self.home_q = solve_home_q()
        self._build()
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.sim_time = 0.0
        self._body_q = self.state_0.body_q.numpy()
        self._body_qd = self.state_0.body_qd.numpy()
        self._capture()

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    def _build(self):
        cfg = self.cfg
        builder = newton.ModelBuilder()
        builder.default_shape_cfg = contact_cfg()

        # --- robot ------------------------------------------------------
        builder.add_urdf(FRANKA_URDF, floating=False, enable_self_collisions=False, joint_damping_mode="passive")
        builder.joint_target_ke[:7], builder.joint_target_kd[:7] = ARM_KE, ARM_KD
        builder.joint_target_ke[7:9], builder.joint_target_kd[7:9] = FINGER_KE, FINGER_KD
        builder.joint_effort_limit[:7], builder.joint_armature[:7] = ARM_EFFORT, ARM_ARMATURE
        builder.joint_effort_limit[7:9], builder.joint_armature[7:9] = FINGER_EFFORT, FINGER_ARMATURE
        builder.joint_q[:9] = list(self.home_q)
        builder.joint_target_q[:9] = list(self.home_q)
        # high-friction fingertip pads for a stable grasp
        finger_shapes = [
            s for s, b in enumerate(builder.shape_body) if b in FINGER_BODIES
        ]
        for s in finger_shapes:
            builder.shape_material_mu[s] = 1.5

        # --- static table + tray -----------------------------------------
        table_cfg = contact_cfg(density=700.0, mu=0.6)
        builder.add_shape_box(
            body=-1,
            hx=cfg.table_half[0],
            hy=cfg.table_half[1],
            hz=cfg.table_half[2],
            xform=wp.transform(wp.vec3(*cfg.table_center), wp.quat_identity()),
            cfg=table_cfg,
        )
        for center, half in tray_wall_boxes(cfg):
            builder.add_shape_box(
                body=-1,
                hx=half[0],
                hy=half[1],
                hz=half[2],
                xform=wp.transform(wp.vec3(*center), wp.quat_identity()),
                cfg=table_cfg,
            )

        # --- bowl: cylinder collision + lathe visual mesh -------------------
        # A concave lathe mesh hull-converted by approximate_meshes welds away
        # the 5 mm bottom ring and rolls off the table; a cylinder primitive
        # gives a flat, stable base while the lathe stays as visual-only mesh.
        bowl_mesh = lathe_mesh(bowl_profile(cfg))
        half_h = cfg.bowl_height / 2.0
        self.bowl_body = builder.add_body(
            xform=wp.transform(wp.vec3(0.45, 0.0, cfg.table_top_z + 0.002), wp.quat_identity()),
            label="bowl",
        )
        builder.add_shape_cylinder(
            body=self.bowl_body,
            radius=cfg.bowl_rim_r,  # collision at the rim radius: the fingers
            # grasp the upper wall, where the taper wedges the bowl on lift
            half_height=half_h,
            xform=wp.transform(wp.vec3(0.0, 0.0, half_h), wp.quat_identity()),
            cfg=contact_cfg(density=cfg.bowl_density, mu=0.8),
        )
        builder.add_shape_mesh(
            body=self.bowl_body,
            mesh=bowl_mesh,
            cfg=contact_cfg(has_shape_collision=False),  # visual only
        )

        # --- cup (vendored newton asset) -----------------------------------
        cup_mesh = load_cup_mesh(cfg)
        self.cup_zmin = float(cup_mesh.vertices[:, 2].min())
        self.cup_body = builder.add_body(
            xform=wp.transform(wp.vec3(0.45, 0.18, cfg.table_top_z - self.cup_zmin + 0.002), wp.quat_identity()),
            label="cup",
        )
        builder.add_shape_mesh(
            body=self.cup_body,
            mesh=cup_mesh,
            cfg=contact_cfg(density=cfg.cup_density, mu=0.8),
        )

        # convex-hull collision ONLY for true URDF collision meshes (link STLs,
        # hand STL, cup asset); visual-only meshes must NOT be hulled — the
        # hulls of the chunky visual meshes become fat phantom colliders that
        # wedge the gripper open (panda_hydro passes shape_indices for the
        # same reason). Tray concavity comes from its 5 boxes.
        collide_mesh_shapes = [
            s
            for s in range(builder.shape_count)
            if builder.shape_type[s] == newton.GeoType.MESH
            and (builder.shape_flags[s] & newton.ShapeFlags.COLLIDE_SHAPES)
        ]
        builder.approximate_meshes(
            method="convex_hull", shape_indices=collide_mesh_shapes, keep_visual_shapes=True
        )

        # hull conversion may weld/trim vertices: measure the cup's real
        # collision-hull bottom so reset() can spawn it exactly on surfaces
        self.bowl_zmin = 0.0  # cylinder collision bottom sits at the body origin
        for s, b in enumerate(builder.shape_body):
            if b == self.cup_body and builder.shape_type[s] == newton.GeoType.CONVEX_MESH:
                self.cup_zmin = float(builder.shape_source[s].vertices[:, 2].min())

        builder.add_ground_plane()
        self.model = builder.finalize()

        # spatial indices used by the task / agent
        self.body_names = list(builder.body_label)
        self.hand_idx = HAND_BODY
        self.tcp_idx = TCP_BODY
        self.finger_idx = FINGER_BODIES
        self.bowl_idx = self.bowl_body
        self.cup_idx = self.cup_body
        self.object_bodies = {"bowl": self.bowl_body, "cup": self.cup_body}

        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            iterations=15,
            ls_iterations=20,
            njmax=512,
            nconmax=256,
        )

        # coord offset of each object's FREE joint inside state.joint_q — free
        # bodies are authoritative in joint coordinates (body_q is derived by
        # FK), so reset() must write these, not body_q.
        self.free_joint_offsets: dict[str, int] = {}
        dof_counts = {0: 1, 1: 1, 2: 4, 3: 0, 4: 7}  # PRISMATIC, REVOLUTE, BALL, FIXED, FREE
        off = 0
        types = self.model.joint_type.numpy()
        labels = self.model.joint_label
        for j, (t, lbl) in enumerate(zip(types, labels)):
            for name in self.object_bodies:
                if lbl.startswith(f"{name}_free"):
                    self.free_joint_offsets[name] = off
            off += dof_counts.get(int(t), 7)
        assert set(self.free_joint_offsets) == set(self.object_bodies), (
            f"free joints not found for all objects: {self.free_joint_offsets}"
        )

    # ------------------------------------------------------------------ #
    # episode lifecycle
    # ------------------------------------------------------------------ #
    def sample_object_poses(self, rng: np.random.Generator) -> dict[str, wp.transform]:
        """Randomized spawn poses for bowl and cup (mutually separated, clear of the tray)."""
        cfg = self.cfg
        tx, ty = cfg.tray_center_xy
        # the tray is RECTANGULAR: reject spawns whose bounding circle
        # intersects the tray rectangle (a distance-to-center check lets
        # objects spawn intersecting the -x wall and get ejected violently)
        r_obj, margin = 0.035, 0.01
        hx, hy = cfg.tray_base_half[0] + r_obj + margin, cfg.tray_base_half[1] + r_obj + margin
        poses: dict[str, wp.transform] = {}
        taken: list[tuple[float, float]] = []
        for name in ("bowl", "cup"):
            for _ in range(200):
                x = rng.uniform(*cfg.spawn_x_range)
                y = rng.uniform(*cfg.spawn_y_range)
                ok = all(np.hypot(x - px, y - py) >= cfg.spawn_min_separation for px, py in taken)
                ok &= not (abs(x - tx) < hx and abs(y - ty) < hy)
                if ok:
                    break
            else:
                raise RuntimeError("could not sample separated object poses")
            taken.append((x, y))
            yaw = rng.uniform(-np.pi, np.pi)
            poses[name] = wp.transform(wp.vec3(x, y, 0.0), wp.quat_from_axis_angle(wp.vec3(0, 0, 1), yaw))
        return poses

    def reset(self, seed: int) -> dict[str, wp.transform]:
        """Reset the arm to HOME, re-spawn objects at randomized poses, clear solver warm-start.

        Object poses are written into their FREE-joint coordinates
        (``state.joint_q``) — the authoritative reduced-coordinate state — and
        body_q is then rebuilt with forward kinematics.
        """
        cfg = self.cfg
        self.solver.reset(self.state_0)
        poses = self.sample_object_poses(np.random.default_rng(seed))

        joint_q = self.state_0.joint_q.numpy()
        for name, xf in poses.items():
            b = self.object_bodies[name]
            # spawn with the collision shape exactly 2 mm above the surface
            z = cfg.table_top_z + 0.002  # bowl cylinder bottom sits at the body origin
            if name == "cup":
                z = cfg.table_top_z - self.cup_zmin + 0.002  # mesh origin is not its base
            off = self.free_joint_offsets[name]
            rot = wp.transform_get_rotation(xf)
            joint_q[off : off + 3] = [xf.p[0], xf.p[1], z]
            joint_q[off + 3 : off + 7] = [rot[0], rot[1], rot[2], rot[3]]
        self.state_0.joint_q.assign(joint_q)

        # rebuild body_q from the updated joint coordinates
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        # mirror into the second state buffer (the next step's output buffer)
        self.state_1.joint_q.assign(self.state_0.joint_q.numpy())
        self.state_1.joint_qd.assign(self.state_0.joint_qd.numpy())

        self._body_q = self.state_0.body_q.numpy()
        self._body_qd = self.state_0.body_qd.numpy()
        self._joint_q = self.state_0.joint_q.numpy()
        self._joint_qd = self.state_0.joint_qd.numpy()

        self.sim_time = 0.0
        return poses

    # ------------------------------------------------------------------ #
    # stepping
    # ------------------------------------------------------------------ #
    def _simulate(self):
        for _ in range(self.cfg.substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, None, self.cfg.dt_substep)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _capture(self):
        """Record the substep loop into a CUDA graph to skip per-launch overhead."""
        self._graph = None
        try:
            with wp.ScopedCapture() as capture:
                self._simulate()
            self._graph = capture.graph
            print("[env] CUDA graph capture enabled")
        except Exception as e:  # noqa: BLE001
            print(f"[env] CUDA graph capture unavailable ({e!r}); using eager stepping")

    def step(self, joint_target: np.ndarray):
        """Apply one 60 Hz control frame; joint_target = [7 arm qpos targets, 2 finger targets]."""
        assert joint_target.shape == (9,), f"joint_target must have shape (9,), got {joint_target.shape}"
        self.control.joint_target_q.assign(np.asarray(joint_target, dtype=np.float32))
        if self._graph is not None:
            wp.capture_launch(self._graph)
        else:
            self._simulate()
        self.sim_time += self.cfg.dt_frame
        # one GPU->CPU read per frame, shared by the agent / judge / clamp
        # (each extra .numpy() forces a device sync worth ~3-5 ms)
        self._body_q = self.state_0.body_q.numpy()
        self._body_qd = self.state_0.body_qd.numpy()
        self._joint_q = self.state_0.joint_q.numpy()
        self._joint_qd = self.state_0.joint_qd.numpy()
        self._clamp_object_velocities()

    def _clamp_object_velocities(self):
        """Safety clamp on the free objects' velocities.

        Rare contact-solver spikes (e.g. a wedged object squirting out of the
        gripper at release) can launch a lightweight body through the floor;
        clamping keeps the episode recoverable. Healthy motion never comes
        close to these limits.
        """
        MAX_LIN, MAX_ANG = 1.5, 12.0
        qd = self._body_qd
        changed = False
        for b in self.object_bodies.values():
            lin = np.linalg.norm(qd[b, :3])
            ang = np.linalg.norm(qd[b, 3:])
            if lin > MAX_LIN:
                qd[b, :3] *= MAX_LIN / lin
                changed = True
            if ang > MAX_ANG:
                qd[b, 3:] *= MAX_ANG / ang
                changed = True
        if changed:
            self.state_0.body_qd.assign(qd)

    def step_frames(self, joint_target_fn, n_frames: int):
        for _ in range(n_frames):
            self.step(joint_target_fn(self))

    # ------------------------------------------------------------------ #
    # observation helpers
    # ------------------------------------------------------------------ #
    def body_poses(self) -> np.ndarray:
        return self._body_q

    def body_vels(self) -> np.ndarray:
        return self._body_qd

    def arm_joint_q(self) -> np.ndarray:
        return self._joint_q[:7]

    def arm_joint_qd(self) -> np.ndarray:
        return self._joint_qd[:7]

    def joint_q(self) -> np.ndarray:
        return self._joint_q.copy()

    def joint_qd(self) -> np.ndarray:
        return self._joint_qd.copy()

    def tcp_pose(self) -> tuple[np.ndarray, np.ndarray]:
        q = self.body_poses()
        return q[self.tcp_idx, :3], q[self.tcp_idx, 3:]

    def finger_gap(self) -> float:
        q = self.body_poses()
        p_l, p_r = q[self.finger_idx[0], :3], q[self.finger_idx[1], :3]
        return float(np.linalg.norm(p_l - p_r))


if __name__ == "__main__":
    import time

    wp.init()
    env = PickPlaceEnv()
    print("bodies:", {k: v for k, v in env.object_bodies.items()})
    print("body count:", env.model.body_count, "shape count:", env.model.shape_count)
    t0 = time.time()
    poses = env.reset(seed=0)
    print("spawn poses:", {k: (round(float(v.p[0]), 3), round(float(v.p[1]), 3)) for k, v in poses.items()})
    hold = np.asarray(env.home_q, dtype=np.float32)
    z_traj = {"bowl": [], "cup": []}
    for _ in range(120):  # 2 s settle
        env.step(hold)
        q = env.body_poses()
        for name, b in env.object_bodies.items():
            z_traj[name].append(float(q[b, 2]))
    print(f"120 frames in {time.time() - t0:.1f}s ({(time.time() - t0) / 120 * 1000:.0f} ms/frame)")
    for name, zs in z_traj.items():
        drift = abs(zs[-1] - zs[-10])
        print(f"{name}: z_final={zs[-1]:.4f} drift={drift:.5f} vel_norm={np.linalg.norm(env.body_vels()[env.object_bodies[name]]) if False else 'n/a'}")
    q = env.body_poses()
    assert all(np.isfinite(q).ravel()), "NaN in body_q"
    hand_pos = q[env.hand_idx, :3]
    print("hand pos:", np.round(hand_pos, 3).tolist())
    print("tcp pos:", np.round(q[env.tcp_idx, :3], 3).tolist())
    print("finger gap:", round(env.finger_gap(), 4))
    print("ENV SMOKE TEST PASS")
