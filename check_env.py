"""Gate check 1/3: verify the kitless Newton stack end-to-end on CUDA.

Checks (exit 0 == pass):
  1. newton / warp / mujoco import; a CUDA device is visible to warp.
  2. A rigid-body world (ground + boxes) steps on GPU with SolverMuJoCo;
     boxes settle on the ground, no NaN.
  3. The vendored FR3 + Franka Hand URDF parses (7 arm joints + 2 finger
     joints) and its meshes resolve from the vendored package layout.
  4. The vendored cup USD loads as a mesh body.
  5. The franka model itself steps on GPU with PD joint targets.

Run: .venv/Scripts/python.exe newton_pick_place/check_env.py
"""

import os
import sys
import time

import numpy as np
import warp as wp

import newton

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
FRANKA_URDF = os.path.join(ASSETS, "franka_emika_panda", "urdf", "fr3_franka_hand.urdf")
CUP_USD = os.path.join(ASSETS, "manipulation_objects", "cup", "model.usda")

# PD gains for the FR3 arm + hand, proven by newton's own panda pick-place example.
ARM_KE, ARM_KD = [650.0] * 7, [100.0] * 7
FINGER_KE, FINGER_KD = [100.0] * 2, [20.0] * 2
ARM_EFFORT, ARM_ARMATURE = [80.0] * 7, [0.1] * 7
FINGER_EFFORT, FINGER_ARMATURE = [5.0] * 2, [0.5] * 2

# FR3 home posture (from newton example_robot_panda_hydro).
HOME_Q = [
    -3.6802115e-03, 2.3901723e-02, 3.6804110e-03, -2.3683236e00,
    -1.2918962e-04, 2.3922248e00, 7.8549200e-01, 0.05, 0.05,
]


def main() -> int:
    report: dict = {}
    t0 = time.time()

    # --- 1. versions + CUDA -------------------------------------------------
    import mujoco

    print(f"[1] newton {newton.__version__} | warp {wp.__version__} | mujoco {mujoco.__version__}")
    wp.init()
    devices = [str(d) for d in wp.get_devices()]
    assert any("cuda" in d for d in devices), f"no CUDA device visible to warp: {devices}"
    report["devices"] = devices
    print(f"    devices: {devices}")

    # --- 2. rigid world steps on GPU ---------------------------------------
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    for i in range(3):
        b = builder.add_body(
            xform=wp.transform(wp.vec3(0.15 * i, 0.0, 0.4), wp.quat_identity()),
            label=f"box{i}",
        )
        builder.add_shape_box(body=b, hx=0.03, hy=0.03, hz=0.03)
    model = builder.finalize()
    s0, s1 = model.state(), model.state()
    control = model.control()
    solver = newton.solvers.SolverMuJoCo(model, iterations=10, ls_iterations=10, njmax=128, nconmax=64)
    for _ in range(240):
        s0.clear_forces()
        solver.step(s0, s1, control, None, 1.0 / 240)
        s0, s1 = s1, s0
    wp.synchronize()
    body_q = s0.body_q.numpy()
    assert np.all(np.isfinite(body_q)), "NaN in body_q after GPU steps"
    z = body_q[:, 2]
    assert np.all((z > 0.02) & (z < 0.06)), f"boxes did not settle on ground: {z}"
    report["boxes_rest_z"] = [round(float(v), 4) for v in z]
    print(f"[2] GPU rigid world OK; boxes settled at z={report['boxes_rest_z']}")

    # --- 3. vendored FR3 URDF ----------------------------------------------
    assert os.path.isfile(FRANKA_URDF), f"missing {FRANKA_URDF}"
    fb = newton.ModelBuilder()
    fb.add_urdf(FRANKA_URDF, floating=False, joint_damping_mode="passive")
    n_joints = fb.joint_count
    n_bodies = fb.body_count
    # 14 joints = 7 arm revolute + 2 finger prismatic + 4 fixed (incl. hand_tcp frame);
    # actuated DOFs = 9. Bodies include massless fixed frames (fr3/base, fr3_link8, fr3_hand_tcp).
    assert n_joints == 14, f"expected 14 joints (7 arm + 2 finger + 4 fixed), got {n_joints}"
    assert n_bodies == 14, f"expected 14 bodies, got {n_bodies}"
    n_coords = len(fb.joint_q)
    assert n_coords == 9, f"expected 9 DOF coords (7 arm + 2 fingers), got {n_coords}"
    assert "fr3/fr3_hand_tcp" in fb.body_label, "TCP frame body missing"
    mesh_shapes = sum(1 for t in fb.shape_type if t == newton.GeoType.MESH)
    assert mesh_shapes > 0, "URDF meshes did not resolve"
    report["urdf"] = {
        "joints": n_joints,
        "bodies": n_bodies,
        "dof_coords": n_coords,
        "mesh_shapes": mesh_shapes,
    }
    print(
        f"[3] FR3+hand URDF OK: {n_joints} joints, {n_coords} DOF coords, "
        f"{n_bodies} bodies, {mesh_shapes} mesh shapes"
    )

    # --- 4. vendored cup USD ------------------------------------------------
    assert os.path.isfile(CUP_USD), f"missing {CUP_USD}"
    from pxr import Usd

    stage = Usd.Stage.Open(CUP_USD)
    cup_prim = stage.GetPrimAtPath("/root/Model/Model")
    cup_mesh = newton.usd.get_mesh(cup_prim, load_normals=True, face_varying_normal_conversion="vertex_splitting")
    scale = np.asarray(newton.usd.get_scale(stage.GetPrimAtPath("/root/Model")), dtype=np.float32)
    if not np.allclose(scale, 1.0):
        cup_mesh = cup_mesh.copy(vertices=cup_mesh.vertices * scale, recompute_inertia=True)
    cb = newton.ModelBuilder()
    cup_body = cb.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.2), wp.quat_identity()), label="cup")
    cb.add_shape_mesh(body=cup_body, mesh=cup_mesh)
    assert cb.body_count == 1 and cb.shape_count == 1, "cup mesh not added"
    assert cup_mesh.vertices.shape[0] > 100, "cup mesh suspiciously small"
    report["cup"] = {
        "vertices": int(cup_mesh.vertices.shape[0]),
        "triangles": int(cup_mesh.indices.shape[0] // 3),
        "scale": [round(float(s), 4) for s in scale],
    }
    print(
        f"[4] cup USD OK: {cup_mesh.vertices.shape[0]} verts, "
        f"{cup_mesh.indices.shape[0] // 3} tris, scale={report['cup']['scale']}"
    )

    # --- 5. franka steps on GPU with PD targets ----------------------------
    fb.default_shape_cfg = newton.ModelBuilder.ShapeConfig(kh=1e11, gap=0.01)
    fb.joint_target_ke[:7], fb.joint_target_kd[:7] = ARM_KE, ARM_KD
    fb.joint_target_ke[7:9], fb.joint_target_kd[7:9] = FINGER_KE, FINGER_KD
    fb.joint_effort_limit[:7], fb.joint_armature[:7] = ARM_EFFORT, ARM_ARMATURE
    fb.joint_effort_limit[7:9], fb.joint_armature[7:9] = FINGER_EFFORT, FINGER_ARMATURE
    fb.joint_q[:9] = list(HOME_Q)
    fb.joint_target_q[:9] = list(HOME_Q)
    fb.add_ground_plane()
    fmodel = fb.finalize()
    f0, f1 = fmodel.state(), fmodel.state()
    newton.eval_fk(fmodel, fmodel.joint_q, fmodel.joint_qd, f0)
    fctrl = fmodel.control()
    fsolver = newton.solvers.SolverMuJoCo(fmodel, iterations=15, ls_iterations=20, njmax=512, nconmax=256)
    for _ in range(60):
        f0.clear_forces()
        fsolver.step(f0, f1, fctrl, None, 1.0 / 240)
        f0, f1 = f1, f0
    wp.synchronize()
    fq = f0.body_q.numpy()
    assert np.all(np.isfinite(fq)), "NaN in franka body_q"
    ee_pose = fq[10]
    report["ee_body_q"] = [round(float(v), 4) for v in ee_pose]
    print(f"[5] franka stepped on GPU; ee body(10) pos={report['ee_body_q'][:3]}")

    report["elapsed_s"] = round(time.time() - t0, 1)
    print(f"CHECK_ENV PASS ({report['elapsed_s']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
