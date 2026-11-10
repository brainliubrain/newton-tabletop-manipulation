"""Mesh-level finger/bowl penetration audit (visual geometry), every grasp-phase frame."""
import csv, os, sys
import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_physics_video as R

model = mujoco.MjModel.from_xml_string(R.build_studio_xml())
data = mujoco.MjData(model)
with open(R.CSV_PATH, encoding="utf-8") as f:
    rows = list(csv.DictReader([l.strip() for l in f if l.strip() and not l.startswith("#")]))

jn = lambda n: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
ja = [jn(f"joint{i}") for i in range(1, 8)]
jf1, jf2, jb, jc = jn("finger_joint1"), jn("finger_joint2"), jn("bowl_joint"), jn("cup_joint")
bowl_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bowl")

# outer profile (r,z) of bowl from the generator
prof = np.array(R.BOWL_OUTER_PROFILE) if hasattr(R, "BOWL_OUTER_PROFILE") else None
if prof is None:
    raise SystemExit("BOWL_OUTER_PROFILE not exported")
pz, pr = prof[:, 1], prof[:, 0]

finger_geoms = []
for g in range(model.ngeom):
    b = model.geom_bodyid[g]
    bn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
    if bn in ("left_finger", "right_finger") and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
        finger_geoms.append(g)

worst = (0.0, -1)
for i, row in enumerate(rows):
    for k in range(7):
        data.qpos[ja[k]] = float(row[f"j{k}"])
    data.qpos[jf1] = data.qpos[jf2] = float(row["finger_gap"]) * 0.5
    data.qpos[jb:jb+3] = [float(row["bowl_x"]), float(row["bowl_y"]), float(row["bowl_z"])]
    data.qpos[jb+3:jb+7] = [float(row[c]) for c in ("bowl_qw", "bowl_qx", "bowl_qy", "bowl_qz")]
    data.qpos[jc:jc+3] = [float(row["cup_x"]), float(row["cup_y"]), float(row["cup_z"])]
    data.qpos[jc+3:jc+7] = [float(row[c]) for c in ("cup_qw", "cup_qx", "cup_qy", "cup_qz")]
    mujoco.mj_forward(model, data)
    bp, bm = data.xpos[bowl_id], data.xmat[bowl_id].reshape(3, 3)
    depth = 0.0
    for g in finger_geoms:
        m = model.geom_dataid[g]
        v = model.mesh_vert[model.mesh_vertadr[m]:model.mesh_vertadr[m] + model.mesh_vertnum[m]]
        w = v @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
        lb = (w - bp) @ bm  # bowl frame
        z = lb[:, 2]
        sel = (z >= 0) & (z <= pz.max())
        if not sel.any():
            continue
        rad = np.hypot(lb[sel, 0], lb[sel, 1])
        rmax = np.interp(z[sel], pz, pr)
        d = rmax - rad  # >0 means vertex is inside bowl outer wall
        depth = max(depth, float(d.max()))
        if i in (150,180,211,240,300,400): print(i, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g), 'gap',float(row['finger_gap']), 'npts',int(sel.sum()), 'min_rad_mm', round(float(rad.min())*1000,2), 'margin_mm', round(float((-d).min())*1000,2), 'bowl_z',round(bp[2],4))
    if depth > worst[0]:
        worst = (depth, i)
    if i % 40 == 0 or depth > 0.0005:
        if depth > 0:
            print(f"frame {i:4d} gap={float(row['finger_gap'])*1000:.1f}mm penetration={depth*1000:.2f}mm")
print(f"WORST penetration: {worst[0]*1000:.2f} mm at frame {worst[1]}")


