import csv, os, sys
import numpy as np, mujoco
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_physics_video as R
model = mujoco.MjModel.from_xml_string(R.build_studio_xml()); data = mujoco.MjData(model)
rows = list(csv.DictReader([l.strip() for l in open(R.CSV_PATH, encoding="utf-8") if l.strip() and not l.startswith("#")]))
jn = lambda n: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
ja=[jn(f"joint{i}") for i in range(1,8)]; jf1,jf2,jb,jc=jn("finger_joint1"),jn("finger_joint2"),jn("bowl_joint"),jn("cup_joint")
fr=int(sys.argv[1]) if len(sys.argv)>1 else 240
row=rows[fr]
for k in range(7): data.qpos[ja[k]]=float(row[f"j{k}"])
data.qpos[jf1]=data.qpos[jf2]=float(row["finger_gap"])*0.5
data.qpos[jb:jb+3]=[float(row[c]) for c in("bowl_x","bowl_y","bowl_z")]; data.qpos[jb+3:jb+7]=[float(row[c]) for c in("bowl_qw","bowl_qx","bowl_qy","bowl_qz")]
data.qpos[jc:jc+3]=[float(row[c]) for c in("cup_x","cup_y","cup_z")]; data.qpos[jc+3:jc+7]=[float(row[c]) for c in("cup_qw","cup_qx","cup_qy","cup_qz")]
mujoco.mj_forward(model,data)
bp=data.xpos[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"bowl")]
r=mujoco.Renderer(model,height=540,width=720)
out=[]
for az,el in((90,-10),(180,-10),(135,-35)):
    cam=mujoco.MjvCamera(); cam.type=mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:]=data.site_xpos[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'tcp')]; cam.distance=0.30; cam.azimuth=az; cam.elevation=el
    r.update_scene(data,camera=cam); out.append(r.render())
Image.fromarray(np.concatenate(out,1)).save(os.path.join(R.RESULTS_DIR,f"closeup_f{fr}.png"))
print('bowl',bp,'tcp',data.site_xpos[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'tcp')])

