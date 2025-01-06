import ipdb.stdout
import mujoco
import mujoco_py
import time
import os
import torch
import torch.nn as nn
import multiprocessing as mp
import io
import imageio

from sim_web_visualizer.mjc_visualizer_client import create_mjc_visualizer, bind_visualizer_to_sim


frames = []

create_mjc_visualizer(port=6000, host="localhost", keep_default_viewer=True)

num_cpu = 1

mj_path = mujoco_py.utils.discover_mujoco()
xml_path = os.path.join(mj_path, "model", "humanoid.xml")
model = mujoco_py.load_model_from_path(xml_path)
model.opt.timestep = 1.0 / 60
sim = mujoco_py.MjSim(model)
sim.forward()
# headless
render_context = mujoco_py.MjRenderContextOffscreen(sim, device_id=-1)
render_context.cam.distance = 5
######
# has screen
# viewer = mujoco_py.MjViewer(sim)
# viewer.cam.distance = 5
####

sim = bind_visualizer_to_sim(sim, xml_path)
sim.data.qpos[2] = 4.95
sim.data.qpos[0] = 0.95
sim.data.qpos[1] = 0.95
sim.forward()


for i in range(2000):
    start_time = time.time()
    for _ in range(2):
        sim.step()
    end_time = time.time()
    delta_time = end_time - start_time
    print(f"{i:04}/{2000}: Time elapsed: {delta_time:.6f} seconds, FPS: {1 / delta_time:.2f}")
    width = 640
    height = 480
    render_context.render(width, height)
    rgb = render_context.read_pixels(width, height, depth=False)
    # image need to be flip
    rgb = rgb[::-1, :, :]
    frames.append(rgb)
    # has screen
    # viewer.render()
    ####

imageio.mimsave("output.mp4", frames, fps=30)
