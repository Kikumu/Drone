"""
Sandbox — just open the environment and look around.
No training, no agent. Drone falls under gravity with random motor noise.

Run:
    python sandbox.py
"""

import sys, os, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env.drone_env import DroneEnv

env = DroneEnv(render_mode="human", target_pos=[0.0, 0.0, 1.0])
obs, _ = env.reset()

print("Environment open — use mouse to rotate/zoom the camera.")
print("Press Ctrl+C to quit.\n")

step = 0
while True:
    # Just enough throttle to roughly fight gravity — drone will wobble around
    action = np.array([0.58, 0.58, 0.58, 0.58], dtype=np.float32)

    obs, reward, terminated, truncated, _ = env.step(action)

    if step % 50 == 0:
        pos  = obs[0:3]
        rpy  = obs[6:9]
        print(f"step {step:4d} | pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
              f" | roll={np.degrees(rpy[0]):.1f}° pitch={np.degrees(rpy[1]):.1f}°"
              f" | reward={reward:.3f}")

    if terminated or truncated:
        print("--- reset ---")
        obs, _ = env.reset()
        step = 0

    step += 1
    time.sleep(1 / 48)  # match control rate so it doesn't run at warp speed