"""
QAV250 Drone Simulation Environment
====================================
Models a 250mm quadcopter with physics matching the Holybro QAV250 kit:
  - Frame: 250mm wheelbase, ~150g
  - Motors: 2207 1950KV brushless
  - Props: 5045 (5 inch, 4.5 pitch)
  - Battery: 4S LiPo (~200g)
  - Total weight: ~500g

Coordinate system:
  - X: forward
  - Y: left
  - Z: up

Motor layout (top-down view):
    M1(CCW)  M2(CW)
        [FRONT]
    M3(CW)   M4(CCW)
"""

import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
import random


# ── Physical constants ────────────────────────────────────────────────────────
GRAVITY          = 9.89          # m/s²
DRONE_MASS       = 0.5           # kg  (frame + electronics + battery)
ARM_LENGTH       = 0.125         # m   (half of 250mm wheelbase)
MOTOR_KV         = 1950          # KV rating (1950KV)
PROP_DIAMETER    = 0.127         # m   (5 inches)
PROP_PITCH       = 0.1143        # m   (4.5 inches)
MAX_RPM          = 20000         # realistic max for 4S + 2207 motor
HOVER_RPM        = 12000         # approximate RPM needed to hover

# Thrust/torque coefficients (derived from prop specs)
# Thrust  = KT  * omega²   (omega in rad/s)
# Torque  = KQ  * omega²
MAX_OMEGA  = MAX_RPM * (2 * np.pi / 60)                # keep for KQ torque calc
MAX_THRUST = (DRONE_MASS * GRAVITY) / (4 * 0.5)        # hover at throttle=0.5 → 1.3 N per motor
KT         = MAX_THRUST / (MAX_OMEGA ** 2)              # back-derive KT from physical target
KQ         = KT * 0.016                                 # torque constant unchanged

# Motor positions relative to centre of mass (X, Y, Z)
# Front-right, Front-left, Back-left, Back-right
MOTOR_POSITIONS = np.array([
    [ ARM_LENGTH,  ARM_LENGTH, 0],   # M1 - front right - CCW
    [-ARM_LENGTH,  ARM_LENGTH, 0],   # M2 - front left  - CW
    [-ARM_LENGTH, -ARM_LENGTH, 0],   # M3 - back left   - CCW
    [ ARM_LENGTH, -ARM_LENGTH, 0],   # M4 - back right  - CW
])

# Spin directions (+1 CCW, -1 CW) — affects yaw torque
MOTOR_DIRS = np.array([1, -1, 1, -1])

# ── Simulation parameters ─────────────────────────────────────────────────────
SIM_HZ        = 240     # physics steps per second
CTRL_HZ       = 48      # control decisions per second
STEPS_PER_CTRL = SIM_HZ // CTRL_HZ   # = 5 physics steps per control step
MAX_EPISODE_STEPS = 1000             # ~20 seconds of flight per episode


class DroneEnv(gym.Env):
    """
    Gymnasium environment for training a drone to hover and follow a target.

    Observation space (18 values):
        - position xyz          (3)
        - velocity xyz          (3)
        - orientation rpy       (3)   roll, pitch, yaw in radians
        - angular velocity xyz  (3)
        - target position xyz   (3)
        - distance to target    (1)   scalar

    Action space (4 values, continuous):
        - Throttle per motor [0, 1] normalised
        - Maps to actual thrust via MAX_THRUST

    Reward:
        + Staying close to target
        + Staying upright (low roll/pitch)
        - Large angular velocities (instability penalty)
        - Crashing (episode ends)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": CTRL_HZ}

    def __init__(self, render_mode=None, target_pos=None):
        super().__init__()
        self.render_mode  = render_mode
        self.target_pos   = np.array(target_pos or [0.0, 0.0, 1.0])  # default: hover 1m up
        self._physics_client = None
        self._drone_id       = None
        self._step_count     = 0

        # ── Action space: 4 motors, each 0→1 ─────────────────────────────────
        self.action_space = spaces.Box(
            low   = np.zeros(4, dtype=np.float32),
            high  = np.ones(4,  dtype=np.float32),
            dtype = np.float32,
        )

        # ── Observation space ─────────────────────────────────────────────────
        obs_low  = np.full(16, -np.inf, dtype=np.float32)
        obs_high = np.full(16,  np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

    # ── Environment lifecycle ─────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Disconnect existing client if present
        if self._physics_client is not None:
            p.disconnect(self._physics_client)

        # Connect PyBullet
        if self.render_mode == "human":
            self._physics_client = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(
                cameraDistance=2.0,
                cameraYaw=45,
                cameraPitch=-30,
                cameraTargetPosition=[0, 0, 1],
                physicsClientId=self._physics_client,
            )
        else:
            self._physics_client = p.connect(p.DIRECT)

        cid = self._physics_client
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
        p.setGravity(0, 0, -GRAVITY, physicsClientId=cid)
        p.setTimeStep(1.0 / SIM_HZ, physicsClientId=cid)

        # Load ground plane
        p.loadURDF("plane.urdf", physicsClientId=cid)

        # Build the drone as a simple box body (no URDF needed)
        self._drone_id = self._create_drone(cid)

        # Draw target position marker in GUI mode
        if self.render_mode == "human":
            self._draw_target()

        self._step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        cid    = self._physics_client
        action = np.clip(action, 0.0, 1.0)

        # Run multiple physics steps per control step
        for _ in range(STEPS_PER_CTRL):
            self._apply_motor_forces(action, cid)
            p.stepSimulation(physicsClientId=cid)

        obs     = self._get_obs()
        reward  = self._compute_reward(obs, action)
        terminated = self._is_terminated(obs)
        truncated  = self._step_count >= MAX_EPISODE_STEPS

        self._step_count += 1
        return obs, reward, terminated, truncated, {}

    def render(self):
        pass  # GUI rendering handled by PyBullet directly

    def close(self):
        if self._physics_client is not None:
            p.disconnect(self._physics_client)
            self._physics_client = None

    # ── Physics helpers ───────────────────────────────────────────────────────

    def _create_drone(self, cid):
        """Build a simple quadcopter body using PyBullet primitives."""
        # Visual: flat box for frame + 4 small spheres for motors
        frame_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[ARM_LENGTH, ARM_LENGTH, 0.01],
            rgbaColor=[0.2, 0.2, 0.2, 1.0],
            physicsClientId=cid,
        )
        frame_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[ARM_LENGTH, ARM_LENGTH, 0.01],
            physicsClientId=cid,
        )

        # Spawn drone slightly above ground
        start_height = np.random.uniform(0.2, 1.8)
        start_pos = [
            np.random.uniform(-0.5, 0.5),
            np.random.uniform(-0.5, 0.5),
            start_height
        ]
        start_orn = p.getQuaternionFromEuler([
                        random.uniform(-0.1, 0.1),  # slight random roll
                        random.uniform(-0.1, 0.1),  # slight random pitch
                        random.uniform(-3.14, 3.14) # yaw random
        ])

        drone_id = p.createMultiBody(
            baseMass            = DRONE_MASS,
            baseCollisionShapeIndex = frame_collision,
            baseVisualShapeIndex    = frame_visual,
            basePosition            = start_pos,
            baseOrientation         = start_orn,
            physicsClientId         = cid,
        )

        # Reduce drag slightly (PyBullet default is high)
        p.changeDynamics(
            drone_id, -1,
            linearDamping  = 0.1,
            angularDamping = 0.1,
            physicsClientId = cid,
        )
        return drone_id

    def _apply_motor_forces(self, action, cid):
        """
        Convert normalised [0,1] action per motor into thrust forces
        and torques, applied in world frame.
        """
        pos, orn = p.getBasePositionAndOrientation(self._drone_id, physicsClientId=cid)
        rot_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)

        for i, throttle in enumerate(action):
            # Thrust force (upward in drone body frame → world frame)
            thrust    = throttle * MAX_THRUST
            force_body = np.array([0, 0, thrust])
            force_world = rot_matrix @ force_body

            # Apply at motor position (offset from centre)
            motor_pos_body  = MOTOR_POSITIONS[i]
            motor_pos_world = np.array(pos) + rot_matrix @ motor_pos_body

            p.applyExternalForce(
                self._drone_id, -1,
                forceObj     = force_world.tolist(),
                posObj       = motor_pos_world.tolist(),
                flags        = p.WORLD_FRAME,
                physicsClientId = cid,
            )

            # Reaction torque (yaw) from motor spin
            torque_val   = KQ * (throttle * MAX_OMEGA) ** 2 * MOTOR_DIRS[i]
            torque_body  = np.array([0, 0, torque_val])
            torque_world = rot_matrix @ torque_body

            p.applyExternalTorque(
                self._drone_id, -1,
                torqueObj   = torque_world.tolist(),
                flags       = p.WORLD_FRAME,
                physicsClientId = cid,
            )

    # ── Observation ───────────────────────────────────────────────────────────

    def _get_obs(self):
        cid = self._physics_client
        pos, orn = p.getBasePositionAndOrientation(self._drone_id, physicsClientId=cid)
        vel, ang_vel = p.getBaseVelocity(self._drone_id, physicsClientId=cid)
        rpy = p.getEulerFromQuaternion(orn)

        pos     = np.array(pos,     dtype=np.float32)
        vel     = np.array(vel,     dtype=np.float32)
        rpy     = np.array(rpy,     dtype=np.float32)
        ang_vel = np.array(ang_vel, dtype=np.float32)
        error = (self.target_pos - pos).astype(np.float32)
        #target  = self.target_pos.astype(np.float32)
        dist    = np.array([np.linalg.norm(error)], dtype=np.float32)

        return np.concatenate([pos, vel, rpy, ang_vel, error, dist])

    # ── Reward ────────────────────────────────────────────────────────────────

    def _compute_reward(self, obs, action):
        pos     = obs[0:3]
        vel     = obs[3:6]
        rpy     = obs[6:9]
        ang_vel = obs[9:12]
        dist    = obs[15]
        
        # --- how close to target position ---
        position_reward = np.exp(-3.0 * dist)              # peaks at 1.0 when exactly on target
        
        # --- how still (velocity near zero) ---
        speed = float(np.linalg.norm(vel))
        stillness_reward = np.exp(-2.0 * speed)            # peaks at 1.0 when stationary
        
        # --- how upright ---
        tilt = rpy[0] ** 2 + rpy[1] ** 2
        upright_reward = np.exp(-2.0 * tilt)               # peaks at 1.0 when level
        
        # --- how rotationally stable ---
        spin = float(np.sum(ang_vel ** 2))
        stability_reward = np.exp(-0.5 * spin)             # peaks at 1.0 when no rotation
        
        # --- combined: ALL conditions must be met simultaneously ---
        hover_reward = (0.4*position_reward) + (0.2*stillness_reward) + (0.2*upright_reward) + (0.2*stability_reward)
        
        # --- hard penalties to keep drone alive and on target ---
        height_deficit    = max(0.0, self.target_pos[2] - pos[2])
        ground_penalty    = -0.5 * (np.exp(1.5 * height_deficit) - 1.0)
        
        height_excess     = max(0.0, pos[2] - self.target_pos[2])
        overshoot_penalty = -2.0 * (np.exp(1.5 * height_excess) - 1.0)
        
        lateral_dist    = float(np.sqrt(pos[0]**2 + pos[1]**2))
        #lateral_penalty = -0.5 * lateral_dist
        lateral_penalty = -0.3 * (np.exp(0.8*lateral_dist)-1.0)
        lateral_vel = float(np.sqrt(obs[3]**2 + obs[4]**2))
        lateral_vel_penalty = -0.5 * lateral_vel
        tilt_penalty = -2.0 * float(tilt)

        airtime_bonus = np.clip(pos[2] / self.target_pos[2], 0.0, 1.0) * 0.5
        
        reward = (
            + 4.0 * hover_reward       # ← dominant signal, only max when ALL conditions met
            + ground_penalty           # ← don't crash
            + overshoot_penalty        # ← don't overshoot
            + lateral_penalty          # ← stay centred
            + lateral_vel_penalty
            + tilt_penalty
            + airtime_bonus
        )
        return float(reward)


#Priority order is now:

#1. airtime_reward  max +2.0   ← stay alive
#2. ground_penalty  up to -8.1 ← don't crash
#3. proximity       max +1.5   ← be near target
#4. height_bonus    max +0.5   ← correct altitude
#5. tilt_penalty    max -0.1   ← stability hint only
#6. spin_penalty    tiny       ← almost negligible

    # ── Termination ───────────────────────────────────────────────────────────

    def _is_terminated(self, obs):
        pos = obs[0:3]
        rpy = obs[6:9]
        crashed   = pos[2] < 0.05
        flipped   = abs(rpy[0]) > 0.5 or abs(rpy[1]) > 1.0
        out_of_bounds = np.any(np.abs(pos[:2]) > 5.0)
        #if crashed or flipped or out_of_bounds:
            #print(f"terminated: crashed={crashed}, flipped={flipped}, oob={out_of_bounds}")
        return bool(crashed or flipped or out_of_bounds)

    # ── GUI helpers ───────────────────────────────────────────────────────────

    def _draw_target(self):
        """Draw a green sphere at the target position in GUI mode."""
        visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius    = 0.05,
            rgbaColor = [0, 1, 0, 0.6],
            physicsClientId=self._physics_client,
        )
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual,
            basePosition=self.target_pos.tolist(),
            physicsClientId=self._physics_client,
        )