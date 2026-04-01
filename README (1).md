# Drone RL — QAV250 Hover Agent

Reinforcement learning simulation for a 250mm quadcopter.
Physics modelled on the **Holybro QAV250** kit with 2207 1950KV motors and 5045 props.

---

## Project Structure

```
drone-rl/
├── envs/
│   └── drone_env.py      ← PyBullet physics environment (Gymnasium compatible)
├── training/
│   ├── train.py          ← PPO training script
│   └── evaluate.py       ← Visual evaluation script
├── models/               ← Saved checkpoints (created on first run)
├── logs/                 ← TensorBoard logs (created on first run)
└── requirements.txt
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install pybullet stable-baselines3 gymnasium numpy
```

### 2. Train the hover agent

```bash
# From the drone-rl/ directory:
python training/train.py
```

This trains for 500,000 steps (~15 mins on a modern CPU).
Checkpoints are saved to `models/` every 50,000 steps.
The best model (by eval reward) is saved to `models/best_model/`.

**Options:**
```bash
python training/train.py --timesteps 1000000   # train longer (better results)
python training/train.py --resume              # continue from last checkpoint
python training/train.py --envs 8             # more parallel envs (faster, needs more RAM)
```

### 3. Watch your agent fly

```bash
python training/evaluate.py
```

Opens a PyBullet GUI window. Green sphere = target hover position.

**Options:**
```bash
python training/evaluate.py --episodes 5
python training/evaluate.py --target 0.5 0.5 1.5   # fly to a different position
```

---

## How It Works

### Environment (`drone_env.py`)
- Models QAV250 physics: mass, motor positions, thrust/torque coefficients
- 4 continuous actions (one per motor, range 0→1)
- 18-dimensional observation: position, velocity, orientation, angular velocity, target, distance
- Episode ends on crash, flip, or 1000 steps (~20 seconds)

### Reward Function
| Component | Effect |
|---|---|
| `exp(-2 * distance)` | Large reward for being close to target |
| `height_bonus` | Encourages climbing toward target altitude |
| `- roll² - pitch²` | Penalises tilting/instability |
| `- angular_velocity²` | Penalises spinning |

### Training (PPO)
PPO (Proximal Policy Optimisation) is used because:
- Works well with continuous action spaces
- Stable — doesn't diverge easily
- Good default for drone control tasks

Network: 2-layer MLP, 256 units per layer.

---

## Next Steps

1. **Sonar following** — add sonar observations and train to follow a moving target
2. **Face tracking** — replace sonar with camera input (Pi Camera module)
3. **Sim-to-real transfer** — once agent is stable, deploy to physical Holybro QAV250

---

## Hardware This Simulation Models

| Part | Spec |
|---|---|
| Frame | QAV250, 250mm wheelbase, carbon fibre |
| Motors | 2207 1950KV brushless x4 |
| ESCs | BLHeli-S 20A x4 |
| Flight controller | Pixhawk 6C Mini |
| Companion computer | Raspberry Pi Zero 2W |
| Battery | 4S LiPo, ~200g |
| Total weight | ~500g |
