# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **AI for Industry Challenge (AIC) Toolkit** — an open-source robotics competition framework for autonomous cable insertion by Intrinsic Innovation LLC. Participants implement Python AI policies that control a UR5e robot arm in a Gazebo simulation to insert cables into task boards.

## Build Commands

The project uses **Pixi** as the primary package manager (ROS 2 Kilted Kaiju via RoboStack) and **colcon** as the ROS 2 build system.

```bash
# Enter Pixi environment
pixi shell

# Build all ROS packages
colcon build

# Build with release optimizations
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release

# Build a specific package
colcon build --packages-select aic_model
```

## Lint & Format

```bash
# Python formatting (Black)
black --check .

# Python import sorting (for select packages like lerobot_robot_aic)
isort -c aic_utils/lerobot_robot_aic

# Python type checking
pyright

# C++ formatting (Google style, checked by CI)
clang-format --style=Google -n <file>.cpp
```

## Run Tests

```bash
# Python unit tests
pixi run pytest aic_model/test/
pixi run pytest aic_example_policies/
```

## Launch the Evaluation Environment

```bash
# Launch Gazebo simulation with full evaluation stack
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=false \
  start_aic_engine:=true \
  spawn_task_board:=true

# Run an example policy against the running simulation
pixi run ros2 run aic_model aic_model \
  --ros-args \
  -p use_sim_time:=true \
  -p policy:=aic_example_policies.ros.WaveArm
```

## Architecture

The project has two distinct components:

### Evaluation Component (Organizer-Provided, C++)
- **`aic_engine`** — Trial orchestrator; runs trials, manages state machine, validates compliance
- **`aic_bringup`** — ROS 2 launch files for the Gazebo simulation environment
- **`aic_controller`** — Impedance/compliance controller; accepts Cartesian or joint-space commands
- **`aic_adapter`** — Sensor fusion & time-synchronization layer between hardware/sim and ROS topics
- **`aic_scoring`** — Automated scoring system (measures insertion quality)

### Participant Model Component (What You Develop, Python)
- **`aic_model`** — ROS 2 Lifecycle node that dynamically loads a user `Policy` class at runtime
- **`aic_example_policies`** — Reference implementations: `WaveArm` (minimal), `CheatCode` (ground-truth debug), `RunACT` (ACT imitation learning)
- **`docker/aic_model/Dockerfile`** — Template for containerizing a policy for submission

### Interface Packages (ROS 2 msgs/srvs/actions)
- **`aic_interfaces/`** — Subdirectories: `aic_control_interfaces`, `aic_model_interfaces`, `aic_task_interfaces`, `aic_engine_interfaces`

### Utilities
- **`aic_utils/lerobot_robot_aic/`** — LeRobot framework integration for data collection and training
- **`aic_utils/aic_teleoperation/`** — Teleop interfaces for collecting demonstrations
- **`aic_utils/aic_mujoco/`** and **`aic_utils/aic_isaac/`** — Alternative simulator support

## Policy Development Pattern

Participants implement a single Python class inheriting from `aic_model.Policy` with one method:

```python
class MyPolicy(Policy):
    def insert_cable(self, get_observation, move_robot, send_feedback):
        obs = get_observation()   # Returns Observation msg with images, joints, wrench
        move_robot(motion_update) # Sends MotionUpdate (Cartesian) or JointMotionUpdate
        send_feedback(msg)        # Reports progress to engine
```

The `aic_model` node loads the policy via the `policy` ROS parameter (e.g., `my_package.MyPolicy`), handles all ROS 2 lifecycle management, and calls `insert_cable` in response to the `/insert_cable` action from the engine.

## ROS 2 Communication

**Key input topics (observations):**
- `/left_camera/image`, `/center_camera/image`, `/right_camera/image` — wrist cameras
- `/joint_states`, `/gripper_state` — robot state
- `/fts_broadcaster/wrench` — force/torque sensor
- `/aic_controller/controller_state` — controller status

**Key output topics (commands):**
- `/aic_controller/pose_commands` — Cartesian motion (MotionUpdate)
- `/aic_controller/joint_commands` — joint-space motion (JointMotionUpdate)

**Action server:**
- `/insert_cable` — InsertCable action (triggered by engine, handled by aic_model)

**Service:**
- `/aic_controller/change_target_mode` — switch between Cartesian and joint control modes

## Docker / Deployment

```bash
# Run full evaluation stack with participant model via docker-compose
docker-compose up

# The Zenoh router connects the two containers over port 7447
# GPU support is enabled in docker-compose.yaml
```

Submission requires building a Docker image from `docker/aic_model/Dockerfile`, verifying it against the scoring tests in `docs/scoring_tests.md`, and uploading via the portal with an auth token.

## Code Style

- **Python:** Black (line length 88), isort (Black profile), Pyright (basic mode)
- **C++:** Google style via clang-format v19
- Both are enforced in CI via `.github/workflows/style.yml`

## Documentation

Comprehensive docs in `docs/` — key references:
- `docs/policy.md` — Policy API and integration guide
- `docs/aic_interfaces.md` — Full ROS interface specification
- `docs/aic_controller.md` — Controller details and motion modes
- `docs/scoring.md` and `docs/scoring_tests.md` — Evaluation metrics and reproducible tests
- `docs/submission.md` — Container packaging and upload process
- `docs/troubleshooting.md` — Common issues
