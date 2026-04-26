# aic_teleoperation

Keyboard-, gamepad-, and scripted ground-truth teleoperation for the robot (joint-space, Cartesian-space, and CheatCode-style insertion).

## Prerequisites

1. Follow the [Getting Started Guide](../../docs/getting_started.md) to set up your development environment
2. X11 display server (pynput has known issues with Wayland)
3. For native builds: `sudo apt install python3-pynput` (pixi installs `pynput` and `pygame` automatically)

## Available Scripts

### 1. Joint Space Teleoperation (`joint_keyboard_teleop`)

Control individual robot joints directly.

**Key Mappings:**
- `q/a` - Joint 1 (shoulder_pan_joint): +/-
- `w/s` - Joint 2 (shoulder_lift_joint): +/-
- `e/d` - Joint 3 (elbow_joint): +/-
- `r/f` - Joint 4 (wrist_1_joint): +/-
- `t/g` - Joint 5 (wrist_2_joint): +/-
- `y/h` - Joint 6 (wrist_3_joint): +/-

**Speed Control:**
- `k` - Slow mode (0.075 rad/s)
- `l` - Fast mode (0.2 rad/s)

**Exit:**
- `ESC` - Quit teleoperation

### 2. Gamepad Joint Teleoperation (`gamepad_joint_teleop`)

Control joints with an **Xbox** or **PlayStation** gamepad. On launch, choose layout `1` or `2` so axis indices match your controller.

**Sticks**

| Input | Joint | Robot |
|-------|-------|--------|
| Left stick ↔ | ± | 1 — shoulder_pan |
| Left stick ↕ | ± | 2 — shoulder_lift |
| Right stick ↔ | ± | 3 — elbow |
| Right stick ↕ | ± | 4 — wrist_1 |

**Triggers (joint 5 — wrist_2)**

| Input | Effect |
|-------|--------|
| Right trigger stronger | +velocity |
| Left trigger stronger | −velocity |

**Bumpers (joint 6 — wrist_3)**

| Input | Effect |
|-------|--------|
| LB / L1 | −velocity |
| RB / R1 | +velocity |

**D-pad (speed, same rad/s as keyboard teleop)**

| Input | Effect |
|-------|--------|
| D-pad **up** | Slow (0.075 rad/s) |
| D-pad **down** | Fast (0.2 rad/s) |

**Quit**

| Layout | Action |
|--------|--------|
| Xbox | Hold **View + Menu** |
| PlayStation | Hold **Share + Options** |

You can also press **Ctrl+C** in the terminal. If sticks or triggers map wrong for your OS/driver, try the other layout or edit `GAMEPAD_PROFILES` in `gamepad_joint_teleop.py`.

### 3. Ground-truth scripted insertion (`cheatcode_ground_truth_teleop`)

Runs the same plug-in-port logic as [`CheatCode.py`](../../aic_example_policies/aic_example_policies/ros/CheatCode.py) by publishing Cartesian **pose** commands to `aic_controller` (no keyboard or SpaceMouse). Use when the simulator publishes **ground-truth TF** for the port and plug (launch with `ground_truth:=true`; see [`aic_bringup/README.md`](../../aic_bringup/README.md) for spawning the task board and cable).

Default TF names match [`sample_config.yaml`](../../aic_engine/config/sample_config.yaml) trial 1 (`cable_0`, `sfp_tip`, **`nic_card_mount_0`**, `sfp_port_0`). If you spawn a different mount (e.g. `nic_card_mount_2_present:=true` on `aic_gz_bringup`), set **`-p target_module_name:=nic_card_mount_2`** so the port frame matches the board (`task_board/nic_card_mount_2/sfp_port_0_link`). Ground truth must stay on (`ground_truth:=true`).

**Parameters:** `controller_namespace`, `cable_name`, `plug_name`, `port_name`, `target_module_name`, `task_id`, `startup_delay_sec`.

The node exits after one insertion attempt (success or failure). This is useful for scripted demos or generating consistent motion (similar in spirit to community workflows that use CheatCode for data-related experiments, e.g. [Open Robotics Discourse](https://discourse.openrobotics.org/t/rockys-open-source-build-thread-ai-for-industry-challenge/53155/11)).

### 4. Cartesian Space Teleoperation (`cartesian_keyboard_teleop`)

Control end-effector pose (position and orientation).

**Linear Movement:**
- `a/d` - X axis: -/+
- `w/s` - Y axis: -/+
- `r/f` - Z axis: -/+

**Angular Movement:**
- `Shift + s/w` : -/+ Angular X
- `Shift + a/d` : -/+ Angular Y
- `q/e` : -/+ Angular Z

**Speed Control:**
- `k` - Slow mode (linear: 0.02 m/s, angular: 0.02 rad/s)
- `l` - Fast mode (linear: 0.1 m/s, angular: 0.1 rad/s)

**Frame Toggle:**
- `n` - Tool frame (`gripper/tcp`)
- `m` - Global frame (`base_link`)

**Exit:**
- `ESC` - Quit teleoperation

## Usage

Start the evaluation environment first following the [Getting Started - Quick Start](../../docs/getting_started.md#quick-start) instructions.

### With pixi (Recommended)

```bash
cd ~/ws_aic/src/aic

# Run teleoperation
pixi run ros2 run aic_teleoperation joint_keyboard_teleop
# or
pixi run ros2 run aic_teleoperation gamepad_joint_teleop
# or
pixi run ros2 run aic_teleoperation cheatcode_ground_truth_teleop --ros-args -p use_sim_time:=true
# or
pixi run ros2 run aic_teleoperation cartesian_keyboard_teleop
```

### With native ROS 2 build

See [Building the Evaluation Component from Source](../../docs/build_eval.md) for workspace build and Zenoh setup.

```bash
# Run teleoperation
ros2 run aic_teleoperation joint_keyboard_teleop
# or
ros2 run aic_teleoperation gamepad_joint_teleop
# or
ros2 run aic_teleoperation cheatcode_ground_truth_teleop --ros-args -p use_sim_time:=true
# or
ros2 run aic_teleoperation cartesian_keyboard_teleop
```

## Notes

- The scripts automatically switch the robot controller to the appropriate control mode (joint or Cartesian) when started
- **Keyboard:** Press ESC to exit; input is captured even when the terminal does not have focus
- **Gamepad:** Exit with the two-button combo in the table above, or Ctrl+C in the terminal
