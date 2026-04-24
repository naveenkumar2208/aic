#!/usr/bin/env python3

#
#  Copyright (C) 2026 Intrinsic Innovation LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

"""
Joint-space teleoperation using a gamepad (Xbox or PlayStation layout).

Uses pygame to read the first connected joystick. SDL axis/button indices differ
by OS and driver; the two profiles match typical Linux + pygame/SDL mappings.
If axes feel wrong, try the other profile or adjust GAMEPAD_PROFILES below.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pygame
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from aic_control_interfaces.msg import (
    JointMotionUpdate,
    TrajectoryGenerationMode,
    TargetMode,
)
from aic_control_interfaces.srv import ChangeTargetMode

SLOW_ANGULAR_VEL = 0.075
FAST_ANGULAR_VEL = 0.2


@dataclass(frozen=True)
class GamepadProfile:
    """Axis/button indices for pygame.Joystick (SDL2)."""

    name: str
    # Stick axes: (axis_index, sign) for shoulder_pan, shoulder_lift, elbow, wrist_1
    axis_j1: tuple[int, float]
    axis_j2: tuple[int, float]
    axis_j3: tuple[int, float]
    axis_j4: tuple[int, float]
    # Analog triggers (0..1): left index, right index — wrist_2 from (R - L)
    trigger_left_axis: int
    trigger_right_axis: int
    # Digital bumpers for wrist_3: LB / RB (or L1 / R1)
    bumper_neg_button: int
    bumper_pos_button: int
    # D-pad hat index (usually 0); use vertical for slow/fast
    hat_index: int
    # Two buttons must be held together to exit
    quit_button_a: int
    quit_button_b: int


# Typical SDL2 / pygame on Linux for Xbox One / Series and DS4 / DualSense.
GAMEPAD_PROFILES: dict[str, GamepadProfile] = {
    "xbox": GamepadProfile(
        name="Xbox (One / Series / XInput-style)",
        axis_j1=(0, 1.0),
        axis_j2=(1, -1.0),
        axis_j3=(3, 1.0),
        axis_j4=(4, -1.0),
        trigger_left_axis=2,
        trigger_right_axis=5,
        bumper_neg_button=4,
        bumper_pos_button=5,
        hat_index=0,
        quit_button_a=6,
        quit_button_b=7,
    ),
    "ps4": GamepadProfile(
        name="PlayStation (DualShock 4 / DualSense-style)",
        axis_j1=(0, 1.0),
        axis_j2=(1, -1.0),
        # Typical Linux/SDL: right stick X/Y on 3,4; L2/R2 analog on 2,5
        axis_j3=(3, 1.0),
        axis_j4=(4, -1.0),
        trigger_left_axis=2,
        trigger_right_axis=5,
        bumper_neg_button=4,
        bumper_pos_button=5,
        hat_index=0,
        quit_button_a=8,
        quit_button_b=9,
    ),
}


def _deadzone(value: float, dz: float) -> float:
    if abs(value) < dz:
        return 0.0
    scale = (abs(value) - dz) / (1.0 - dz)
    return float(np.copysign(scale, value))


def _axis(js: pygame.joystick.Joystick, index: int) -> float:
    if index < 0 or index >= js.get_numaxes():
        return 0.0
    return float(js.get_axis(index))


def _button(js: pygame.joystick.Joystick, index: int) -> bool:
    if index < 0 or index >= js.get_numbuttons():
        return False
    return bool(js.get_button(index))


def _hat(js: pygame.joystick.Joystick, hat_index: int) -> tuple[int, int]:
    if hat_index < 0 or hat_index >= js.get_numhats():
        return (0, 0)
    return js.get_hat(hat_index)


def _trigger_01(js: pygame.joystick.Joystick, axis_index: int) -> float:
    """Normalize LT/RT style axis to 0..1 (supports 0..1 or -1..1 drivers)."""
    v = _axis(js, axis_index)
    if -0.02 <= v <= 1.02:
        return float(np.clip(v, 0.0, 1.0))
    return float(np.clip((v + 1.0) / 2.0, 0.0, 1.0))


def prompt_controller_layout() -> str:
    print(
        "\nSelect your gamepad layout (axis indices differ between Xbox and PlayStation):\n"
        "  1) Xbox (One / Series, XInput-style mapping)\n"
        "  2) PlayStation (DualShock 4 / DualSense-style mapping)\n"
    )
    while True:
        choice = input("Enter 1 or 2: ").strip()
        if choice == "1":
            return "xbox"
        if choice == "2":
            return "ps4"
        print("Invalid choice. Type 1 or 2.")


def print_layout_help(profile: GamepadProfile) -> None:
    print(
        f"""
Gamepad joint teleoperation — {profile.name}
================================================================================
Joints (velocity mode, same topics as joint_keyboard_teleop):

  Left stick horizontal   → Joint 1  (shoulder_pan_joint)   axis {profile.axis_j1[0]}
  Left stick vertical     → Joint 2  (shoulder_lift_joint)  axis {profile.axis_j2[0]} (inverted)
  Right stick horizontal  → Joint 3  (elbow_joint)          axis {profile.axis_j3[0]}
  Right stick vertical    → Joint 4  (wrist_1_joint)        axis {profile.axis_j4[0]} (inverted)

  Right trigger − Left trigger (analog) → Joint 5 (wrist_2_joint)
      LT/L2 axis {profile.trigger_left_axis},  RT/R2 axis {profile.trigger_right_axis}

  LB/L1 held → wrist_3 negative    (button {profile.bumper_neg_button})
  RB/R1 held → wrist_3 positive     (button {profile.bumper_pos_button})

  D-pad UP    → slow speed ({SLOW_ANGULAR_VEL} rad/s)
  D-pad DOWN  → fast speed ({FAST_ANGULAR_VEL} rad/s)

  Hold BOTH quit buttons to exit:
      Xbox: View + Menu (buttons {profile.quit_button_a} + {profile.quit_button_b})
      PS:   Share + Options (same indices on PS profile: {profile.quit_button_a} + {profile.quit_button_b})

You can also press Ctrl+C in this terminal to stop.
================================================================================
"""
    )


class AICGamepadJointTeleoperatorNode(Node):
    def __init__(self, profile: GamepadProfile, deadzone: float = 0.12):
        super().__init__("aic_gamepad_joint_teleop")
        self.profile = profile
        self.deadzone = deadzone
        self.angular_vel = FAST_ANGULAR_VEL

        self.controller_namespace = self.declare_parameter(
            "controller_namespace", "aic_controller"
        ).value

        self.joint_motion_update_publisher = self.create_publisher(
            JointMotionUpdate,
            f"/{self.controller_namespace}/joint_commands",
            10,
        )

        while self.joint_motion_update_publisher.get_subscription_count() == 0:
            self.get_logger().info(
                f"Waiting for subscriber to '{self.controller_namespace}/joint_commands'..."
            )
            time.sleep(1.0)

        self.client = self.create_client(
            ChangeTargetMode,
            f"/{self.controller_namespace}/change_target_mode",
        )
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(
                f"Waiting for service '{self.controller_namespace}/change_target_mode'..."
            )

    def generate_joint_motion_update(self, velocities: np.ndarray) -> JointMotionUpdate:
        msg = JointMotionUpdate()
        msg.target_state.velocities = velocities.tolist()
        msg.target_stiffness = [85.0, 85.0, 85.0, 85.0, 85.0, 85.0]
        msg.target_damping = [75.0, 75.0, 75.0, 75.0, 75.0, 75.0]
        msg.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_VELOCITY
        return msg

    def process_joystick(self, js: pygame.joystick.Joystick) -> bool:
        """
        Read joystick and publish one command. Returns False if user requested quit.
        """
        p = self.profile

        hat = _hat(js, p.hat_index)
        if hat[1] > 0:
            if self.angular_vel != SLOW_ANGULAR_VEL:
                self.get_logger().info(f"Slow mode: {SLOW_ANGULAR_VEL} rad/s")
            self.angular_vel = SLOW_ANGULAR_VEL
        elif hat[1] < 0:
            if self.angular_vel != FAST_ANGULAR_VEL:
                self.get_logger().info(f"Fast mode: {FAST_ANGULAR_VEL} rad/s")
            self.angular_vel = FAST_ANGULAR_VEL

        v = np.zeros(6, dtype=float)

        a1 = _deadzone(_axis(js, p.axis_j1[0]) * p.axis_j1[1], self.deadzone)
        a2 = _deadzone(_axis(js, p.axis_j2[0]) * p.axis_j2[1], self.deadzone)
        a3 = _deadzone(_axis(js, p.axis_j3[0]) * p.axis_j3[1], self.deadzone)
        a4 = _deadzone(_axis(js, p.axis_j4[0]) * p.axis_j4[1], self.deadzone)

        w = self.angular_vel
        v[0] = a1 * w
        v[1] = a2 * w
        v[2] = a3 * w
        v[3] = a4 * w

        lt = _trigger_01(js, p.trigger_left_axis)
        rt = _trigger_01(js, p.trigger_right_axis)
        trig = float(np.clip(rt - lt, -1.0, 1.0))
        if abs(trig) > 0.08:
            v[4] = trig * w

        if _button(js, p.bumper_neg_button):
            v[5] -= w
        if _button(js, p.bumper_pos_button):
            v[5] += w

        self.joint_motion_update_publisher.publish(
            self.generate_joint_motion_update(v)
        )

        if _button(js, p.quit_button_a) and _button(js, p.quit_button_b):
            self.get_logger().info("Quit button combo pressed — exiting.")
            return False
        return True

    def send_change_control_mode_req(self, mode: int) -> None:
        req = ChangeTargetMode.Request()
        req.target_mode.mode = mode
        self.get_logger().info(f"Requesting control mode {mode}")
        future = self.client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        if response.success:
            self.get_logger().info("Control mode change accepted.")
        else:
            self.get_logger().warn("Control mode change failed.")
        time.sleep(0.5)


def main(args: Sequence[str] | None = None) -> None:
    layout_key = prompt_controller_layout()
    profile = GAMEPAD_PROFILES[layout_key]
    print_layout_help(profile)

    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() < 1:
        print("No gamepad detected. Connect a controller and try again.")
        pygame.quit()
        return

    js = pygame.joystick.Joystick(0)
    js.init()
    print(f"Using joystick: {js.get_name()}\n")

    rclpy.init(args=args)
    node: AICGamepadJointTeleoperatorNode | None = None
    try:
        node = AICGamepadJointTeleoperatorNode(profile=profile)
        node.send_change_control_mode_req(TargetMode.MODE_JOINT)

        clock = pygame.time.Clock()
        rate_hz = 50
        running = True
        while rclpy.ok() and running:
            pygame.event.pump()
            running = node.process_joystick(js)
            rclpy.spin_once(node, timeout_sec=0.0)
            clock.tick(rate_hz)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()
        pygame.quit()


if __name__ == "__main__":
    main(sys.argv)
