#
# Copyright (C) 2026
#
# Licensed under the Apache License, Version 2.0
#
"""LeRobot ACT policy for SFP/SC cable insertion (team submission).

Keep this module import-light: ``aic_model`` imports it in ``__init__`` before
``executor.spin()``, so heavy imports block lifecycle ``GetState`` responses.
"""

from __future__ import annotations

from rclpy.exceptions import ParameterAlreadyDeclaredException
from rclpy.node import Node

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task


class SfpInsertACT(Policy):
    """ACT policy tuned for cheatcode-recorded cartesian datasets at 10 Hz."""

    @staticmethod
    def _ensure_cartesian_frame_id(parent_node: Node) -> str:
        try:
            parent_node.declare_parameter("cartesian_twist_frame_id", "gripper/tcp")
        except ParameterAlreadyDeclaredException:
            pass
        fid = (
            parent_node.get_parameter("cartesian_twist_frame_id")
            .get_parameter_value()
            .string_value.strip()
        )
        return fid or "gripper/tcp"

    def __init__(self, parent_node: Node):
        Policy.__init__(self, parent_node)
        self._parent_node = parent_node
        self._run_act = None
        self._cartesian_twist_frame_id = self._ensure_cartesian_frame_id(parent_node)
        self._declare_runtime_parameters(parent_node)
        self._control_hz = self._read_float_param(parent_node, "control_hz", 10.0)
        self._trial_duration_sec = self._read_float_param(
            parent_node, "trial_duration_sec", 60.0
        )
        self._action_scale = self._read_float_param(parent_node, "action_scale", 1.0)
        self._action_log_interval = max(
            1, int(self._read_float_param(parent_node, "action_log_interval", 20))
        )
        self._step_count = 0
        self.get_logger().info(
            "SfpInsertACT configured (ACT loads on first insert_cable): "
            f"frame_id={self._cartesian_twist_frame_id!r} control_hz={self._control_hz}"
        )

    def _ensure_act_loaded(self) -> None:
        if self._run_act is not None:
            return
        self.get_logger().info("SfpInsertACT loading ACT weights...")
        from aic_example_policies.ros.RunACT import RunACT

        self._run_act = RunACT(self._parent_node)
        self.get_logger().info(
            f"SfpInsertACT ACT ready on {self._run_act.device}, "
            f"frame_id={self._cartesian_twist_frame_id!r}"
        )

    @staticmethod
    def _declare_runtime_parameters(parent_node: Node) -> None:
        for name, default in (
            ("control_hz", 10.0),
            ("trial_duration_sec", 60.0),
            ("action_scale", 1.0),
            ("action_log_interval", 20.0),
        ):
            try:
                parent_node.declare_parameter(name, default)
            except ParameterAlreadyDeclaredException:
                pass

    @staticmethod
    def _read_float_param(parent_node: Node, name: str, fallback: float) -> float:
        value = parent_node.get_parameter(name).value
        try:
            return float(value)
        except (TypeError, ValueError):
            return fallback

    def set_cartesian_twist_target(self, twist, frame_id: str = "gripper/tcp"):
        import numpy as np
        from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
        from geometry_msgs.msg import Vector3, Wrench

        motion_update_msg = MotionUpdate()
        motion_update_msg.velocity = twist
        motion_update_msg.header.frame_id = frame_id
        motion_update_msg.header.stamp = self.get_clock().now().to_msg()
        motion_update_msg.target_stiffness = np.diag(
            [85.0, 85.0, 85.0, 85.0, 85.0, 85.0]
        ).flatten()
        motion_update_msg.target_damping = np.diag(
            [75.0, 75.0, 75.0, 75.0, 75.0, 75.0]
        ).flatten()
        motion_update_msg.feedforward_wrench_at_tip = Wrench(
            force=Vector3(x=0.0, y=0.0, z=0.0),
            torque=Vector3(x=0.0, y=0.0, z=0.0),
        )
        motion_update_msg.wrench_feedback_gains_at_tip = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        motion_update_msg.trajectory_generation_mode.mode = (
            TrajectoryGenerationMode.MODE_VELOCITY
        )
        return motion_update_msg

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        **kwargs,
    ):
        import time

        import torch
        from geometry_msgs.msg import Twist, Vector3

        self._ensure_act_loaded()
        run_act = self._run_act
        run_act.policy.reset()
        self._step_count = 0
        self.get_logger().info(f"SfpInsertACT.insert_cable() task={task}")

        period = 1.0 / max(self._control_hz, 1.0)
        deadline = time.time() + max(self._trial_duration_sec, period)

        while time.time() < deadline:
            loop_start = time.time()
            observation_msg = get_observation()
            if observation_msg is None:
                time.sleep(period)
                continue

            obs_tensors = run_act.prepare_observations(observation_msg)
            with torch.inference_mode():
                normalized_action = run_act.policy.select_action(obs_tensors)

            raw_action_tensor = (
                normalized_action * run_act.action_std
            ) + run_act.action_mean
            action = raw_action_tensor[0].cpu().numpy()[:6] * self._action_scale

            self._step_count += 1
            if self._step_count % self._action_log_interval == 0:
                self.get_logger().info(
                    f"step={self._step_count} action="
                    f"[{action[0]:.4f}, {action[1]:.4f}, {action[2]:.4f}, "
                    f"{action[3]:.4f}, {action[4]:.4f}, {action[5]:.4f}]"
                )

            twist = Twist(
                linear=Vector3(
                    x=float(action[0]), y=float(action[1]), z=float(action[2])
                ),
                angular=Vector3(
                    x=float(action[3]), y=float(action[4]), z=float(action[5])
                ),
            )
            motion_update = self.set_cartesian_twist_target(
                twist, frame_id=self._cartesian_twist_frame_id
            )
            move_robot(motion_update=motion_update)
            send_feedback("in progress...")

            elapsed = time.time() - loop_start
            time.sleep(max(0.0, period - elapsed))

        self.get_logger().info(
            f"SfpInsertACT.insert_cable() done after {self._step_count} steps"
        )
        return True
