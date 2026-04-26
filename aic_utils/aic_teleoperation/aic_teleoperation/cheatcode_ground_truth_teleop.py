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
Scripted Cartesian teleop using the same ground-truth insertion logic as
`aic_example_policies.ros.CheatCode`.

Use with Gazebo (or sim) launched with ``ground_truth:=true`` and a scene that
provides TF for the task board port and cable plug (e.g. ``spawn_task_board``,
``spawn_cable``, ``attach_cable_to_gripper:=true`` from ``aic_gz_bringup``,
or trials spawned by ``aic_engine``).

This node publishes ``MotionUpdate`` pose targets to ``aic_controller`` — the
same path as ``aic_model`` policies — without running the full ``aic_model``
lifecycle.
"""

from __future__ import annotations

import sys
import threading
import time
import rclpy
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener

from aic_control_interfaces.msg import JointMotionUpdate, MotionUpdate, TargetMode
from aic_control_interfaces.srv import ChangeTargetMode
from aic_example_policies.ros.CheatCode import CheatCode
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task


class CheatCodeGroundTruthTeleopNode(Node):
    """Hosts CheatCode policy logic and forwards ``MotionUpdate`` to the controller."""

    def __init__(self) -> None:
        super().__init__("cheatcode_gt_teleop")

        self.declare_parameter("controller_namespace", "aic_controller")
        self.declare_parameter("cable_name", "cable_0")
        self.declare_parameter("plug_name", "sfp_tip")
        self.declare_parameter("port_name", "sfp_port_0")
        self.declare_parameter("target_module_name", "nic_card_mount_0")
        self.declare_parameter("task_id", "scripted_insert")
        self.declare_parameter("startup_delay_sec", 2.0)

        ns = self.get_parameter("controller_namespace").value

        # Match aic_model: default Buffer() uses tf2's default cache (API differs by distro).
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._pose_pub = self.create_publisher(
            MotionUpdate, f"/{ns}/pose_commands", 10
        )
        self._client = self.create_client(
            ChangeTargetMode, f"/{ns}/change_target_mode"
        )

        while self._pose_pub.get_subscription_count() == 0:
            self.get_logger().info(
                f"Waiting for subscriber on /{ns}/pose_commands ...",
            )
            time.sleep(0.5)

        while not self._client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(
                f"Waiting for service /{ns}/change_target_mode ...",
            )

        self.get_logger().info("Controller endpoints ready.")

    def _move_robot(
        self,
        motion_update: MotionUpdate | None = None,
        joint_motion_update: JointMotionUpdate | None = None,
    ) -> None:
        if motion_update is not None:
            motion_update.header.stamp = self.get_clock().now().to_msg()
            self._pose_pub.publish(motion_update)
        if joint_motion_update is not None:
            self.get_logger().warn("Ignoring joint_motion_update (Cartesian-only node).")

    def _change_target_mode(self, mode: int) -> bool:
        req = ChangeTargetMode.Request()
        req.target_mode.mode = mode
        future = self._client.call_async(req)
        while rclpy.ok() and not future.done():
            time.sleep(0.01)
        resp = future.result()
        return bool(resp and resp.success)

    def _build_task(self) -> Task:
        t = Task()
        t.id = str(self.get_parameter("task_id").value)
        t.cable_type = "sfp_sc"
        t.cable_name = str(self.get_parameter("cable_name").value)
        t.plug_type = "sfp"
        t.plug_name = str(self.get_parameter("plug_name").value)
        t.port_type = "sfp"
        t.port_name = str(self.get_parameter("port_name").value)
        t.target_module_name = str(self.get_parameter("target_module_name").value)
        t.time_limit = 180
        return t

    def run_insertion(self) -> None:
        try:
            delay = float(self.get_parameter("startup_delay_sec").value)
            self.get_logger().info(f"Waiting {delay:.1f}s for TF tree ...")
            time.sleep(delay)

            if not self._change_target_mode(TargetMode.MODE_CARTESIAN):
                self.get_logger().error("Failed to switch to MODE_CARTESIAN; aborting.")
                return

            cheat = CheatCode(self)
            task = self._build_task()

            def _dummy_obs() -> Observation:
                return Observation()

            def _feedback(msg: str) -> None:
                self.get_logger().info(f"policy feedback: {msg}")

            self.get_logger().info(f"Starting CheatCode.insert_cable for task={task}")
            ok = cheat.insert_cable(task, _dummy_obs, self._move_robot, _feedback)
            self.get_logger().info(f"CheatCode.insert_cable finished, success={ok}")
        finally:
            rclpy.shutdown()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = CheatCodeGroundTruthTeleopNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    worker = threading.Thread(target=node.run_insertion, daemon=True)
    worker.start()

    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        worker.join(timeout=1.0)
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main(sys.argv)
