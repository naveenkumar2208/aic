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
"""ACT policy for checkpoints trained with ``gripper/tcp`` velocity labels.

Use this policy when ``lerobot-record`` was run with::

    --robot.teleop_target_mode=cartesian
    --robot.teleop_frame_id=gripper/tcp

This module reuses the full LeRobot ACT loading / preprocessing pipeline from
``RunACT`` and only changes the outgoing ``MotionUpdate.header.frame_id`` so the
controller interprets the predicted twist in the same frame the dataset used.
"""

from geometry_msgs.msg import Twist
from rclpy.node import Node

from .RunACT import RunACT as _RunACTBase


class RunACTGripperTCP(_RunACTBase):
    """Run ACT with Cartesian velocity commands in ``gripper/tcp`` frame."""

    def __init__(self, parent_node: Node):
        super().__init__(parent_node)
        self.get_logger().info(
            "RunACTGripperTCP using MotionUpdate.header.frame_id='gripper/tcp'"
        )

    def set_cartesian_twist_target(
        self, twist: Twist, frame_id: str = "gripper/tcp"
    ):
        return super().set_cartesian_twist_target(twist, frame_id=frame_id)
