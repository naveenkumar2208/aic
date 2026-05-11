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
"""ACT policy for ROS inference.

Weights (first match): env ``AIC_ACT_POLICY_PRETRAINED_PATH``; ROS ``policy_pretrained_path``;
ROS ``policy_repo_id`` (+ optional ``policy_hf_revision``); else default ``grkw/aic_act_policy``.

After ``lerobot-train``, point at ``<output_dir>/checkpoints/last/pretrained_model`` (or a
specific step folder). Example::

    export AIC_ACT_POLICY_PRETRAINED_PATH=$PWD/outputs/train/act_run/checkpoints/last/pretrained_model
    ros2 run aic_model aic_model --ros-args -p policy:=aic_example_policies.ros.RunACT
"""

import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import json
import re
import time
import torch
import numpy as np
import cv2
import draccus
from pathlib import Path
from typing import Callable, Dict, Any, List
from rclpy.exceptions import ParameterAlreadyDeclaredException
from rclpy.node import Node
from geometry_msgs.msg import Twist, Vector3

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task

from aic_control_interfaces.msg import (
    MotionUpdate,
    TrajectoryGenerationMode,
)
from geometry_msgs.msg import Wrench

# LeRobot & Safetensors
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.configuration_act import ACTConfig
from safetensors.torch import load_file
from huggingface_hub import snapshot_download


def _cuda_runtime_ok() -> bool:
    """True if this PyTorch build can actually run ops on the current GPU (not just torch.cuda.is_available())."""
    if not torch.cuda.is_available():
        return False
    try:
        a = torch.randn(32, 32, device="cuda", dtype=torch.float32)
        b = torch.randn(32, 32, device="cuda", dtype=torch.float32)
        (a @ b).sum().item()
        torch.cuda.synchronize()
        return True
    except Exception:
        return False


def _normalize_pretrained_dir(path: Path) -> Path:
    """Accept either .../pretrained_model or .../checkpoints/<step> (parent of pretrained_model)."""
    if (path / "model.safetensors").is_file():
        return path.resolve()
    child = path / "pretrained_model"
    if (child / "model.safetensors").is_file():
        return child.resolve()
    raise FileNotFoundError(
        f"Expected config.json and model.safetensors under {path} or {child}."
    )


def _find_normalizer_stats_path(pretrained_dir: Path) -> Path:
    """LeRobot saves ``*_step_<N>_normalizer_processor.safetensors``; step index varies by pipeline."""
    candidates = list(pretrained_dir.glob("*_normalizer_processor.safetensors"))
    if not candidates:
        legacy = pretrained_dir / "policy_preprocessor_step_3_normalizer_processor.safetensors"
        if legacy.is_file():
            return legacy
        raise FileNotFoundError(
            f"No *normalizer_processor.safetensors under {pretrained_dir}. "
            "Train with lerobot-train and point policy_pretrained_path at "
            "checkpoints/<step>/pretrained_model (or checkpoints/last/pretrained_model)."
        )

    def step_index(p: Path) -> int:
        m = re.search(r"_step_(\d+)_", p.name)
        return int(m.group(1)) if m else -1

    return max(candidates, key=step_index)


def _resolve_pretrained_policy_dir(parent_node: Node, logger) -> Path:
    """Directory containing ACT ``config.json``, ``model.safetensors``, and preprocessor stats."""
    env_path = os.environ.get("AIC_ACT_POLICY_PRETRAINED_PATH", "").strip()
    if env_path:
        p = _normalize_pretrained_dir(Path(env_path).expanduser())
        logger.info(f"ACT weights from AIC_ACT_POLICY_PRETRAINED_PATH: {p}")
        return p

    try:
        parent_node.declare_parameter("policy_pretrained_path", "")
    except ParameterAlreadyDeclaredException:
        pass
    try:
        parent_node.declare_parameter("policy_repo_id", "")
    except ParameterAlreadyDeclaredException:
        pass
    try:
        parent_node.declare_parameter("policy_hf_revision", "")
    except ParameterAlreadyDeclaredException:
        pass

    local = parent_node.get_parameter("policy_pretrained_path").get_parameter_value().string_value.strip()
    if local:
        p = _normalize_pretrained_dir(Path(local).expanduser())
        logger.info(f"ACT weights from policy_pretrained_path: {p}")
        return p

    repo_id = parent_node.get_parameter("policy_repo_id").get_parameter_value().string_value.strip()
    if repo_id:
        revision = parent_node.get_parameter("policy_hf_revision").get_parameter_value().string_value.strip()
        revision_kw = {"revision": revision} if revision else {}
        logger.info(f"Downloading ACT weights from Hugging Face repo_id={repo_id!r} ...")
        p = Path(
            snapshot_download(
                repo_id=repo_id,
                allow_patterns=["*.json", "*.safetensors"],
                **revision_kw,
            )
        )
        p = _normalize_pretrained_dir(p)
        logger.info(f"ACT weights from Hub: {p}")
        return p

    default_repo = "grkw/aic_act_policy"
    logger.info(f"No policy path or repo set; using default Hub repo {default_repo!r}")
    p = Path(
        snapshot_download(
            repo_id=default_repo,
            allow_patterns=["config.json", "model.safetensors", "*.safetensors"],
        )
    )
    return _normalize_pretrained_dir(p)


def _resolve_act_device(
    parent_node: Node, logger
) -> torch.device:
    """Env ``AIC_RUN_ACT_DEVICE`` (cpu|cuda) overrides; else ROS param ``act_device`` (auto|cpu|cuda)."""
    env = os.environ.get("AIC_RUN_ACT_DEVICE", "").strip().lower()
    if env == "cpu":
        return torch.device("cpu")
    if env == "cuda":
        if _cuda_runtime_ok():
            return torch.device("cuda")
        logger.warning("AIC_RUN_ACT_DEVICE=cuda but CUDA is not usable; using CPU.")
        return torch.device("cpu")

    try:
        mode = parent_node.get_parameter("act_device").get_parameter_value().string_value
        mode = (mode or "auto").strip().lower()
    except Exception:
        mode = "auto"
    if mode not in ("auto", "cpu", "cuda"):
        mode = "auto"

    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        if _cuda_runtime_ok():
            return torch.device("cuda")
        logger.warning("act_device=cuda but CUDA is not usable; using CPU.")
        return torch.device("cpu")
    if _cuda_runtime_ok():
        return torch.device("cuda")
    logger.warning(
        "act_device=auto: CUDA is not usable with this PyTorch (e.g. RTX 50 / sm_120 "
        "with an older torch). Using CPU. For GPU, install a PyTorch build for your "
        "architecture, or set AIC_RUN_ACT_DEVICE=cpu to silence this message."
    )
    return torch.device("cpu")


class RunACT(Policy):
    def __init__(self, parent_node: Node):
        super().__init__(parent_node)
        try:
            parent_node.declare_parameter("act_device", "auto")
        except ParameterAlreadyDeclaredException:
            pass
        self.device = _resolve_act_device(parent_node, self.get_logger())

        # -------------------------------------------------------------------------
        # 1. Configuration & Weights Loading
        # -------------------------------------------------------------------------
        policy_path = _resolve_pretrained_policy_dir(parent_node, self.get_logger())

        # Load Config Manually (Fixes 'Draccus' error by removing unknown 'type' field)
        with open(policy_path / "config.json", "r") as f:
            config_dict = json.load(f)
            if "type" in config_dict:
                del config_dict["type"]

        config = draccus.decode(ACTConfig, config_dict)

        # Load Policy Architecture & Weights
        self.policy = ACTPolicy(config)
        model_weights_path = policy_path / "model.safetensors"
        self.policy.load_state_dict(load_file(model_weights_path))
        self.policy.eval()
        self.policy.to(self.device)

        self.get_logger().info(f"ACT Policy loaded on {self.device} from {policy_path}")

        # -------------------------------------------------------------------------
        # 2. Normalization Stats Loading
        # -------------------------------------------------------------------------
        stats_path = _find_normalizer_stats_path(policy_path)
        self.get_logger().info(f"Loading preprocessor stats from {stats_path.name}")
        stats = load_file(stats_path)

        # Helper to extract and shape stats for broadcasting
        def get_stat(key, shape):
            return stats[key].to(self.device).view(*shape)

        # Image Stats (1, 3, 1, 1) for broadcasting against (Batch, Channel, Height, Width)
        self.img_stats = {
            "left": {
                "mean": get_stat("observation.images.left_camera.mean", (1, 3, 1, 1)),
                "std": get_stat("observation.images.left_camera.std", (1, 3, 1, 1)),
            },
            "center": {
                "mean": get_stat("observation.images.center_camera.mean", (1, 3, 1, 1)),
                "std": get_stat("observation.images.center_camera.std", (1, 3, 1, 1)),
            },
            "right": {
                "mean": get_stat("observation.images.right_camera.mean", (1, 3, 1, 1)),
                "std": get_stat("observation.images.right_camera.std", (1, 3, 1, 1)),
            },
        }
        print(f"Image stats: {self.img_stats}")

        # Robot State Stats (1, 26)
        self.state_mean = get_stat("observation.state.mean", (1, -1))
        self.state_std = get_stat("observation.state.std", (1, -1))
        print(f"Robot state mean: {self.state_mean}")
        print(f"Robot state std: {self.state_std}")

        # Action Stats (1, 7) - Used for Un-normalization
        self.action_mean = get_stat("action.mean", (1, -1))
        self.action_std = get_stat("action.std", (1, -1))
        print(f"Action mean: {self.action_mean}")
        print(f"Action std: {self.action_std}")

        # Config
        self.image_scaling = 0.25  # Must match AICRobotAICControllerConfig

        self.get_logger().info("Normalization statistics loaded successfully.")

    @staticmethod
    def _img_to_tensor(
        raw_img,
        device: torch.device,
        scale: float,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        """Converts ROS Image -> Resized -> Permuted -> Normalized Tensor."""
        # 1. Bytes to Numpy (H, W, C)
        img_np = np.frombuffer(raw_img.data, dtype=np.uint8).reshape(
            raw_img.height, raw_img.width, 3
        )

        # 2. Resize
        if scale != 1.0:
            img_np = cv2.resize(
                img_np, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )

        # 3. To Tensor -> Permute (HWC -> CHW) -> Float -> Div(255) -> Batch Dim
        tensor = (
            torch.from_numpy(img_np)
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .unsqueeze(0)
            .to(device)
        )

        # 4. Normalize (Apply Mean/Std)
        # Formula: (x - mean) / std
        return (tensor - mean) / std

    def prepare_observations(self, obs_msg: Observation) -> Dict[str, torch.Tensor]:
        """Convert ROS Observation message into dictionary of normalized tensors."""

        # --- Process Cameras ---
        obs = {
            "observation.images.left_camera": self._img_to_tensor(
                obs_msg.left_image,
                self.device,
                self.image_scaling,
                self.img_stats["left"]["mean"],
                self.img_stats["left"]["std"],
            ),
            "observation.images.center_camera": self._img_to_tensor(
                obs_msg.center_image,
                self.device,
                self.image_scaling,
                self.img_stats["center"]["mean"],
                self.img_stats["center"]["std"],
            ),
            "observation.images.right_camera": self._img_to_tensor(
                obs_msg.right_image,
                self.device,
                self.image_scaling,
                self.img_stats["right"]["mean"],
                self.img_stats["right"]["std"],
            ),
        }

        # --- Process Robot State ---
        # Construct flat state vector (26 dims) matching training order
        tcp_pose = obs_msg.controller_state.tcp_pose
        tcp_vel = obs_msg.controller_state.tcp_velocity

        state_np = np.array(
            [
                # TCP Position (3)
                tcp_pose.position.x,
                tcp_pose.position.y,
                tcp_pose.position.z,
                # TCP Orientation (4)
                tcp_pose.orientation.x,
                tcp_pose.orientation.y,
                tcp_pose.orientation.z,
                tcp_pose.orientation.w,
                # TCP Linear Vel (3)
                tcp_vel.linear.x,
                tcp_vel.linear.y,
                tcp_vel.linear.z,
                # TCP Angular Vel (3)
                tcp_vel.angular.x,
                tcp_vel.angular.y,
                tcp_vel.angular.z,
                # TCP Error (6)
                *obs_msg.controller_state.tcp_error,
                # Joint Positions (7)
                *obs_msg.joint_states.position[:7],
            ],
            dtype=np.float32,
        )

        # Normalize State
        raw_state_tensor = (
            torch.from_numpy(state_np).float().unsqueeze(0).to(self.device)
        )
        obs["observation.state"] = (raw_state_tensor - self.state_mean) / self.state_std

        return obs

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        **kwargs,
    ):
        self.policy.reset()
        self.get_logger().info(f"RunACT.insert_cable() enter. Task: {task}")

        start_time = time.time()

        # Run inference for 30 seconds
        while time.time() - start_time < 30.0:
            loop_start = time.time()

            # 1. Get & Process Observation
            observation_msg = get_observation()

            if observation_msg is None:
                self.get_logger().info("No observation received.")
                continue

            obs_tensors = self.prepare_observations(observation_msg)

            # 2. Model Inference
            with torch.inference_mode():
                # returns shape [1, 7] (first action of chunk)
                normalized_action = self.policy.select_action(obs_tensors)

            # 3. Un-normalize Action
            # Formula: (norm * std) + mean
            raw_action_tensor = (normalized_action * self.action_std) + self.action_mean

            # 4. Extract and Command
            # raw_action_tensor is [1, 7], taking [0] gives vector of 7
            action = raw_action_tensor[0].cpu().numpy()

            self.get_logger().info(f"Action: {action}")

            twist = Twist(
                linear=Vector3(
                    x=float(action[0]), y=float(action[1]), z=float(action[2])
                ),
                angular=Vector3(
                    x=float(action[3]), y=float(action[4]), z=float(action[5])
                ),
            )
            motion_update = self.set_cartesian_twist_target(twist)
            move_robot(motion_update=motion_update)
            send_feedback("in progress...")

            # Maintain control rate (approx 4Hz loop = 0.25s sleep)
            elapsed = time.time() - loop_start
            time.sleep(max(0, 0.25 - elapsed))

        self.get_logger().info("RunACT.insert_cable() exiting...")
        return True

    def set_cartesian_twist_target(self, twist: Twist, frame_id: str = "base_link"):
        motion_update_msg = MotionUpdate()
        motion_update_msg.velocity = twist
        motion_update_msg.header.frame_id = frame_id
        motion_update_msg.header.stamp = self.get_clock().now().to_msg()

        motion_update_msg.target_stiffness = np.diag(
            [100.0, 100.0, 100.0, 50.0, 50.0, 50.0]
        ).flatten()
        motion_update_msg.target_damping = np.diag(
            [40.0, 40.0, 40.0, 15.0, 15.0, 15.0]
        ).flatten()

        motion_update_msg.feedforward_wrench_at_tip = Wrench(
            force=Vector3(x=0.0, y=0.0, z=0.0), torque=Vector3(x=0.0, y=0.0, z=0.0)
        )

        motion_update_msg.wrench_feedback_gains_at_tip = [0.5, 0.5, 0.5, 0.0, 0.0, 0.0]

        motion_update_msg.trajectory_generation_mode.mode = (
            TrajectoryGenerationMode.MODE_VELOCITY
        )

        return motion_update_msg
