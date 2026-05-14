#!/usr/bin/env python3
#
# Copyright (C) 2026 Intrinsic Innovation LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# Per-episode reset (default on), aligned with aic_engine trial cleanup + scene docs:
#   https://github.com/intrinsic-dev/aic/blob/main/docs/scene_description.md
#   https://github.com/intrinsic-dev/aic/blob/main/aic_bringup/README.md
#
# Important — what is NOT respawned each episode:
#   The UR5e Gazebo model is created once by aic_gz_bringup (ros_gz_sim create, name "ur5e").
#   This script does not delete or respawn that robot; only task_board* and cable_0* are
#   removed via /gz_server/delete_entity (same idea as programmatic resets in scene_description).
#
# What IS reset each episode (default):
#   1) Delete Gazebo task board + cable entities
#   2) Optional --pre-spawn-shell
#   3) Home the arm: deactivate aic_controller -> /scoring/reset_joints (Gazebo ResetJointsPlugin)
#      -> reactivate aic_controller (same sequence as aic_engine::home_robot)
#   4) spawn_task_board + spawn_cable -> tare -> lerobot-record
#
# Homing uses the same /scoring/reset_joints + controller switch sequence as
# aic_engine::home_robot() (see aic_engine.cpp), with joint targets from
# aic_engine/config/sample_config.yaml under robot.home_joint_positions.
#
# =============================================================================
# HOW TO RUN (two terminals, same ROS 2 / Zenoh domain)
# =============================================================================
#
# Terminal A — eval container (sim stays up; no aic_engine so CheatCode + record stay manual):
#
#   /entrypoint.sh ground_truth:=true start_aic_engine:=false launch_rviz:=false
#
# Terminal B — from this repo (example paths; adjust install to your machine or /ws_aic in container):
#
#   cd ~/ws_aic/src/aic
#   export RMW_IMPLEMENTATION=rmw_zenoh_cpp
#   pixi run python aic_utils/lerobot_robot_aic/scripts/auto_record_cheatcode_loop.py \
#     --use-pixi \
#     --install-setup ~/ws_aic/install/setup.bash \
#     --dataset-root "${HOME}/lerobot_datasets/aic_sfp_insert_v001" \
#     --repo-id local/aic_sfp_insert \
#     --board-x0 0.2 --board-y0 -0.2
#
# If ROS runs only inside the same container as Terminal A, use that install path, e.g.
#   --install-setup /ws_aic/install/setup.bash
#
# Optional: --episode-mode training333 --max-episodes 99  (first 99: three fixed trials 1/3 each;
#   then continues forever with random board/spawn like --episode-mode random, but 2/3 nic_card_mount
#   vs 1/3 sc_port episodes)
#           --on-home-failure exit --home-failure-shell 'your_restart_snippet.sh'
# Focus the lerobot-record terminal for pynput Right/Left keys.
#
# See: https://github.com/intrinsic-dev/aic/blob/main/aic_bringup/README.md
#      https://github.com/intrinsic-dev/aic/blob/main/docs/scene_description.md
#      https://github.com/intrinsic-dev/aic/blob/main/aic_engine/README.md

from __future__ import annotations

import argparse
import dataclasses
import math
import os
import random
import re
import select
import shutil
import subprocess
import sys
import time


def _bool_args(prefix: str, n: int, true_index: int | None) -> list[str]:
    out: list[str] = []
    for i in range(n):
        v = "true" if true_index is not None and i == true_index else "false"
        out.append(f"{prefix}_{i}_present:={v}")
    return out


def _nic_spawn_args(mount_index: int) -> list[str]:
    nic = _bool_args("nic_card_mount", 5, mount_index)
    sc = _bool_args("sc_port", 2, None)
    return nic + sc


def _sc_spawn_args(port_index: int) -> list[str]:
    nic = _bool_args("nic_card_mount", 5, None)
    sc = _bool_args("sc_port", 2, port_index)
    return nic + sc


def _randomize_board_pose(
    rng: random.Random, x0: float, y0: float, z0: float, dxy: float
) -> tuple[float, float, float, float]:
    """Randomize x,y within ±dxy of (x0,y0); z fixed at z0; yaw uniform in [-pi, pi]."""
    x = x0 + rng.uniform(-dxy, dxy)
    y = y0 + rng.uniform(-dxy, dxy)
    z = z0
    yaw = rng.uniform(-math.pi, math.pi)
    return x, y, z, yaw


# Cable initial pose — match aic_gz_bringup.launch.py (not spawn_cable.launch.py defaults).
# README: use cable_z:=1.508 when cable_type is sfp_sc_cable_reversed.
def _cable_spawn_params(cable_type: str) -> tuple[tuple[str, float], ...]:
    cz = 1.508 if cable_type == "sfp_sc_cable_reversed" else 1.518
    return (
        ("cable_x", 0.172),
        ("cable_y", 0.024),
        ("cable_z", cz),
        ("cable_roll", 0.4432),
        ("cable_pitch", -0.48),
        ("cable_yaw", 1.3303),
    )


@dataclasses.dataclass(frozen=True)
class TrainingTrialRecipe:
    """Fixed spawn + teleop for one of three training scenarios (1/3 sampling => 2/3 NIC, 1/3 SC)."""

    trial_id: int
    board_x: float
    board_y: float
    board_z: float
    board_yaw: float
    mount_launch_args: tuple[str, ...]
    cable_type: str
    teleop_module: str
    teleop_port: str
    task_plug: str
    single_task: str


# Mirrors user trial 1/2/3 entrypoint snippets (NIC mount 0, NIC mount 1, SC port 1 + rails).
_TRAINING333_TRIALS: tuple[TrainingTrialRecipe, ...] = (
    TrainingTrialRecipe(
        trial_id=1,
        board_x=0.15,
        board_y=-0.2,
        board_z=1.14,
        board_yaw=3.1415,
        mount_launch_args=(
            "nic_card_mount_0_present:=true",
            "nic_card_mount_0_translation:=0.036",
            "sc_port_0_present:=true",
            "sc_port_0_translation:=0.042",
            "lc_mount_rail_0_present:=true",
            "lc_mount_rail_0_translation:=0.02",
            "sfp_mount_rail_0_present:=true",
            "sfp_mount_rail_0_translation:=0.03",
            "sc_mount_rail_0_present:=true",
            "sc_mount_rail_0_translation:=-0.02",
            "lc_mount_rail_1_present:=true",
            "lc_mount_rail_1_translation:=-0.01",
        ),
        cable_type="sfp_sc_cable",
        teleop_module="nic_card_mount_0",
        teleop_port="sfp_port_0",
        task_plug="sfp_tip",
        single_task="Insert SFP into NIC port",
    ),
    TrainingTrialRecipe(
        trial_id=2,
        board_x=0.15,
        board_y=-0.2,
        board_z=1.14,
        board_yaw=3.1415,
        mount_launch_args=(
            "nic_card_mount_1_present:=true",
            "nic_card_mount_1_translation:=0.036",
            "sc_port_0_present:=true",
            "sc_port_0_translation:=0.042",
            "lc_mount_rail_0_present:=true",
            "lc_mount_rail_0_translation:=0.02",
            "sfp_mount_rail_0_present:=true",
            "sfp_mount_rail_0_translation:=0.03",
            "sc_mount_rail_0_present:=true",
            "sc_mount_rail_0_translation:=-0.02",
            "lc_mount_rail_1_present:=true",
            "lc_mount_rail_1_translation:=-0.01",
        ),
        cable_type="sfp_sc_cable",
        teleop_module="nic_card_mount_1",
        teleop_port="sfp_port_0",
        task_plug="sfp_tip",
        single_task="Insert SFP into NIC port",
    ),
    TrainingTrialRecipe(
        trial_id=3,
        board_x=0.17,
        board_y=0.0,
        board_z=1.14,
        board_yaw=3.0,
        mount_launch_args=(
            "sc_port_1_present:=true",
            "sc_port_1_translation:=-0.055",
            "sfp_mount_rail_0_present:=true",
            "sfp_mount_rail_0_translation:=0.05",
            "sc_mount_rail_0_present:=true",
            "sc_mount_rail_0_translation:=-0.03",
            "lc_mount_rail_1_present:=true",
            "lc_mount_rail_1_translation:=0.04",
        ),
        cable_type="sfp_sc_cable_reversed",
        teleop_module="sc_port_1",
        teleop_port="sc_port_base",
        task_plug="sc_tip",
        single_task="Insert SC into port",
    ),
)


def _build_spawn_shell(
    setup_bash: str,
    x: float,
    y: float,
    z: float,
    yaw: float,
    mount_args: list[str],
    cable_type: str,
    spawn_cable: bool,
) -> str:
    """spawn_task_board.launch.py only creates the board; cable uses spawn_cable.launch.py."""
    board_launch = " ".join(
        [
            "ros2 launch aic_bringup spawn_task_board.launch.py",
            f"task_board_x:={x:.6f}",
            f"task_board_y:={y:.6f}",
            f"task_board_z:={z:.6f}",
            "task_board_roll:=0.0",
            "task_board_pitch:=0.0",
            f"task_board_yaw:={yaw:.6f}",
            *mount_args,
        ]
    )
    parts = [f"source {sh_quote(setup_bash)}", board_launch]
    if spawn_cable:
        cable_args = [
            "ros2 launch aic_bringup spawn_cable.launch.py",
            "attach_cable_to_gripper:=true",
            f"cable_type:={cable_type}",
        ]
        for name, val in _cable_spawn_params(cable_type):
            cable_args.append(f"{name}:={val:.6f}")
        parts.append(" ".join(cable_args))
    return " && ".join(parts)


def sh_quote(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"


# UR arm home pose — must match `robot.home_joint_positions` in
# aic_engine/config/sample_config.yaml (order: pan, lift, elbow, wrist_1–3).
_HOME_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
_HOME_JOINT_POSITIONS = (-0.1597, -1.3542, -1.6648, -1.6933, 1.5710, 1.4110)


def _ros2_exec_prefix(pixi: bool) -> str:
    return "pixi run ros2 " if pixi else "ros2 "


def _home_robot_shell(
    setup_bash: str,
    pixi: bool,
    sleep_after_deactivate: float,
    sleep_after_reset: float,
) -> str:
    """Bash snippet: controller switch + /scoring/reset_joints like aic_engine::home_robot."""
    r2 = _ros2_exec_prefix(pixi)
    names_literal = str(list(_HOME_JOINT_NAMES)).replace("'", '"')
    pos_literal = str(list(_HOME_JOINT_POSITIONS))
    sd = max(0.0, sleep_after_deactivate)
    sr = max(0.0, sleep_after_reset)
    # strictness 1 = BEST_EFFORT (matches aic_engine::home_robot). timeout must be a Duration
    # dict (Kilted/Rolling); float 0.0 breaks ros2 service call with:
    # "Value '0.0' is expected to be a dictionary but is a float".
    switch_off = (
        f"{r2}service call /controller_manager/switch_controller "
        "controller_manager_msgs/srv/SwitchController "
        "'{activate_controllers: [], deactivate_controllers: [aic_controller], "
        "strictness: 1, activate_asap: true, timeout: {sec: 0, nanosec: 0}}'"
    )
    reset_j = (
        f"{r2}service call /scoring/reset_joints aic_engine_interfaces/srv/ResetJoints "
        f"'{{joint_names: {names_literal}, initial_positions: {pos_literal}}}'"
    )
    switch_on = (
        f"{r2}service call /controller_manager/switch_controller "
        "controller_manager_msgs/srv/SwitchController "
        "'{activate_controllers: [aic_controller], deactivate_controllers: [], "
        "strictness: 1, activate_asap: true, timeout: {sec: 0, nanosec: 0}}'"
    )
    return (
        f"source {sh_quote(setup_bash)} && "
        f"{switch_off} && sleep {sd:.3f} && {reset_j} && sleep {sr:.3f} && {switch_on}"
    )


# Gazebo model names from aic_bringup spawn_task_board / spawn_cable (allow_renaming may add _0).
_DEFAULT_DELETE_ENTITY_NAMES: tuple[str, ...] = (
    "task_board",
    "task_board_0",
    "task_board_1",
    "cable_0",
    "cable_0_0",
    "cable_0_1",
)


def _delete_entities_shell(
    setup_bash: str, pixi: bool, names: tuple[str, ...]
) -> str:
    """Bash snippet: delete_entity for each name (ignore failures), like aic_engine reset_simulator."""
    r2 = _ros2_exec_prefix(pixi)
    parts: list[str] = [f"source {sh_quote(setup_bash)}", "set +e"]
    for n in names:
        safe = n.replace("'", "")
        req = "{entity: '%s'}" % safe
        parts.append(
            f"{r2}service call /gz_server/delete_entity "
            "simulation_interfaces/srv/DeleteEntity "
            f"{repr(req)} || true"
        )
        parts.append("sleep 0.25")
    return " && ".join(parts)


def _tare_cmd(pixi: str | None) -> list[str]:
    base = [
        "ros2",
        "service",
        "call",
        "/aic_controller/tare_force_torque_sensor",
        "std_srvs/srv/Trigger",
        "{}",
    ]
    if pixi:
        return ["pixi", "run", *base]
    return base


def _drain_subprocess_stdout(proc: subprocess.Popen, tee: bool = True) -> None:
    """Read remaining child stdout (after an early break from the main read loop).

    If we stop reading mid-stream, lerobot-record can block when the pipe buffer fills;
    draining also lets encoding / 'Stop recording' logs finish before wait().
    """
    if proc.stdout is None:
        return
    try:
        for line in proc.stdout:
            if tee:
                sys.stdout.write(line)
                sys.stdout.flush()
    except (BrokenPipeError, ValueError, TypeError):
        pass


def _wait_before_success_right(proc: subprocess.Popen, delay_s: float) -> None:
    """Wait delay_s (wall clock) before sending Right; tee stdout when select() works on the pipe."""
    if delay_s <= 0.0:
        return
    end = time.monotonic() + delay_s
    use_select = False
    if proc.stdout is not None:
        try:
            proc.stdout.fileno()
            use_select = True
        except (AttributeError, OSError):
            use_select = False
    while time.monotonic() < end:
        remaining = end - time.monotonic()
        if remaining <= 0.0:
            break
        if proc.stdout is not None and use_select:
            try:
                rlist, _, _ = select.select(
                    [proc.stdout], [], [], min(0.25, remaining)
                )
            except (ValueError, OSError):
                time.sleep(min(0.25, remaining))
                continue
            if proc.stdout in rlist:
                line = proc.stdout.readline()
                if line:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                else:
                    time.sleep(min(0.05, remaining))
        else:
            time.sleep(min(0.25, remaining))


def _lerobot_record_cmd(
    pixi: str | None,
    dataset_root: str,
    repo_id: str,
    teleop_module: str,
    teleop_port: str,
    task_plug: str,
    single_task: str,
    resume: bool,
) -> list[str]:
    cmd = [
        "lerobot-record",
        "--dataset.root",
        dataset_root,
        "--dataset.repo_id",
        repo_id,
        "--robot.type=aic_controller",
        "--robot.id=aic",
        "--teleop.type=aic_cheatcode",
        "--teleop.id=aic",
        "--robot.teleop_target_mode=cartesian",
        "--robot.teleop_frame_id=gripper/tcp",
        f"--teleop.task_module_name={teleop_module}",
        f"--teleop.task_port_name={teleop_port}",
        f"--teleop.task_plug_name={task_plug}",
        "--teleop.task_cable_name=cable_0",
        f"--dataset.single_task={single_task}",
        "--dataset.num_episodes=1",
        "--dataset.fps=10",
        "--dataset.push_to_hub=false",
        "--play_sounds=false",
        "--display_data=false",
    ]
    if resume:
        cmd.insert(1, "--resume=true")
    if pixi:
        return ["pixi", "run", *cmd]
    return cmd


def _press_key(key_name: str) -> None:
    """Send a key to the active X/Wayland session (lerobot-record / pynput listener)."""
    try:
        from pynput.keyboard import Controller, Key

        key = Key.right if key_name == "right" else Key.left
        Controller().tap(key)
        print(f"[auto_record] Sent {key_name} arrow via pynput", flush=True)
    except Exception as e:
        print(
            f"[auto_record] Could not inject {key_name} key ({e}). "
            f"Press {key_name.upper()} manually in the lerobot-record terminal.",
            flush=True,
        )


def _episode_teleop_config(
    kind: str, index: int, nic_plug: str, sc_plug: str
) -> tuple[str, str, str, str, str]:
    """Returns (task_module, task_port, task_plug, cable_type, single_task)."""
    if kind == "nic":
        return (
            f"nic_card_mount_{index}",
            "sfp_port_0",
            nic_plug,
            "sfp_sc_cable",
            "Insert SFP into NIC port",
        )
    return (
        f"sc_port_{index}",
        "sc_port_base",
        sc_plug,
        "sfp_sc_cable_reversed",
        "Insert SC into port",
    )


_SUCCESS_PATTERNS = (
    "Phase: DONE",
    "Insertion complete after",
    "[UNIVERSAL] Insertion complete",
)

_CHEATCODE_PLUG_Z_RE = re.compile(
    r"plug_z_actual:\s*([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)"
)
_CHEATCODE_FORCE_N_RE = re.compile(r"Force:\s*([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)\s*N")
_CHEATCODE_PHASE_RE = re.compile(r"Phase:\s*(\w+)")


def _is_success_line(line: str) -> bool:
    return any(p in line for p in _SUCCESS_PATTERNS)


def _parse_plug_z_actual(line: str) -> float | None:
    """Parse plug_z_actual from a CheatCode status line (see aic_teleop.py _obs_callback)."""
    if "[CheatCode]" not in line:
        return None
    m = _CHEATCODE_PLUG_Z_RE.search(line)
    if not m:
        return None
    return float(m.group(1))


def _nic_plug_z_depth_success(
    line: str, z_at_most: float, require_insert_phase: bool
) -> bool:
    """NIC: success when plug_z_actual <= z_at_most (e.g. fully seated around -0.0009)."""
    if require_insert_phase and "Phase: INSERT" not in line:
        return False
    z = _parse_plug_z_actual(line)
    if z is None:
        return False
    return z <= z_at_most


def _parse_cheatcode_force_n(line: str) -> float | None:
    """Parse Force (Newtons) from a CheatCode _obs_callback line."""
    if "[CheatCode]" not in line:
        return None
    m = _CHEATCODE_FORCE_N_RE.search(line)
    if not m:
        return None
    return float(m.group(1))


def _parse_cheatcode_phase(line: str) -> str | None:
    if "[CheatCode]" not in line:
        return None
    m = _CHEATCODE_PHASE_RE.search(line)
    return m.group(1) if m else None


def _cheatcode_force_abort(
    line: str,
    threshold_n: float,
    phases: frozenset[str] | None,
) -> bool:
    """True if we should discard: force above threshold in an allowed phase.

    phases is None (from --abort-force-phases ALL) => any CheatCode phase matches.
    """
    if threshold_n <= 0.0:
        return False
    phase = _parse_cheatcode_phase(line)
    if phase is None:
        return False
    if phases is not None and phase not in phases:
        return False
    f = _parse_cheatcode_force_n(line)
    if f is None:
        return False
    # >= so "20N limit" includes exactly 20.0N (CheatCode logs one decimal).
    return f >= threshold_n


def _parse_abort_force_phases(s: str) -> frozenset[str] | None:
    raw = s.strip()
    if raw.upper() == "ALL":
        return None
    parts = frozenset(p.strip() for p in raw.split(",") if p.strip())
    if not parts:
        return frozenset({"APPROACH", "ALIGN"})
    return parts


def _run_shell(script: str, env: dict[str, str]) -> None:
    print(f"[auto_record] Running: {script}", flush=True)
    subprocess.run(["bash", "-lc", script], env=env, check=True)


def _run_home_robot_with_retries(
    setup: str,
    pixi: bool,
    env: dict[str, str],
    retries: int,
    sleep_after_deactivate: float,
    sleep_after_reset: float,
) -> bool:
    """Run homing shell; return True if any attempt succeeds (matches aic_engine::home_robot)."""
    for attempt in range(1, retries + 1):
        try:
            _run_shell(
                _home_robot_shell(
                    setup, bool(pixi), sleep_after_deactivate, sleep_after_reset
                ),
                env,
            )
            print(f"[auto_record] home_robot succeeded (attempt {attempt}/{retries})", flush=True)
            return True
        except subprocess.CalledProcessError as e:
            print(
                f"[auto_record] home_robot attempt {attempt}/{retries} failed: {e}. "
                "Check controller_manager, /scoring/reset_joints (ResetJointsPlugin in sim), "
                "and that ROS matches the sim. If reset_joints returns busy, increase "
                "--home-sleep-after-reset.",
                flush=True,
            )
            if attempt < retries:
                time.sleep(2.0)
    print(
        "[auto_record] home_robot: all attempts failed; continuing without homing "
        "(arm may stay in the previous episode pose).",
        flush=True,
    )
    return False


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Infinite CheatCode + lerobot-record loop with randomized task board."
    )
    ap.add_argument(
        "--install-setup",
        required=True,
        help="Path to workspace install/setup.bash (sourced before ros2 launch).",
    )
    ap.add_argument(
        "--dataset-root",
        default=os.environ.get("LEROBOT_DATASET_ROOT", ""),
        help="LeRobot dataset root (directory containing meta/).",
    )
    ap.add_argument(
        "--repo-id",
        default=os.environ.get("LEROBOT_REPO_ID", "local/aic_sfp_insert"),
    )
    ap.add_argument(
        "--use-pixi",
        action="store_true",
        help="Prefix ros2 / lerobot-record with `pixi run` (run from src/aic).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for board pose and NIC vs SC episode selection.",
    )
    ap.add_argument(
        "--reset-gazebo-entities",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Before each episode, call /gz_server/delete_entity for names in "
        "--delete-entity-names (same service as aic_engine). Use --no-reset-gazebo-entities to skip.",
    )
    ap.add_argument(
        "--delete-entity-names",
        default=",".join(_DEFAULT_DELETE_ENTITY_NAMES),
        help="Comma-separated Gazebo entity names to delete between episodes.",
    )
    ap.add_argument(
        "--pre-spawn-shell",
        default="",
        help="Optional bash snippet after entity delete, before homing (extra world cleanup).",
    )
    ap.add_argument(
        "--max-episode-seconds",
        type=float,
        default=600.0,
        help="After this duration without success, send Left and kill record.",
    )
    ap.add_argument(
        "--nic-plug-name",
        default="sfp_tip",
        help="CheatCode task_plug_name for NIC episodes.",
    )
    ap.add_argument(
        "--nic-success-plug-z-at-most",
        type=float,
        default=-0.0009,
        help="NIC episodes only: save (Right) when a [CheatCode] line shows plug_z_actual <= this value "
        "while in INSERT (see aic_teleop CheatCode logs). Still succeeds on Phase: DONE if reached first.",
    )
    ap.add_argument(
        "--nic-success-disable-plug-z",
        action="store_true",
        help="NIC only: ignore plug_z_actual early success; wait for Phase: DONE / insertion strings only.",
    )
    ap.add_argument(
        "--nic-success-any-phase",
        action="store_true",
        help="NIC only: allow plug_z threshold outside INSERT (default: require Phase: INSERT).",
    )
    ap.add_argument(
        "--abort-on-force-greater-than",
        type=float,
        default=20.0,
        help="When CheatCode logs Force (N) >= this on a line with a phase, discard the episode: "
        "best-effort Left, then terminate lerobot-record (piped stdout is often fully buffered "
        "without PYTHONUNBUFFERED=1 on the child; Left from another process may not reach "
        "lerobot-record's listener). Next loop iteration resets spawn/home per script. Scoped by "
        "--abort-force-phases (default ALL phases).",
    )
    ap.add_argument(
        "--no-abort-on-high-cheatcode-force",
        action="store_true",
        help="Disable CheatCode force-based discard.",
    )
    ap.add_argument(
        "--abort-force-phases",
        default="ALL",
        help="Comma-separated CheatCode phases for force abort (e.g. APPROACH,ALIGN), or ALL for "
        "every phase including INSERT. Default ALL: any force above --abort-on-force-greater-than "
        "discards the episode (Left). Narrow phases if you only want early-phase aborts.",
    )
    ap.add_argument(
        "--sc-plug-name",
        default="sc_tip",
        help="CheatCode task_plug_name for SC episodes.",
    )
    ap.add_argument(
        "--no-resume",
        action="store_true",
        help="Omit --resume=true on lerobot-record.",
    )
    ap.add_argument(
        "--board-x0",
        type=float,
        default=0.25,
        help="Center X for randomization (spawn_task_board default is 0.25).",
    )
    ap.add_argument(
        "--board-y0",
        type=float,
        default=0.0,
        help="Center Y for randomization (spawn_task_board default is 0.0).",
    )
    ap.add_argument(
        "--board-z",
        type=float,
        default=1.14,
        help="Fixed task board Z (meters).",
    )
    ap.add_argument(
        "--board-xy-jitter",
        type=float,
        default=0.15,
        help="Uniform jitter ± this value on X and Y around the centers. Large values can push "
        "the board under the robot base in XY; try 0.05–0.15 if you see clipping.",
    )
    ap.add_argument(
        "--spawn-cable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After spawn_task_board, run spawn_cable.launch.py (required for cable in sim). "
        "Use --no-spawn-cable to debug board-only.",
    )
    ap.add_argument(
        "--home-robot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After deletes / pre-spawn, home the UR arm like aic_engine::home_robot (controller "
        "switch + /scoring/reset_joints). Use --no-home-robot to disable.",
    )
    ap.add_argument(
        "--home-retries",
        type=int,
        default=3,
        help="Retries for homing if ros2 service calls fail (e.g. transient controller switch).",
    )
    ap.add_argument(
        "--home-sleep-after-deactivate",
        type=float,
        default=2.0,
        help="Seconds to sleep after deactivating aic_controller before /scoring/reset_joints.",
    )
    ap.add_argument(
        "--home-sleep-after-reset",
        type=float,
        default=3.0,
        help="Seconds to sleep after reset_joints before reactivating aic_controller "
        "(Gazebo applies joint reset on PreUpdate; too short can race the next command).",
    )
    ap.add_argument(
        "--post-episode-sleep",
        type=float,
        default=4.0,
        help="Seconds to wait after each lerobot-record exits before the next episode "
        "(lets controllers and sim settle).",
    )
    ap.add_argument(
        "--success-right-arrow-delay",
        type=float,
        default=1.0,
        help="Seconds after success is detected before sending Right (save). "
        "While waiting, stdout is read when select() works on the pipe (Unix) to avoid backpressure.",
    )
    ap.add_argument(
        "--episode-mode",
        choices=("random", "training333"),
        default="random",
        help="random: 50/50 NIC vs SC + board jitter. training333: each episode picks trial 1, 2, or 3 "
        "with equal probability (2/3 NIC-style trials + 1/3 SC trial) using fixed board/rails/cable "
        "from the three training recipes embedded in this script. If --max-episodes is set with "
        "training333, that many curriculum episodes run first, then recording continues in random "
        "layout with 2/3 NIC vs 1/3 SC.",
    )
    ap.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="With --episode-mode random: exit after this many episodes. "
        "With --episode-mode training333: run this many fixed-trial episodes, then keep recording "
        "with random board/spawn (same as random mode) but each episode is 2/3 nic_card_mount vs "
        "1/3 sc_port (no exit). Default: run forever in the chosen mode.",
    )
    ap.add_argument(
        "--on-home-failure",
        choices=("continue", "exit"),
        default="continue",
        help="If all homing retries fail: continue (arm may stay in last pose) or exit with code 3 "
        "so a supervisor can restart Gazebo / the eval container.",
    )
    ap.add_argument(
        "--home-failure-shell",
        default="",
        help="Optional bash snippet run after homing fails (before --on-home-failure), e.g. wrap "
        "a container restart or notify script.",
    )
    args = ap.parse_args()

    if not args.dataset_root:
        ap.error("--dataset-root (or LEROBOT_DATASET_ROOT) is required")

    setup = os.path.abspath(os.path.expanduser(args.install_setup))
    if not os.path.isfile(setup):
        print(f"ERROR: --install-setup not found: {setup}", file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    delete_names = tuple(
        x.strip() for x in args.delete_entity_names.split(",") if x.strip()
    )
    pixi = True if args.use_pixi else None
    if args.use_pixi and not shutil.which("pixi"):
        print("ERROR: --use-pixi set but `pixi` not in PATH", file=sys.stderr)
        return 2

    env = os.environ.copy()
    if "RMW_IMPLEMENTATION" not in env:
        env["RMW_IMPLEMENTATION"] = "rmw_zenoh_cpp"

    cheatcode_force_abort_phases = _parse_abort_force_phases(args.abort_force_phases)

    print(
        "\n=== AIC auto-record loop ===\n"
        "1) Start simulation once (example, inside eval container):\n"
        "   /entrypoint.sh ground_truth:=true start_aic_engine:=false launch_rviz:=false\n"
        "2) Each episode (default): delete board/cable -> optional pre-spawn shell -> home arm "
        "(joints via /scoring/reset_joints; the ur5e Gazebo model is not respawned) -> "
        "spawn_task_board + spawn_cable -> tare -> lerobot-record.\n"
        "3) Focus the terminal running lerobot-record so pynput can send Right/Left.\n"
        "4) If deletes miss a model name, extend --delete-entity-names or add --pre-spawn-shell.\n"
        "5) SwitchController timeout uses builtin_interfaces/Duration (required on Kilted+).\n"
        "6) If the arm stays in the previous pose, check logs for home_robot failures; tune "
        "--home-sleep-after-deactivate / --home-sleep-after-reset / --post-episode-sleep.\n",
        flush=True,
    )

    episode = 0
    while True:
        episode += 1
        use_training333_curriculum = args.episode_mode == "training333" and (
            args.max_episodes is None or episode <= args.max_episodes
        )
        if use_training333_curriculum:
            trial = rng.choice(_TRAINING333_TRIALS)
            x, y, z, yaw = trial.board_x, trial.board_y, trial.board_z, trial.board_yaw
            mount_args = list(trial.mount_launch_args)
            cable = trial.cable_type
            mod = trial.teleop_module
            port = trial.teleop_port
            plug = trial.task_plug
            task = trial.single_task
            print(
                f"\n--- Episode {episode}: training333 trial_id={trial.trial_id} "
                f"board=({x:.3f},{y:.3f},{z:.3f}) yaw={yaw:.3f} cable={cable} ---",
                flush=True,
            )
        else:
            post333_weighted = (
                args.episode_mode == "training333"
                and args.max_episodes is not None
                and episode > args.max_episodes
            )
            if post333_weighted and episode == args.max_episodes + 1:
                print(
                    "[auto_record] Completed fixed-trial curriculum (--max-episodes). "
                    "Continuing with random board/spawn; task mix 2/3 nic_card_mount, 1/3 sc_port.",
                    flush=True,
                )
            if post333_weighted:
                kind = "nic" if rng.random() < (2.0 / 3.0) else "sc"
            else:
                kind = rng.choice(["nic", "sc"])
            idx = rng.randint(0, 4) if kind == "nic" else rng.randint(0, 1)
            x, y, z, yaw = _randomize_board_pose(
                rng, args.board_x0, args.board_y0, args.board_z, args.board_xy_jitter
            )
            mod, port, plug, cable, task = _episode_teleop_config(
                kind, idx, args.nic_plug_name, args.sc_plug_name
            )
            mount_args = _nic_spawn_args(idx) if kind == "nic" else _sc_spawn_args(idx)
            mix = "2/3nic" if post333_weighted else "50/50"
            print(
                f"\n--- Episode {episode}: {kind} index={idx} board=({x:.3f},{y:.3f},{z:.3f}) "
                f"yaw={yaw:.3f} mix={mix} ---",
                flush=True,
            )

        if args.reset_gazebo_entities and delete_names:
            try:
                _run_shell(_delete_entities_shell(setup, bool(pixi), delete_names), env)
            except subprocess.CalledProcessError as e:
                print(f"[auto_record] delete_entity batch failed (continuing): {e}", flush=True)

        if args.pre_spawn_shell.strip():
            _run_shell(args.pre_spawn_shell.strip(), env)

        home_ok = True
        if args.home_robot:
            home_ok = _run_home_robot_with_retries(
                setup,
                bool(pixi),
                env,
                max(1, args.home_retries),
                args.home_sleep_after_deactivate,
                args.home_sleep_after_reset,
            )
        if not home_ok:
            if args.home_failure_shell.strip():
                print(
                    f"[auto_record] Running --home-failure-shell after homing failure.",
                    flush=True,
                )
                try:
                    subprocess.run(
                        ["bash", "-lc", args.home_failure_shell.strip()],
                        env=env,
                        check=False,
                    )
                except OSError as e:
                    print(f"[auto_record] home-failure-shell OS error: {e}", flush=True)
            if args.on_home_failure == "exit":
                print(
                    "[auto_record] Exiting with code 3 (--on-home-failure exit). "
                    "Restart Gazebo or the eval container, then rerun.",
                    file=sys.stderr,
                    flush=True,
                )
                return 3

        spawn_script = _build_spawn_shell(
            setup, x, y, z, yaw, mount_args, cable, args.spawn_cable
        )
        try:
            _run_shell(spawn_script, env)
        except subprocess.CalledProcessError as e:
            print(f"[auto_record] spawn_task_board failed: {e}", flush=True)
            time.sleep(5.0)
            continue

        try:
            subprocess.run(_tare_cmd(pixi), env=env, check=True)
        except subprocess.CalledProcessError as e:
            print(
                f"[auto_record] tare failed (continuing anyway): {e}. "
                "Force-abort still applies from CheatCode logs; fix tare if readings look uncalibrated.",
                flush=True,
            )

        record_cmd = _lerobot_record_cmd(
            pixi,
            os.path.expanduser(args.dataset_root),
            args.repo_id,
            mod,
            port,
            plug,
            task,
            resume=not args.no_resume,
        )
        print(f"[auto_record] Popen: {' '.join(record_cmd)}", flush=True)

        # Child stdout is a pipe (not a TTY) → Python block-buffering unless unbuffered; without
        # this, [CheatCode] lines may not reach this parent until the buffer fills and force-abort
        # would fire far too late or never during a long episode.
        record_env = {**env, "PYTHONUNBUFFERED": "1"}

        proc = subprocess.Popen(
            record_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=record_env,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        success = False
        force_aborted = False
        t0 = time.monotonic()
        is_nic_episode = mod.startswith("nic_card_mount_")
        require_insert = not args.nic_success_any_phase
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                # Force abort before success: high force must discard even if the same line
                # could otherwise match a success heuristic.
                if not args.no_abort_on_high_cheatcode_force and _cheatcode_force_abort(
                    line,
                    args.abort_on_force_greater_than,
                    cheatcode_force_abort_phases,
                ):
                    print(
                        f"[auto_record] CheatCode force >= {args.abort_on_force_greater_than}N; "
                        "discarding episode (Left + terminate child). Next iteration resets spawn/home.",
                        flush=True,
                    )
                    force_aborted = True
                    break
                if _is_success_line(line):
                    success = True
                    break
                if (
                    is_nic_episode
                    and not args.nic_success_disable_plug_z
                    and _nic_plug_z_depth_success(
                        line,
                        args.nic_success_plug_z_at_most,
                        require_insert_phase=require_insert,
                    )
                ):
                    print(
                        "[auto_record] NIC plug_z_actual threshold reached; "
                        "treating as success (Right).",
                        flush=True,
                    )
                    success = True
                    break
                if time.monotonic() - t0 > args.max_episode_seconds:
                    print("[auto_record] Episode timeout", flush=True)
                    break
        except KeyboardInterrupt:
            if proc.poll() is None:
                proc.terminate()
            raise

        if success:
            print(
                f"[auto_record] Waiting {args.success_right_arrow_delay:.1f}s before Right (save)...",
                flush=True,
            )
            _wait_before_success_right(proc, args.success_right_arrow_delay)
            _press_key("right")
            # Read rest of lerobot-record stdout (video encode, "Stop recording", etc.) so the
            # child cannot deadlock on a full pipe, then wait for process exit — that is our
            # best signal the episode pipeline finished (LeRobot commits on exit after Right).
            _drain_subprocess_stdout(proc)
            try:
                proc.wait(timeout=120.0)
            except subprocess.TimeoutExpired:
                print("[auto_record] lerobot-record still running after success; terminating.", flush=True)
                proc.terminate()
                try:
                    proc.wait(timeout=20.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
            if proc.returncode:
                print(
                    f"[auto_record] WARNING: lerobot-record exited with code {proc.returncode}; "
                    "verify the episode in the dataset.",
                    flush=True,
                )
        else:
            _press_key("left")
            if not force_aborted:
                print(
                    "[auto_record] Discarding episode (Left): lerobot-record should drop this run; "
                    "if an episode folder still appears, delete it with lerobot-edit-dataset.",
                    flush=True,
                )
            # Force-abort: pynput from this parent often does not reach lerobot-record's in-process
            # listener (focus / synthetic events). Stop recording by terminating the child promptly.
            discard_sleep = 0.06 if force_aborted else 0.4
            time.sleep(discard_sleep)
            _drain_subprocess_stdout(proc)
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=20.0 if not force_aborted else 8.0)
                except subprocess.TimeoutExpired:
                    proc.kill()

        time.sleep(max(0.0, args.post_episode_sleep))

        if args.max_episodes is not None and episode >= args.max_episodes:
            if args.episode_mode != "training333":
                print(
                    f"\n[auto_record] Completed {args.max_episodes} episodes (--max-episodes). Exiting.",
                    flush=True,
                )
                break
            # training333 + --max-episodes: episode N is last curriculum; after that, weighted random.


if __name__ == "__main__":
    raise SystemExit(main())
