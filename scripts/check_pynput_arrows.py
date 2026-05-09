#!/usr/bin/env python3
"""
Quick pynput sanity check for LeRobot key controls.

Usage:
  pixi run python scripts/check_pynput_arrows.py

Press:
  - Left arrow  -> prints detection
  - Right arrow -> prints detection
  - Esc         -> exits
  - q           -> exits
"""

from __future__ import annotations

import sys
from datetime import datetime


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def main() -> int:
    print(f"[{ts()}] Starting pynput arrow-key test.")
    print(f"[{ts()}] DISPLAY={repr(__import__('os').environ.get('DISPLAY'))}")
    print(f"[{ts()}] Press Left/Right arrows. Press Esc or q to quit.")

    try:
        from pynput import keyboard
    except Exception as exc:
        print(f"[{ts()}] FAILED: could not import/initialize pynput.")
        print(f"[{ts()}] Error: {exc}")
        return 1

    def on_press(key):
        try:
            if key == keyboard.Key.left:
                print(f"[{ts()}] LEFT arrow detected")
            elif key == keyboard.Key.right:
                print(f"[{ts()}] RIGHT arrow detected")
            elif key == keyboard.Key.esc:
                print(f"[{ts()}] ESC detected, exiting.")
                return False
            elif hasattr(key, "char") and key.char == "q":
                print(f"[{ts()}] q detected, exiting.")
                return False
            else:
                print(f"[{ts()}] Other key: {key}")
        except Exception as exc:
            print(f"[{ts()}] Error handling key event: {exc}")
            return False

    print(f"[{ts()}] Listener started.")
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

    print(f"[{ts()}] Listener stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
