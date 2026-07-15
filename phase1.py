"""
阶段一：纯文字识别 + 动作执行
摄像头持续扫描白色文字，识别到"位置X 劈砍/刺击"即执行对应动作。
底盘不动，仅视觉+动作。
"""

from __future__ import annotations

import argparse
import time
from typing import Optional

from robot_control import RobotController, RobotControlConfig
from vision import VisionPerception, parse_task_text


def main():
    parser = argparse.ArgumentParser(description="阶段一：文字识别 + 动作")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--dry-run", action="store_true", help="不连串口")
    parser.add_argument("--display", action="store_true", help="显示摄像头画面")
    args = parser.parse_args()

    cfg = RobotControlConfig(port=args.port, dry_run=args.dry_run)
    vp = VisionPerception(camera_id=args.camera)

    with RobotController(cfg) as robot, vp:
        robot.ping()
        robot.stop()
        print("[phase1] 开始持续文字识别，Ctrl+C 退出...")

        while True:
            result = vp.read_task_text()

            if result.found and result.label:
                valid, pos, att = parse_task_text(result.label)
                print(f"[phase1] OCR: \"{result.label}\" → pos={pos}, attack={att}, valid={valid}")

                if valid:
                    print(f"[phase1] 执行动作: 位置{pos} {att}")
                    robot.run_task(pos, att)
                    time.sleep(1)
                    robot.stop()
                    print("[phase1] 动作完成，继续扫描...")

            if args.display and hasattr(vp, 'show_debug'):
                vp.show_debug("PHASE1_SCAN")

            time.sleep(0.3)


if __name__ == "__main__":
    main()
