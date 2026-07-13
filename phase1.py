"""
阶段一：文字识别 + QR 识别 + 动作执行
交替扫描文字和二维码，识别到即执行动作。
"""

from __future__ import annotations

import argparse
import re
import time
from robot_control import RobotController, RobotControlConfig
from vision import VisionPerception, parse_task_text


def parse_qr(text):
    """从 QR 内容提取 POS 编号。"""
    m = re.search(r"POS\s*[=：:]\s*(\d+)", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"\b([12])\b", text)
    return int(m.group(1)) if m else 0


def main():
    parser = argparse.ArgumentParser(description="阶段一：文字 + QR 识别")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--display", action="store_true")
    args = parser.parse_args()

    cfg = RobotControlConfig(port=args.port, dry_run=args.dry_run)
    vp = VisionPerception(camera_id=args.camera)

    with RobotController(cfg) as robot, vp:
        robot.ping()
        robot.stop()
        print("[phase1] 开始扫描文字和 QR...")

        last_pos = 1
        last_att = "chop"

        while True:
            # 扫文字
            result = vp.read_task_text()
            if result.found and result.label:
                valid, pos, att = parse_task_text(result.label)
                print(f"[phase1] OCR: \"{result.label}\" → pos={pos}, att={att}, valid={valid}")
                if valid:
                    last_pos = pos
                    last_att = att
                    print(f"[phase1] 执行: 位置{pos} {att}")
                    robot.run_task(pos, att)
                    time.sleep(1)
                    robot.stop()

            # 扫二维码
            result = vp.detect_qr()
            if result.found and result.label:
                qr_pos = parse_qr(result.label)
                print(f"[phase1] QR: \"{result.label}\" → pos={qr_pos}")
                if qr_pos in (1, 2):
                    if qr_pos == last_pos:
                        print(f"[phase1] QR 匹配文字位置{last_pos}，执行: {last_att}")
                        robot.run_task(last_pos, last_att)
                        time.sleep(1)
                        robot.stop()
                    else:
                        print(f"[phase1] QR 位置{qr_pos} 与文字位置{last_pos} 不一致，后退")
                        robot.backward(speed=15, duration_ms=800)
                        time.sleep(1)
                        robot.stop()

            time.sleep(0.3)


if __name__ == "__main__":
    main()
