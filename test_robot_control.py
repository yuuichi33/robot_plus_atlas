from __future__ import annotations

import argparse
import sys

from robot_control import RobotController, RobotControlConfig


def safe_print(text: str) -> None:
    data = str(text).encode(sys.stdout.encoding or "utf-8", errors="backslashreplace")
    sys.stdout.buffer.write(data + b"\n")
    sys.stdout.buffer.flush()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--move-test", action="store_true")
    parser.add_argument("--action", type=int, choices=[0, 1, 2, 3])
    parser.add_argument("--task", nargs=2, metavar=("POSITION", "ATTACK"))
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    cfg = RobotControlConfig(
        port=args.port,
        baudrate=args.baudrate,
        dry_run=args.dry_run,
    )

    with RobotController(cfg) as robot:
        safe_print("== PING ==")
        reply = robot.ping()
        safe_print(reply.text())

        safe_print("")
        safe_print("== STOP ==")
        reply = robot.stop()
        safe_print(reply.text())

        if args.move_test:
            input("Safe? Press ENTER to run MOVE_TEST...")
            reply = robot.move_test()
            safe_print(reply.text())

            safe_print("Run STOP after MOVE_TEST")
            reply = robot.stop()
            safe_print(reply.text())

        if args.action is not None:
            input("Safe? Press ENTER to run ACTION_{}...".format(args.action))
            reply = robot.action_by_index(args.action)
            safe_print(reply.text())

            safe_print("Run STOP after ACTION")
            reply = robot.stop()
            safe_print(reply.text())

        if args.task:
            position = int(args.task[0])
            attack = args.task[1]

            input("Safe? Press ENTER to run task: pos={} attack={}...".format(position, attack))
            reply = robot.run_task(position, attack)
            safe_print(reply.text())

            safe_print("Run STOP after task")
            reply = robot.stop()
            safe_print(reply.text())


if __name__ == "__main__":
    main()