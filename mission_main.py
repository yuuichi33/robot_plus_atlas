from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Union

from robot_control import RobotController, RobotControlConfig

# 真实视觉模块（可选，仅在 Atlas 上运行时导入）
try:
    from vision import VisionPerception, parse_task_text as _parse_task_text

    _HAS_VISION = True
except ImportError:
    _HAS_VISION = False
    VisionPerception = None  # type: ignore
    _parse_task_text = None


class MissionState(Enum):
    INIT = "INIT"
    SEARCH_BLOCK = "SEARCH_BLOCK"
    APPROACH_BLOCK = "APPROACH_BLOCK"
    CIRCLE_SCAN_TEXT = "CIRCLE_SCAN_TEXT"
    NAVIGATE_TO_GONG = "NAVIGATE_TO_GONG"
    SCAN_QR = "SCAN_QR"
    EXECUTE_TASK = "EXECUTE_TASK"
    FINISHED = "FINISHED"
    ERROR = "ERROR"


@dataclass
class VisionResult:
    found: bool
    center_x: int = 0
    center_y: int = 0
    area: int = 0
    distance_level: str = "unknown"
    label: str = ""


@dataclass
class TaskResult:
    valid: bool
    position: int = 0
    attack: str = ""


class PerceptionStub:
    """桩模块：用于无摄像头 / dry-run 调试。"""

    def __init__(self, mock_position: int = 1, mock_attack: str = "chop"):
        self.mock_position = mock_position
        self.mock_attack = mock_attack
        self.search_count = 0
        self.scan_count = 0

    def detect_block(self) -> VisionResult:
        self.search_count += 1

        if self.search_count < 3:
            return VisionResult(found=False)

        return VisionResult(
            found=True,
            center_x=320,
            center_y=240,
            area=12000,
            distance_level="middle",
        )

    def read_task_text(self) -> VisionResult:
        self.scan_count += 1

        if self.scan_count < 2:
            return VisionResult(found=False)

        return VisionResult(
            found=True,
            center_x=320,
            center_y=240,
            area=8000,
            label=f"位置{self.mock_position} {self.mock_attack}",
        )

    def detect_qr(self) -> VisionResult:
        """桩模块：模拟二维码扫描（默认始终返回无 QR）。"""
        return VisionResult(found=False)


class MissionController:
    """任务控制器：协调感知 → 决策 → 动作。"""

    def __init__(
        self,
        robot: RobotController,
        perception: Any,  # VisionPerception | PerceptionStub（鸭子类型）
        enable_chassis: bool = False,
        enable_qr_check: bool = True,
        max_search_steps: int = 60,
        max_scan_steps: int = 40,
    ):
        self.robot = robot
        self.perception = perception
        self.enable_chassis = enable_chassis
        self.enable_qr_check = enable_qr_check
        self.max_search_steps = max_search_steps
        self.max_scan_steps = max_scan_steps
        self.state = MissionState.INIT

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def log(self, msg: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}][{self.state.value}] {msg}", flush=True)

    def drive_stop(self) -> None:
        self.robot.stop()

    def drive_forward(self, speed: int, duration_ms: int) -> None:
        if not self.enable_chassis:
            self.log(f"[SKIP] chassis forward: speed={speed}, duration_ms={duration_ms}")
            time.sleep(duration_ms / 1000.0)
            return
        self.robot.forward(speed=speed, duration_ms=duration_ms)

    def drive_rotate_left(self, turn: int, duration_ms: int) -> None:
        if not self.enable_chassis:
            self.log(f"[SKIP] chassis rotate_left: turn={turn}, duration_ms={duration_ms}")
            time.sleep(duration_ms / 1000.0)
            return
        self.robot.rotate_left(turn=turn, duration_ms=duration_ms)

    def drive_rotate_right(self, turn: int, duration_ms: int) -> None:
        if not self.enable_chassis:
            self.log(f"[SKIP] chassis rotate_right: turn={turn}, duration_ms={duration_ms}")
            time.sleep(duration_ms / 1000.0)
            return
        self.robot.rotate_right(turn=turn, duration_ms=duration_ms)

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------

    def run(self) -> None:
        try:
            self.state = MissionState.INIT
            self.log("[START] mission start")
            self.log(f"  enable_chassis  = {self.enable_chassis}")
            self.log(f"  enable_qr_check = {self.enable_qr_check}")

            # 初始化通信
            self.robot.ping()
            self.robot.stop()
            self.log("[OK] Atlas communication OK")

            # 阶段 1：转圈搜索方块
            self.state = MissionState.SEARCH_BLOCK
            found = self.search_block()
            if not found:
                raise RuntimeError("未找到方块，任务终止")

            # 阶段 2：靠近方块
            self.state = MissionState.APPROACH_BLOCK
            self.approach_block()

            # 阶段 3：绕柱子扫描 A4 纸文字 -> 识别到就做动作
            self.state = MissionState.CIRCLE_SCAN_TEXT
            task = self.circle_and_scan_text()
            if not task.valid:
                raise RuntimeError("未识别到 A4 纸任务文字，任务终止")
            self.log(f"[TASK] recognized: position={task.position}, attack={task.attack}")

            self.state = MissionState.EXECUTE_TASK
            self.execute_task(task)
            self.log("[ACTION] first action done (text-based)")

            # 阶段 4：导航到锣 + 扫描二维码 -> 匹配后做第二次动作
            if self.enable_qr_check:
                self.state = MissionState.NAVIGATE_TO_GONG
                self.navigate_to_gong(task.position)

                self.state = MissionState.SCAN_QR
                qr_ok = self.scan_qr_and_verify(task.position)
                if qr_ok:
                    self.state = MissionState.EXECUTE_TASK
                    self.execute_task(task)
                    self.log("[ACTION] second action done (QR-based)")
                else:
                    self.log("[WARN] QR verification failed, skip second action")

            self.state = MissionState.FINISHED
            self.robot.stop()
            self.log("[FINISH] mission complete")

        except Exception as exc:
            self.state = MissionState.ERROR
            self.log(f"[ERROR] mission error: {exc}")
            try:
                self.robot.stop()
            except Exception:
                pass
            raise

    # ------------------------------------------------------------------
    # 阶段 1：转圈搜索方块
    # ------------------------------------------------------------------

    def search_block(self) -> bool:
        self.log("[SEARCH] searching for colored block...")
        step = 0
        while True:
            result = self.perception.detect_block()

            self.log(
                f"  step {step:04d}: found={result.found}, "
                f"area={result.area}, dist={result.distance_level}"
                + (f", color={result.label}" if result.label else "")
            )

            if result.found:
                self.log(f"[FOUND] block detected: color={result.label}, area={result.area}")
                return True

            self.drive_rotate_left(turn=90, duration_ms=400)
            time.sleep(0.15)
            step += 1

    # ------------------------------------------------------------------
    # 阶段 2：靠近方块
    # ------------------------------------------------------------------

    def approach_block(self) -> None:
        self.log("[MOVE] approaching block...")
        # 分步靠近，每一步后检查距离
        for i in range(3):
            result = self.perception.detect_block()
            if result.found and result.distance_level == "near":
                self.log(f"  已足够接近（area={result.area}），停止前进")
                break
            self.log(f"  第{i+1}步前进...")
            self.drive_forward(speed=15, duration_ms=400)
            time.sleep(0.2)

        self.drive_stop()
        time.sleep(0.3)

    # ------------------------------------------------------------------
    # 阶段 3：绕柱子扫描 A4 纸
    # ------------------------------------------------------------------

    def circle_and_scan_text(self) -> TaskResult:
        self.log("[SCAN] circling to scan A4 paper...")
        step = 0
        while True:
            result = self.perception.read_task_text()

            if result.found and result.label:
                # 尝试解析文字 → TaskResult
                if _parse_task_text is not None:
                    valid, pos, att = _parse_task_text(result.label)
                else:
                    # 桩模块已预先解析好（label 为 "位置X chop/stab"）
                    valid = True
                    # 简单解析
                    import re
                    pm = re.search(r"位置\s*([12])", result.label)
                    pos = int(pm.group(1)) if pm else 1
                    att = "chop" if "劈" in result.label or "砍" in result.label or "chop" in result.label.lower() else "stab"

                self.log(
                    f"  scan step {step:02d}: found={valid}, "
                    f"text=\"{result.label}\", pos={pos}, att={att}"
                )

                if valid:
                    self.log(f"[TASK] text recognized: position={pos}, attack={att}")
                    return TaskResult(valid=True, position=pos, attack=att)
            else:
                self.log(f"  scan step {step:02d}: 未检测到文字")

            # 绕柱子旋转扫描
            self.drive_rotate_left(turn=45, duration_ms=300)
            time.sleep(0.15)
            step += 1

    # ------------------------------------------------------------------
    # 阶段 4a：导航到锣的位置
    # ------------------------------------------------------------------

    def navigate_to_gong(self, target_position: int) -> None:
        """
        根据任务文字指示的位置（1 或 2），导航到对应的锣。
        此处实现为简单的定向移动，实际可能需要更复杂的路径规划。
        """
        self.log(f"[NAV] navigating to gong at position {target_position}...")

        # 位置 1 和位置 2 的锣在不同方向
        # 这里假设：位置1在左前方，位置2在右前方
        # 实际使用时需要根据场地调整
        if target_position == 1:
            self.log("  向左转，前往位置 1...")
            self.drive_rotate_left(turn=90, duration_ms=600)
        else:
            self.log("  向右转，前往位置 2...")
            self.drive_rotate_right(turn=90, duration_ms=600)

        time.sleep(0.3)

        # 前进到锣前
        for i in range(5):
            self.log(f"  第{i+1}步前进...")
            self.drive_forward(speed=15, duration_ms=400)
            time.sleep(0.2)

        self.drive_stop()
        time.sleep(0.3)

    # ------------------------------------------------------------------
    # 阶段 4b：扫描二维码验证位置
    # ------------------------------------------------------------------

    def scan_qr_and_verify(self, expected_position: int) -> bool:
        """
        扫描锣上的二维码，验证是否到达正确位置。
        二维码格式：POS=1 或 POS=2
        """
        self.log(f"[QR] scanning QR code, expected POS={expected_position}...")
        attempt = 0
        while True:
            result = self.perception.detect_qr()

            if result.found and result.label:
                self.log(f"  QR 内容: \"{result.label}\"")

                # 解析 POS=1 / POS=2
                pos_value = self._parse_qr_position(result.label)
                if pos_value == expected_position:
                    self.log(f"[OK] QR verified: POS={pos_value}")
                    return True
                elif pos_value in (1, 2):
                    self.log(f"[WARN] QR POS={pos_value}, expected POS={expected_position}, mismatch!")
                    # 仍然返回 True 继续执行（可根据需求改为 False）
                    return True
                else:
                    self.log(f"  无法解析 POS 值，重试...")
            else:
                self.log(f"  attempt {attempt:02d}: 未检测到二维码")

            time.sleep(0.3)
            attempt += 1

    @staticmethod
    def _parse_qr_position(qr_text: str) -> int:
        """从 QR 码文本中提取 POS 编号。"""
        import re
        m = re.search(r"POS\s*[=：:]\s*(\d+)", qr_text, re.IGNORECASE)
        if m:
            return int(m.group(1))
        # 也尝试匹配纯数字
        m = re.search(r"\b([12])\b", qr_text)
        if m:
            return int(m.group(1))
        return 0

    # ------------------------------------------------------------------
    # 阶段 5：执行动作（劈砍/刺击，动作自带语音）
    # ------------------------------------------------------------------

    def execute_task(self, task: TaskResult) -> None:
        self.log(f"[ACTION] execute: position={task.position}, attack={task.attack}")
        self.robot.run_task(task.position, task.attack)
        time.sleep(1.0)
        self.robot.stop()


# =========================================================================
# 入口
# =========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="机器人任务控制程序",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 无硬件 dry-run 测试
  python mission_main.py --dry-run

  # 使用真实摄像头 + mock 底盘
  python mission_main.py --vision --camera 0 --mock-position 1 --mock-attack chop

  # 完整真实运行（摄像头 + 底盘）
  python mission_main.py --vision --camera 0 --enable-chassis --port /dev/ttyUSB0
        """,
    )

    # ---- 通信参数 ----
    parser.add_argument("--port", default="/dev/ttyUSB0", help="Atlas 串口设备")
    parser.add_argument("--baudrate", type=int, default=115200, help="串口波特率")

    # ---- 视觉参数 ----
    parser.add_argument("--vision", action="store_true", help="启用真实摄像头视觉（默认使用桩模块）")
    parser.add_argument("--camera", type=int, default=0, help="摄像头设备 ID（默认 0 → /dev/video0）")
    parser.add_argument(
        "--block-colors", nargs="+", default=None,
        help="要检测的方块颜色（默认：red blue green yellow purple orange）",
    )

    # ---- 任务参数 ----
    parser.add_argument("--mock-position", type=int, choices=[1, 2], default=1,
                        help="桩模块模拟的目标位置")
    parser.add_argument("--mock-attack", choices=["chop", "stab"], default="chop",
                        help="桩模块模拟的动作类型")
    parser.add_argument("--enable-chassis", action="store_true",
                        help="启用真实底盘控制（否则跳过所有移动指令）")
    parser.add_argument("--enable-qr", action="store_true", default=True,
                        help="启用二维码验证（默认开启）")
    parser.add_argument("--no-qr", action="store_true",
                        help="禁用二维码验证")
    parser.add_argument("--dry-run", action="store_true",
                        help="完全模拟模式（不连接任何硬件）")

    # ---- 调试参数 ----
    parser.add_argument("--save-debug-frames", action="store_true",
                        help="保存视觉调试帧到磁盘")
    parser.add_argument("--debug-dir", default="/tmp/robot_debug",
                        help="调试帧保存目录")

    args = parser.parse_args()

    # ------- 创建机器人控制器 -------
    cfg = RobotControlConfig(
        port=args.port,
        baudrate=args.baudrate,
        dry_run=args.dry_run,
    )

    # ------- 创建感知模块 -------
    perception = None
    vision_obj = None  # 保持引用以便关闭

    if args.vision or not args.dry_run:
        # 尝试使用真实视觉
        if not _HAS_VISION:
            print("[WARN] vision.py not found, fallback to stub")
            perception = PerceptionStub(
                mock_position=args.mock_position,
                mock_attack=args.mock_attack,
            )
        else:
            vision_obj = VisionPerception(
                camera_id=args.camera,
                block_colors=args.block_colors,
                debug_dir=args.debug_dir if args.save_debug_frames else "",
            )
            vision_obj.open()
            perception = vision_obj
            print(f"[VISION] real camera enabled: /dev/video{args.camera}")
    else:
        perception = PerceptionStub(
            mock_position=args.mock_position,
            mock_attack=args.mock_attack,
        )
        print("[STUB] using mock perception")

    # ------- 运行任务 -------
    try:
        with RobotController(cfg) as robot:
            mission = MissionController(
                robot=robot,
                perception=perception,
                enable_chassis=args.enable_chassis,
                enable_qr_check=args.enable_qr and not args.no_qr,
            )
            mission.run()
    finally:
        if vision_obj is not None:
            vision_obj.close()


if __name__ == "__main__":
    main()
