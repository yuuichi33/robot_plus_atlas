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
    ORBIT_AND_SCAN = "ORBIT_AND_SCAN"
    SEARCH_QR = "SEARCH_QR"
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
        perception: Any,
        enable_chassis: bool = False,
        enable_qr_check: bool = True,
        max_search_steps: int = 60,
        max_scan_steps: int = 40,
        show_display: bool = False,
    ):
        self.robot = robot
        self.perception = perception
        self.enable_chassis = enable_chassis
        self.enable_qr_check = enable_qr_check
        self.max_search_steps = max_search_steps
        self.max_scan_steps = max_scan_steps
        self.show_display = show_display
        self.state = MissionState.INIT

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def log(self, msg: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}][{self.state.value}] {msg}", flush=True)

    def _show_frame(self) -> None:
        """如果启用了显示，展示当前摄像头画面。"""
        if self.show_display and hasattr(self.perception, 'show_debug'):
            self.perception.show_debug(self.state.value)

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

            # 阶段 1：转圈搜索四方体
            self.state = MissionState.SEARCH_BLOCK
            found = self.search_block()
            if not found:
                raise RuntimeError("未找到四方体，任务终止")

            # 阶段 2：缓慢绕四方体移动 + 扫描文字，发现文字立即停
            self.state = MissionState.ORBIT_AND_SCAN
            task = self.orbit_and_scan()
            if not task.valid:
                raise RuntimeError("未识别到任务文字，任务终止")
            self.log(f"[TASK] recognized: position={task.position}, attack={task.attack}")

            self.state = MissionState.EXECUTE_TASK
            self.execute_task(task)
            self.log("[ACTION] first action done (text-based)")

            # 阶段 3：转圈搜索 QR 码，找到后靠近并验证 -> 做第二次动作
            if self.enable_qr_check:
                self.state = MissionState.SEARCH_QR
                qr_ok = self.search_and_approach_qr(task.position)
                if qr_ok:
                    self.state = MissionState.EXECUTE_TASK
                    self.execute_task(task)
                    self.log("[ACTION] second action done (QR-based)")
                else:
                    self.log("[WARN] QR search failed, skip second action")

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
    # 阶段 1：转圈搜索方块（每60度停5秒检测）
    # ------------------------------------------------------------------

    def search_block(self) -> bool:
        self.log("[SEARCH] step-rotate 60°, stop 5s to detect...")
        step = 0
        while True:
            # 转 60 度，途中也检测
            self.log(f"  step {step:04d}: rotating 60°...")
            rotate_end = time.time() + 0.7  # 400ms 旋转 + 余量
            self.drive_rotate_left(turn=60, duration_ms=400)
            while time.time() < rotate_end:
                result = self.perception.detect_block()
                self._show_frame()
                if result.found:
                    self.drive_stop()
                    self.log(
                        f"  step {step:04d}: FOUND during rotate! "
                        f"area={result.area:.0f} cx={result.center_x}"
                    )
                    return True
                time.sleep(0.1)

            # 停下检测 5 秒
            self.log(f"  step {step:04d}: stopped, detecting for 5s...")
            deadline = time.time() + 5.0
            while time.time() < deadline:
                result = self.perception.detect_block()
                self._show_frame()
                if result.found:
                    self.log(
                        f"  step {step:04d}: FOUND area={result.area:.0f} "
                        f"cx={result.center_x} score={result.score:.2f}"
                    )
                    return True
                time.sleep(0.2)

            self.log(f"  step {step:04d}: not found in this direction")
            step += 1

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 阶段 2：缓慢绕四方体移动 + 扫描文字
    # ------------------------------------------------------------------

    def orbit_and_scan(self) -> TaskResult:
        """
        绕四方体移动，保持其在视野中央。丢失时自动原地搜索找回。
        同时检测文字，一旦发现立即停止，静止 OCR 确认。
        """
        self.log("[ORBIT] orbiting cuboid, scanning for text...")
        step = 0

        while True:
            # ---- 检测四方体 ----
            block = self.perception.detect_block()

            if not block.found:
                # 丢了：转60°/停5秒搜索，途中检测到即停
                self.drive_stop()
                self.log(f"  orbit {step:02d}: block lost, searching...")
                while True:
                    # 旋转中检测
                    self.drive_rotate_left(turn=60, duration_ms=400)
                    rotate_end = time.time() + 0.7
                    while time.time() < rotate_end:
                        block = self.perception.detect_block()
                        if block.found:
                            self.drive_stop()
                            self.log(f"  [FOUND] block recovered, area={block.area:.0f}")
                            break
                        time.sleep(0.1)
                    if block.found:
                        break
                    # 停 5 秒检测
                    deadline = time.time() + 5.0
                    while time.time() < deadline:
                        block = self.perception.detect_block()
                        self._show_frame()
                        if block.found:
                            self.log(f"  [FOUND] block recovered, area={block.area:.0f}")
                            break
                        time.sleep(0.2)
                    if block.found:
                        break
                    self.log(f"  orbit {step:02d}: still searching...")
                    step += 1
                continue

            # ---- 四方体在视野中，保持居中 ----
            error_x = block.center_x - 320
            if step % 5 == 0:
                self.log(
                    f"  orbit {step:02d}: block area={block.area:.0f}, "
                    f"err_x={error_x}, dist={block.distance_level}"
                )

            # 偏差大 → 优先旋转对准
            if abs(error_x) > 50:
                turn_amount = min(60, abs(error_x) // 2)
                if error_x > 0:
                    self.drive_rotate_right(turn=turn_amount, duration_ms=200)
                else:
                    self.drive_rotate_left(turn=turn_amount, duration_ms=200)
                time.sleep(0.15)
            else:
                # 居中 → 小步绕行 + 扫描文字
                self._show_frame()
                self.drive_forward(speed=6, duration_ms=300)
                time.sleep(0.2)
                self.drive_rotate_left(turn=15, duration_ms=300)
                time.sleep(0.2)

                # 扫描文字
                result = self.perception.read_task_text()
                if result.found and result.label:
                    self.drive_stop()
                    self.log(f"[TEXT] spotted: \"{result.label[:30]}\", stopping to confirm...")
                    time.sleep(0.5)
                    retry = 0
                    while True:
                        retry += 1
                        result2 = self.perception.read_task_text()
                        if result2.found and result2.label:
                            if _parse_task_text is not None:
                                valid, pos, att = _parse_task_text(result2.label)
                            else:
                                valid = True
                                import re
                                pm = re.search(r"位置\s*([12])", result2.label)
                                pos = int(pm.group(1)) if pm else 1
                                att = ("chop" if "劈" in result2.label or "砍" in result2.label
                                       or "chop" in result2.label.lower() else "stab")
                            self.log(
                                f"  confirm {retry}: valid={valid}, "
                                f"text=\"{result2.label[:30]}\", pos={pos}, att={att}"
                            )
                            if valid:
                                self.log(f"[TASK] confirmed: position={pos}, attack={att}")
                                return TaskResult(valid=True, position=pos, attack=att)
                        else:
                            if retry % 10 == 0:
                                self.log(f"  confirm {retry}: waiting for OCR...")
                        time.sleep(0.5)

            step += 1

    # ------------------------------------------------------------------
    # 阶段 3：转圈搜索 QR 码 → 对准 → 靠近 → 验证
    # ------------------------------------------------------------------

    def search_and_approach_qr(self, expected_position: int) -> bool:
        """
        原地慢转搜索 QR 码。找到后：
        1. 旋转使 QR 居中
        2. 前进靠近
        3. 靠近后重新扫描验证位置是否匹配
        """
        self.log(f"[QR] searching for QR code, expected POS={expected_position}...")
        search_step = 0

        while True:
            result = self.perception.detect_qr()

            if result.found and result.label:
                self.log(f"  [FOUND] QR: \"{result.label}\" at x={result.center_x}")
                self.drive_stop()
                time.sleep(0.2)

                # 靠近 QR 直到足够近
                self.log("  [QR] approaching QR code...")
                for _ in range(999):  # 靠近步数
                    # 居中对准
                    error_x = result.center_x - 320  # 640/2
                    if abs(error_x) > 50:
                        if error_x > 0:
                            self.drive_rotate_right(turn=min(60, abs(error_x) // 2), duration_ms=200)
                        else:
                            self.drive_rotate_left(turn=min(60, abs(error_x) // 2), duration_ms=200)
                        time.sleep(0.2)

                    # 前进一小步
                    self.drive_forward(speed=12, duration_ms=350)
                    time.sleep(0.2)

                    # 重新检测 QR
                    result = self.perception.detect_qr()
                    if not result.found:
                        self.log("  [QR] lost during approach, re-searching...")
                        break

                self.drive_stop()
                time.sleep(0.3)

                # 靠近后做最终验证
                if result.found and result.label:
                    pos_value = self._parse_qr_position(result.label)
                    if pos_value == expected_position:
                        self.log(f"[OK] QR verified: POS={pos_value}, matched!")
                        return True
                    elif pos_value in (1, 2):
                        self.log(f"[WARN] QR POS={pos_value}, expected {expected_position}, but proceed anyway")
                        return True
                    else:
                        self.log(f"  QR parse failed: \"{result.label}\"")
                continue  # 验证失败，继续转圈找

            if search_step % 10 == 0:
                self.log(f"  qr search {search_step:03d}: not found")

            self.drive_rotate_left(turn=45, duration_ms=250)
            time.sleep(0.1)
            search_step += 1

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
    # 阶段 4：执行动作（劈砍/刺击，动作自带语音）
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
        help="[已废弃] 四方体检测不再依赖颜色，此参数被忽略",
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

    parser.add_argument("--display", action="store_true",
                        help="显示摄像头实时画面（需要图形界面）")

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
                show_display=args.display,
            )
            mission.run()
    finally:
        if vision_obj is not None:
            vision_obj.close()


if __name__ == "__main__":
    main()
