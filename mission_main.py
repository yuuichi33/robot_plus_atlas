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

    def drive_arc_left(self, speed: int, turn: int, duration_ms: int) -> None:
        """同时前进+左转，走出弧线绕行。"""
        if not self.enable_chassis:
            self.log(f"[SKIP] chassis arc_left: speed={speed}, turn={turn}, ms={duration_ms}")
            time.sleep(duration_ms / 1000.0)
            return
        self.robot.move(0, speed, abs(turn), duration_ms)

    def drive_strafe_left(self, speed: int, duration_ms: int) -> None:
        """横向左移（切向运动），保持径向对准四方体。"""
        if not self.enable_chassis:
            self.log(f"[SKIP] chassis strafe_left: speed={speed}, ms={duration_ms}")
            time.sleep(duration_ms / 1000.0)
            return
        self.robot.move(90, speed, 0, duration_ms)

    def drive_backward(self, speed: int, duration_ms: int) -> None:
        """后退。"""
        if not self.enable_chassis:
            self.log(f"[SKIP] chassis backward: speed={speed}, duration_ms={duration_ms}")
            time.sleep(duration_ms / 1000.0)
            return
        self.robot.backward(speed=speed, duration_ms=duration_ms)

    def _find_and_center_block(self, max_search: int = 8) -> VisionResult:
        """视觉找回方块并回正到视野中央。丢失时左右搜索找回。"""
        block = self.perception.detect_block()

        if not block.found:
            self.log("  cuboid not in view, searching...")
            found_in_search = False
            for attempt in range(max_search):
                # 左转搜索
                self.drive_rotate_left(turn=500, duration_ms=700)
                time.sleep(0.3)
                block = self.perception.detect_block()
                self._show_frame()
                if block.found:
                    found_in_search = True
                    break
                # 右转搜索
                self.drive_rotate_right(turn=500, duration_ms=700)
                time.sleep(0.3)
                block = self.perception.detect_block()
                self._show_frame()
                if block.found:
                    found_in_search = True
                    break
                self.log(f"  search attempt {attempt + 1}/{max_search}")

            if not found_in_search:
                self.log("  cuboid not found after search")
                return VisionResult(found=False)

        # 回正：微调让方块回到视野中央
        self.log(f"  cuboid found (cx={block.center_x}), centering...")
        for adjust in range(10):
            error_x = block.center_x - 320
            if abs(error_x) < 40:
                self.log(f"  centered (error_x={error_x})")
                break
            if error_x > 0:
                self.drive_rotate_right(turn=200, duration_ms=200)
            else:
                self.drive_rotate_left(turn=200, duration_ms=200)
            time.sleep(0.2)
            block = self.perception.detect_block()
            self._show_frame()
            if not block.found:
                self.log("  cuboid lost during centering")
                return VisionResult(found=False)

        return block

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

            # 找到后停 5 秒
            self.log("[FOUND] cuboid found, pausing 5s...")
            self.drive_stop()
            time.sleep(5)

            # 阶段 2：缓慢绕四方体移动 + 扫描文字
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
    # 阶段 1：旋转搜索方块
    # ------------------------------------------------------------------

    def search_block(self) -> bool:
        self.log("[SEARCH] rotating to find cuboid...")
        step = 0
        while True:
            # 旋转搜索，途中也检测
            self.log(f"  step {step:04d}: rotating...")
            rotate_end = time.time() + 1.2  # 700ms 旋转 + 余量
            self.drive_rotate_left(turn=500, duration_ms=700)
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
                        f"cx={result.center_x} score={getattr(result, 'score', 0):.2f}"
                    )
                    return True
                time.sleep(0.2)

            self.log(f"  step {step:04d}: not found in this direction")
            step += 1

    # ------------------------------------------------------------------
    # 阶段 2：分段绕行立方体，逐面扫描文字
    # ------------------------------------------------------------------

    def orbit_and_scan(self) -> TaskResult:
        """
        分段绕行立方体：面向方块 → 扫文字 → 没找到 → 右转90° → 前进 → 左转90° → 再面向方块。
        立方体4个面，最多绕行4次覆盖所有面。
        """
        self.log("[ORBIT] walk-around cuboid, scanning for text on each face...")

        MAX_FACES = 4               # 立方体最多4个面
        OCR_CONFIRM_RETRY = 10      # OCR确认最多重试次数
        NEAR_AREA_THRESHOLD = 50000 # 太近时后退的面积阈值

        # ---- 0. 如果太近方块，先后退 ----
        block = self.perception.detect_block()
        if block.found and block.area > NEAR_AREA_THRESHOLD:
            self.log(f"  too close (area={block.area:.0f}), backing up 1.5s...")
            self.drive_backward(speed=10, duration_ms=1500)
            time.sleep(0.5)

        for face_idx in range(MAX_FACES):
            self.log(f"  === face {face_idx}/{MAX_FACES - 1} ===")

            # ---- 1. 视觉找回并回正方块 ----
            block = self._find_and_center_block(max_search=8)
            if not block.found:
                self.log("[ORBIT] block lost and not recovered, aborting")
                return TaskResult(valid=False)

            self.log(f"  face {face_idx}: cuboid centered (area={block.area:.0f}), scanning text...")

            # ---- 2. 扫文字：没白纸等3秒，有白纸最多等15秒 ----
            text_found = False
            result = None
            start = time.time()
            while True:
                result = self.perception.read_task_text()
                if result.found and result.label:
                    text_found = True
                    break
                has_white = getattr(self.perception, '_white_detected', False)
                elapsed = time.time() - start
                if (not has_white and elapsed > 3) or (has_white and elapsed > 15):
                    if has_white and elapsed > 15:
                        self.log(f"  face {face_idx}: OCR failed 15s despite white paper")
                    else:
                        self.log(f"  face {face_idx}: no white paper in 3s")
                    break
                time.sleep(0.3)

            if text_found:
                # ---- 3. OCR确认 ----
                self.drive_stop()
                self.log(f"[TEXT] spotted: \"{result.label[:30]}\", confirming...")
                time.sleep(0.5)
                for retry in range(1, OCR_CONFIRM_RETRY + 1):
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
                        self.log(f"  confirm {retry}: valid={valid}, text=\"{result2.label[:30]}\"")
                        if valid:
                            self.log(f"[TASK] confirmed: position={pos}, attack={att}")
                            return TaskResult(valid=True, position=pos, attack=att)
                    else:
                        if retry % 5 == 0:
                            self.log(f"  confirm {retry}: waiting for OCR...")
                    time.sleep(0.5)
                self.log(f"  OCR confirm failed, moving to next face")

            # ---- 4. 没找到文字，绕到下一面 ----
            if face_idx < MAX_FACES - 1:
                self.log(f"  face {face_idx}: no text found, walking to next face...")
                # 右转90° → 前进 → 左转90°（绕到立方体下一面）
                self.drive_rotate_right(turn=500, duration_ms=700)
                time.sleep(0.3)
                self.drive_forward(speed=15, duration_ms=1500)
                time.sleep(0.3)
                self.drive_rotate_left(turn=500, duration_ms=700)
                time.sleep(0.5)

        self.log("[ORBIT] checked all 4 faces, no valid text found")
        return TaskResult(valid=False)

    # ------------------------------------------------------------------
    # 阶段 3：转圈搜索 QR 码 → 对准 → 靠近 → 验证
    # ------------------------------------------------------------------

    def search_and_approach_qr(self, expected_position: int) -> bool:
        """
        旋转搜索 → 停 5 秒扫 QR。找到后直接验证。
        """
        self.log(f"[QR] searching for QR code, expected POS={expected_position}...")
        search_step = 0

        while True:
            # 旋转搜索，途中也扫
            self.log(f"  qr step {search_step:03d}: rotating...")
            rotate_end = time.time() + 1.2
            self.drive_rotate_left(turn=500, duration_ms=700)
            while time.time() < rotate_end:
                result = self.perception.detect_qr()
                self._show_frame()
                if result.found and result.label:
                    self.drive_stop()
                    if self._verify_qr(result, expected_position):
                        return True
                time.sleep(0.1)

            # 停 5 秒扫描
            deadline = time.time() + 5.0
            while time.time() < deadline:
                result = self.perception.detect_qr()
                self._show_frame()
                if result.found and result.label:
                    if self._verify_qr(result, expected_position):
                        return True
                time.sleep(0.2)

            if search_step % 5 == 0:
                self.log(f"  qr step {search_step:03d}: not found")
            search_step += 1

    def _verify_qr(self, result, expected_position: int) -> bool:
        """扫到 QR → 前进 3 秒 → 按扫到的内容匹配。"""
        self.log(f"  [FOUND] QR: \"{result.label}\" at x={result.center_x}")

        # 前进 3 秒
        self.log("  [QR] moving forward 3s...")
        self.drive_forward(speed=10, duration_ms=3000)
        time.sleep(0.5)

        # 用前进前扫到的内容匹配
        pos_value = self._parse_qr_position(result.label)
        if pos_value == expected_position:
            self.log(f"[OK] QR verified: POS={pos_value}, matched!")
            return True
        elif pos_value in (1, 2):
            self.log(f"[WARN] QR POS={pos_value}, expected {expected_position}, proceed anyway")
            return True
        self.log(f"  QR parse failed: \"{result.label}\"")
        return False

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
