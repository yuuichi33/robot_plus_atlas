from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

try:
    import serial
except ImportError as exc:
    raise RuntimeError("pyserial is not installed. Run: pip3 install pyserial") from exc


class RobotAction(Enum):
    POS1_CHOP = "ACTION_0"
    POS2_CHOP = "ACTION_1"
    POS1_STAB = "ACTION_2"
    POS2_STAB = "ACTION_3"


TASK_TO_ACTION: Dict[Tuple[int, str], RobotAction] = {
    (1, "chop"): RobotAction.POS1_CHOP,
    (2, "chop"): RobotAction.POS2_CHOP,
    (1, "stab"): RobotAction.POS1_STAB,
    (2, "stab"): RobotAction.POS2_STAB,
}


@dataclass
class RobotControlConfig:
    port: str = "/dev/ttyUSB0"
    baudrate: int = 115200
    timeout: float = 0.2
    command_timeout: float = 2.0
    boot_wait_s: float = 1.5
    line_ending: str = "\n"
    dry_run: bool = False


@dataclass
class RobotReply:
    command: str
    ok: bool
    lines: List[str]
    error: Optional[str] = None

    def text(self) -> str:
        return "\n".join(self.lines)


class RobotController:
    def __init__(self, config: Optional[RobotControlConfig] = None):
        self.config = config or RobotControlConfig()
        self.ser: Optional[serial.Serial] = None

    def open(self) -> None:
        if self.config.dry_run:
            print("[DRY-RUN] open serial skipped")
            return

        self.ser = serial.Serial(
            port=self.config.port,
            baudrate=self.config.baudrate,
            timeout=self.config.timeout,
            write_timeout=1.0,
        )

        time.sleep(self.config.boot_wait_s)
        self._clear_input()

    def close(self) -> None:
        if self.ser and self.ser.is_open:
            self.ser.close()

    def __enter__(self) -> "RobotController":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self.stop()
        except Exception:
            pass
        self.close()

    def _check_opened(self) -> None:
        if self.config.dry_run:
            return
        if self.ser is None or not self.ser.is_open:
            raise RuntimeError("Serial port is not open.")

    def _clear_input(self) -> None:
        if self.ser:
            self.ser.reset_input_buffer()

    def _write_line(self, command: str) -> None:
        self._check_opened()

        data = (command.strip() + self.config.line_ending).encode("ascii")

        if self.config.dry_run:
            print("[DRY-RUN TX] " + command)
            return

        assert self.ser is not None
        self.ser.write(data)
        self.ser.flush()

    def _read_lines_until_result(self, timeout: Optional[float] = None) -> RobotReply:
        self._check_opened()

        timeout = timeout or self.config.command_timeout
        deadline = time.time() + timeout
        lines: List[str] = []

        if self.config.dry_run:
            return RobotReply(command="", ok=True, lines=["OK_DRY_RUN"])

        assert self.ser is not None

        while time.time() < deadline:
            raw = self.ser.readline()

            if not raw:
                continue

            line = raw.decode("utf-8", errors="backslashreplace").strip()
            if not line:
                continue

            lines.append(line)
            upper = line.upper()

            if "ERR" in upper or "ERROR" in upper or "FAIL" in upper:
                return RobotReply(command="", ok=False, lines=lines, error=line)

            if upper == "OK" or "OK_" in upper:
                return RobotReply(command="", ok=True, lines=lines)

        return RobotReply(
            command="",
            ok=False,
            lines=lines,
            error="timeout waiting reply",
        )

    def send_command(self, command: str, timeout: Optional[float] = None) -> RobotReply:
        command = command.strip()

        if not command:
            raise ValueError("command is empty")

        try:
            command.encode("ascii")
        except UnicodeEncodeError as exc:
            raise ValueError("command must be ASCII only") from exc

        self._clear_input()
        self._write_line(command)

        reply = self._read_lines_until_result(timeout)
        reply.command = command

        if not reply.ok:
            raise RuntimeError(
                "command failed: "
                + command
                + "\nerror="
                + str(reply.error)
                + "\nreply=\n"
                + reply.text()
            )

        return reply

    def ping(self) -> RobotReply:
        return self.send_command("PING", timeout=1.5)

    def stop(self) -> RobotReply:
        return self.send_command("STOP", timeout=1.0)

    def move_test(self) -> RobotReply:
        return self.send_command("MOVE_TEST", timeout=2.5)

    def move(self, angle: int, speed: int, turn: int, duration_ms: int) -> RobotReply:
        if not 0 <= angle <= 360:
            raise ValueError("angle must be 0..360")
        if not 0 <= speed <= 100:
            raise ValueError("speed must be 0..100")
        if not -1000 <= turn <= 1000:
            raise ValueError("turn must be -1000..1000")
        if not 1 <= duration_ms <= 65000:
            raise ValueError("duration_ms must be 1..65000")

        cmd = "MOVE {} {} {} {}".format(angle, speed, turn, duration_ms)
        wait_s = max(2.0, duration_ms / 1000.0 + 1.0)
        return self.send_command(cmd, timeout=wait_s)

    def forward(self, speed: int = 20, duration_ms: int = 500) -> RobotReply:
        return self.move(0, speed, 0, duration_ms)

    def backward(self, speed: int = 20, duration_ms: int = 500) -> RobotReply:
        return self.move(180, speed, 0, duration_ms)

    def left(self, speed: int = 20, duration_ms: int = 500) -> RobotReply:
        return self.move(90, speed, 0, duration_ms)

    def right(self, speed: int = 20, duration_ms: int = 500) -> RobotReply:
        return self.move(270, speed, 0, duration_ms)

    def rotate_left(self, turn: int = 250, duration_ms: int = 400) -> RobotReply:
        return self.move(0, 0, abs(turn), duration_ms)

    def rotate_right(self, turn: int = 250, duration_ms: int = 400) -> RobotReply:
        return self.move(0, 0, -abs(turn), duration_ms)

    def action(self, action: RobotAction) -> RobotReply:
        return self.send_command(action.value, timeout=4.0)

    def action_by_index(self, index: int) -> RobotReply:
        if index not in (0, 1, 2, 3):
            raise ValueError("index must be 0, 1, 2, or 3")
        return self.send_command("ACTION_{}".format(index), timeout=4.0)

    def run_task(self, position: int, attack_code: str) -> RobotReply:
        action_code = self.normalize_attack_code(attack_code)
        key = (position, action_code)

        if key not in TASK_TO_ACTION:
            raise ValueError(
                "unknown task: position={}, attack_code={}".format(
                    position, attack_code
                )
            )

        return self.action(TASK_TO_ACTION[key])

    @staticmethod
    def normalize_attack_code(text: str) -> str:
        s = str(text).strip().lower()

        chop_cn = "\u5288\u780d"
        stab_cn = "\u523a\u6740"

        if s in ("chop", "slash", "cut", "0", chop_cn):
            return "chop"

        if s in ("stab", "thrust", "pierce", "1", stab_cn):
            return "stab"

        raise ValueError("unknown attack code: " + str(text))

    def voice(self, index: int) -> RobotReply:
        if not 0 <= index <= 255:
            raise ValueError("voice index must be 0..255")
        return self.send_command("VOICE {}".format(index), timeout=3.0)

    def victory(self) -> RobotReply:
        return self.send_command("VICTORY", timeout=4.0)
