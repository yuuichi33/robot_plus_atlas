"""
YOLO 任务文字识别适配器 —— 用 Atlas NPU YOLOv5 替换 EasyOCR。

用法:
    from vision import VisionPerception
    from atlas_task import AclBackend, AtlasTaskRecognizer
    from yolo_task_reader import YoloTaskReader

    vision = VisionPerception(camera_id=1)
    vision.open()

    backend = AclBackend("models/task_yolov5n_fp16.om")
    recognizer = AtlasTaskRecognizer(backend)
    reader = YoloTaskReader(vision, recognizer)

    # reader 拥有与 VisionPerception 完全相同的接口:
    #   reader.detect_block()   → 代理到 vision (几何柱检测)
    #   reader.read_task_text() → YOLOv5 NPU 分类 (替换 OCR)
    #   reader.detect_qr()      → 代理到 vision (QR 扫描)
    #   reader._white_detected  → 代理到 vision (绕行逻辑需要)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

# VisionResult 与 vision.py 中定义一致，这里单独定义避免循环导入
from dataclasses import dataclass


@dataclass
class VisionResult:
    found: bool
    center_x: int = 0
    center_y: int = 0
    area: float = 0.0
    distance_level: str = "unknown"
    label: str = ""
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    aspect_ratio: float = 0.0
    rectangularity: float = 0.0
    solidity: float = 0.0
    verticality: float = 0.0
    score: float = 0.0


# ---------------------------------------------------------------------------
# Task1~4 → (position, attack) 默认映射
#
# 对着 4 张 A4 纸各跑一次 recognize_once.py 确认映射是否正确：
#   Task1 (返回 1) → 位置1 劈砍
#   Task2 (返回 2) → 位置2 劈砍
#   Task3 (返回 3) → 位置1 刺击
#   Task4 (返回 4) → 位置2 刺击
#
# 如果模型实际映射不同，通过 task_map 参数覆盖。
# ---------------------------------------------------------------------------
DEFAULT_TASK_MAP: Dict[int, Tuple[int, str]] = {
    1: (1, "chop"),   # Task1 → 位置1 劈砍
    2: (2, "chop"),   # Task2 → 位置2 劈砍
    3: (1, "stab"),   # Task3 → 位置1 刺击
    4: (2, "stab"),   # Task4 → 位置2 刺击
}

# 用于生成兼容 parse_task_text() 的 label
_ATTACK_CN = {"chop": "劈砍", "stab": "刺击"}


class YoloTaskReader:
    """用 YOLOv5 NPU 推理替换 vision.py 的 read_task_text()。

    所有其他方法 (detect_block, detect_qr, show_debug, ...) 和属性
    (_white_detected, _last_frame, ...) 透明代理到底层 VisionPerception，
    因此可以直接替代 MissionController 中的 perception 对象。
    """

    def __init__(
        self,
        vision,               # VisionPerception 实例
        recognizer,           # AtlasTaskRecognizer 实例
        task_map: Dict[int, Tuple[int, str]] | None = None,
        stable_threshold: int = 2,
    ):
        """
        Args:
            vision: 已 open() 的 VisionPerception（提供摄像头帧 + 几何检测 + QR）
            recognizer: 已初始化的 AtlasTaskRecognizer（YOLOv5 NPU 推理）
            task_map: Task class_id → (position, attack) 映射，None 用默认
            stable_threshold: 连续多少帧同一结果才确认（防止误检）
        """
        self._vision = vision
        self._recognizer = recognizer
        self._task_map = task_map or DEFAULT_TASK_MAP
        self._stable_threshold = stable_threshold

        # 稳定性计数器
        self._stable_count = 0
        self._last_task_id = 0

        # 与 OCR 路径兼容：白纸检测状态（代理到 vision）
        # YOLO 不需要白纸检测，但绕行逻辑检查此属性决定超时策略
        self._white_detected = False

    # ------------------------------------------------------------------
    # 代理：所有未显式定义的属性/方法回退到 VisionPerception
    # ------------------------------------------------------------------

    def __getattr__(self, name):
        """将 detect_block / detect_qr / show_debug / _last_frame 等
        全部透明代理到底层 VisionPerception。"""
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return getattr(self._vision, name)

    # ------------------------------------------------------------------
    # 核心：用 YOLOv5 NPU 替换 OCR 文字识别
    # ------------------------------------------------------------------

    def read_task_text(self) -> VisionResult:
        """与 VisionPerception.read_task_text() 签名兼容。

        调用链:
          1. 从 VisionPerception 读取当前帧
          2. 送 YOLOv5 NPU 推理
          3. 筛选 Task1~4 检测结果
          4. 稳定确认后返回 VisionResult

        返回的 VisionResult.label 格式为 "位置1 劈砍" 等，
        可被 mission_main 中 parse_task_text() 正确解析。
        """
        # 1) 读帧
        frame = self._vision.read_frame()
        if frame is None:
            return VisionResult(found=False)

        # 2) NPU 推理
        try:
            detections = self._recognizer.detect(frame)
        except Exception as exc:
            print(f"[yolo] inference error: {exc}")
            return VisionResult(found=False)

        if not detections:
            self._stable_count = 0
            self._last_task_id = 0
            return VisionResult(found=False)

        # 3) 筛选 Task1~4
        task_dets = [d for d in detections if 1 <= d.class_id <= 4]
        if not task_dets:
            self._stable_count = 0
            self._last_task_id = 0
            return VisionResult(found=False)

        best = max(task_dets, key=lambda d: d.confidence)
        task_id = best.class_id
        confidence = best.confidence

        # 4) 稳定性确认（连续 stable_threshold 帧返回相同结果才有效）
        if task_id == self._last_task_id:
            self._stable_count += 1
        else:
            self._stable_count = 1
            self._last_task_id = task_id

        if self._stable_count < self._stable_threshold:
            # 检测到但未稳定，标记 _white_detected 让绕行逻辑延长等待
            self._white_detected = True
            return VisionResult(found=False)

        # 5) 构造 VisionResult（兼容 parse_task_text 解析）
        pos, att = self._task_map.get(task_id, (0, ""))
        x1, y1, x2, y2 = best.xyxy
        w, h = x2 - x1, y2 - y1

        label = f"位置{pos} {_ATTACK_CN.get(att, att)}"
        print(
            f"[yolo] task recognized: class=Task{task_id} "
            f"conf={confidence:.2f} → {label}"
        )

        # 重置稳定计数器，为下次识别做准备
        self._stable_count = 0
        self._last_task_id = 0

        return VisionResult(
            found=True,
            center_x=int((x1 + x2) / 2),
            center_y=int((y1 + y2) / 2),
            area=float(w * h),
            label=label,
            bbox=(int(x1), int(y1), int(w), int(h)),
            score=float(confidence),
        )

    # ------------------------------------------------------------------
    # 资源管理
    # ------------------------------------------------------------------

    def close(self) -> None:
        """释放 YOLO 推理资源，同时关闭底层 VisionPerception。"""
        try:
            self._recognizer.close()
        except Exception:
            pass
        try:
            self._vision.close()
        except Exception:
            pass
