"""
视觉感知模块 —— 基于 OpenCV + PaddleOCR / EasyOCR。

功能：
1. detect_block()      – 检测纯色立方体柱子（HSV 颜色分割 + 轮廓）
2. read_task_text()    – 检测白色 A4 纸上的中文/英文任务文字
3. detect_qr()         – 检测二维码内容（如 POS=1 / POS=2）

摄像头：Hikvision USB 摄像头，在 Linux 上映射为 /dev/video0
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 尝试导入 OCR 库
# ARM64 (aarch64) 上 PaddlePaddle 有 segfault 风险，优先用 EasyOCR / Tesseract
# ---------------------------------------------------------------------------
_OCR_ENGINE = None  # "easyocr" | "paddle" | "tesseract" | None


def _is_aarch64() -> bool:
    """检测是否运行在 ARM64 架构上。"""
    import platform
    return platform.machine() in ("aarch64", "arm64", "armv7l", "armv8l")


def _init_ocr():
    """延迟初始化 OCR 引擎，避免导入时的启动开销。"""
    global _OCR_ENGINE
    if _OCR_ENGINE is not None:
        return

    is_arm = _is_aarch64()
    if is_arm:
        print("[vision] ARM64 detected, EasyOCR first, Tesseract fallback")

    # 1) EasyOCR（中文识别最准，ARM/x86 通用）
    try:
        import easyocr  # type: ignore

        _ocr = easyocr.Reader(["ch_sim", "en"], gpu=False)
        _OCR_ENGINE = ("easyocr", _ocr)
        print("[vision] OCR engine: EasyOCR")
        return
    except ImportError:
        print("[vision] EasyOCR not installed, trying Tesseract...")
    except Exception as exc:
        print(f"[vision] EasyOCR init failed: {exc}, trying Tesseract...")

    # 2) Tesseract（轻量快速，备选）
    try:
        import pytesseract  # type: ignore

        _OCR_ENGINE = ("tesseract", pytesseract)
        print("[vision] OCR engine: Tesseract")
        return
    except ImportError:
        print("[vision] pytesseract not installed.")
    except Exception as exc:
        print(f"[vision] Tesseract init failed: {exc}")

    # 4) PaddleOCR（仅 x86_64，ARM 上跳过避免 segfault）
    if not is_arm:
        try:
            from paddleocr import PaddleOCR  # type: ignore

            for kwargs in [
                {"lang": "ch", "use_textline_orientation": True},
                {"lang": "ch", "use_angle_cls": True},
                {"lang": "ch"},
            ]:
                try:
                    _ocr = PaddleOCR(**kwargs)
                    break
                except TypeError:
                    continue

            _OCR_ENGINE = ("paddle", _ocr)
            print("[vision] OCR engine: PaddleOCR")
            return
        except ImportError:
            print("[vision] PaddleOCR not installed.")
        except Exception as exc:
            print(f"[vision] PaddleOCR init failed: {exc}")

    print("[vision] [WARN] no OCR engine available, text recognition disabled!")


def _ocr_preprocess(image: np.ndarray) -> np.ndarray:
    """OCR 预处理：放大 + 锐化，提高小字识别率。"""
    h, w = image.shape[:2]
    # 如果图片太小（任一边 < 200px），放大 2 倍
    if h < 200 or w < 200:
        image = cv2.resize(image, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    # 锐化
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    image = cv2.filter2D(image, -1, kernel)
    return image


def _ocr_text(image: np.ndarray) -> str:
    """对图像执行 OCR，返回识别出的文本（拼接所有行）。"""
    _init_ocr()
    if _OCR_ENGINE is None:
        return ""

    engine_name, engine = _OCR_ENGINE

    if engine_name == "easyocr":
        results = engine.readtext(image)
        # 置信度过滤：只保留 > 0.2 的结果
        texts = [item[1] for item in results if item[2] > 0.2]
        return " ".join(texts)

    if engine_name == "tesseract":
        import pytesseract
        # 预处理：放大 + 锐化
        image = _ocr_preprocess(image)
        # PSM 6: 假设为均匀文本块，适合 A4 纸上的文字
        text = pytesseract.image_to_string(
            image, lang="chi_sim+eng", config="--psm 6"
        )
        # 如果 PSM 6 没结果，用 PSM 3（全自动）再试
        if not text.strip():
            text = pytesseract.image_to_string(
                image, lang="chi_sim+eng", config="--psm 3"
            )
        return text.strip()

    if engine_name == "paddle":
        try:
            results = engine.ocr(image)
        except TypeError:
            results = engine.ocr(image, cls=True)

        if results is None or len(results) == 0:
            return ""
        lines = []
        for group in results:
            if group is None:
                continue
            for line_info in group:
                text = line_info[1][0]
                lines.append(text)
        return " ".join(lines)

    return ""


# ---------------------------------------------------------------------------
# 颜色范围配置（HSV）
# ---------------------------------------------------------------------------
# 可以在运行时通过 set_block_color() 修改
_COLOR_RANGES = {
    "red": (np.array([0, 100, 100]), np.array([10, 255, 255])),
    "red2": (np.array([160, 100, 100]), np.array([180, 255, 255])),
    "blue": (np.array([100, 100, 100]), np.array([130, 255, 255])),
    "green": (np.array([40, 80, 80]), np.array([80, 255, 255])),
    "yellow": (np.array([20, 100, 100]), np.array([35, 255, 255])),
    "white": (np.array([0, 0, 180]), np.array([180, 30, 255])),
    "black": (np.array([0, 0, 0]), np.array([180, 255, 50])),
    "purple": (np.array([130, 50, 50]), np.array([160, 255, 255])),
    "orange": (np.array([10, 100, 100]), np.array([20, 255, 255])),
}

# 默认要检测的方块颜色（会尝试所有颜色，找到面积最大的）
DEFAULT_SCAN_COLORS = ["red", "blue", "green", "yellow", "purple", "orange"]


@dataclass
class VisionResult:
    found: bool
    center_x: int = 0
    center_y: int = 0
    area: int = 0
    distance_level: str = "unknown"  # "near" | "middle" | "far"
    label: str = ""  # 颜色标签 / 文字内容


# ---------------------------------------------------------------------------
# 任务文字解析
# ---------------------------------------------------------------------------

def parse_task_text(raw_text: str) -> Tuple[bool, int, str]:
    """
    从 OCR 识别的原始文本中解析出 (valid, position, attack)。
    必须同时识别到位置号(1/2)和动作词(劈砍/刺击)才返回 valid=True。
    """
    text = raw_text.strip()
    if not text:
        return False, 0, ""

    import re

    # 找位置号
    pos_match = re.search(r"(?:pos|位置)\s*[=:：]?\s*([12])", text, re.IGNORECASE)
    if pos_match is None:
        pos_match = re.search(r"\b([12])\b", text)
    position = int(pos_match.group(1)) if pos_match else 0

    if position not in (1, 2):
        return False, position, ""

    # 找动作词（必须找到）
    chop_words = ["劈", "砍", "chop", "slash", "cut", "斩"]
    stab_words = ["刺", "击", "stab", "thrust", "pierce", "捅"]

    text_lower = text.lower()
    for w in stab_words:
        if w in text_lower:
            return True, position, "stab"
    for w in chop_words:
        if w in text_lower:
            return True, position, "chop"

    # 找到位置但没找到动作，无效
    return False, position, ""


# ---------------------------------------------------------------------------
# 视觉感知主类
# ---------------------------------------------------------------------------

class VisionPerception:
    """
    真实摄像头视觉感知。
    用法:
        vp = VisionPerception(camera_id=0, block_colors=["red","blue"])
        result = vp.detect_block()
        task = vp.read_task_text()
    """

    def __init__(
        self,
        camera_id: int = 0,
        frame_width: int = 640,
        frame_height: int = 480,
        block_colors: Optional[list] = None,
        block_min_area: int = 5000,
        block_max_area: int = 200000,
        debug_dir: str = "",
    ):
        self.camera_id = camera_id
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.block_colors = block_colors or DEFAULT_SCAN_COLORS
        self.block_min_area = block_min_area
        self.block_max_area = block_max_area
        self.debug_dir = debug_dir

        self.cap: Optional[cv2.VideoCapture] = None
        self._last_frame: Optional[np.ndarray] = None
        self._debug_frame_count = 0

        # OCR 跳帧优化：每 N 帧才跑一次 OCR（检测仍然每帧跑）
        self._ocr_frame_counter = 0
        self._ocr_skip_interval = 3  # 每 3 帧 OCR 一次

        # 辅助：用于判断 A4 纸是否被检测到的连续帧计数
        self._paper_stable_count = 0
        self._paper_stable_threshold = 2  # 连续 2 帧确认（减少等待）

    # ---- 摄像头控制 ----

    def open(self) -> None:
        """打开摄像头。"""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"无法打开摄像头 /dev/video{self.camera_id}，"
                f"请检查摄像头连接和权限。"
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        print(f"[vision] 摄像头已打开: /dev/video{self.camera_id}")

    def close(self) -> None:
        if self.cap and self.cap.isOpened():
            self.cap.release()
            print("[vision] 摄像头已关闭")

    def __enter__(self) -> "VisionPerception":
        self.open()
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def read_frame(self) -> Optional[np.ndarray]:
        """读取一帧 BGR 图像。"""
        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None
        self._last_frame = frame
        return frame

    # ---- 方块检测 ----

    def detect_block(self) -> VisionResult:
        """
        在当前帧中检测纯色立方体柱子。
        返回 VisionResult，其中 found=True 表示检测到。
        """
        frame = self.read_frame()
        if frame is None:
            return VisionResult(found=False)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        best_contour = None
        best_area = 0
        best_label = ""

        for color_name in self.block_colors:
            masks = []
            if color_name in _COLOR_RANGES:
                low, high = _COLOR_RANGES[color_name]
                masks.append(cv2.inRange(hsv, low, high))
            # red 需要组合两个区间
            if color_name == "red":
                if "red" in _COLOR_RANGES:
                    low, high = _COLOR_RANGES["red"]
                    masks.append(cv2.inRange(hsv, low, high))
                if "red2" in _COLOR_RANGES:
                    low, high = _COLOR_RANGES["red2"]
                    masks.append(cv2.inRange(hsv, low, high))
                if len(masks) > 1:
                    mask = cv2.bitwise_or(masks[0], masks[1])
                elif len(masks) == 1:
                    mask = masks[0]
                else:
                    continue
            else:
                if not masks:
                    continue
                mask = masks[0]

            # 形态学去噪
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < self.block_min_area or area > self.block_max_area:
                    continue

                # 检查形状是否接近矩形（立方体柱子）
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                # 矩形有 4 个顶点，但我们放宽到 4~8（考虑透视变形）
                if len(approx) < 4 or len(approx) > 8:
                    continue

                # 检查宽高比是否合理（柱子高约 38cm，宽约 15cm，比例约 2.5:1）
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = h / w if w > 0 else 0
                if aspect_ratio < 1.2 or aspect_ratio > 4.0:
                    continue

                if area > best_area:
                    best_area = area
                    best_contour = cnt
                    best_label = color_name

        if best_contour is None:
            return VisionResult(found=False)

        # 计算中心点
        M = cv2.moments(best_contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            x, y, w, h = cv2.boundingRect(best_contour)
            cx = x + w // 2
            cy = y + h // 2

        # 根据面积估算距离等级
        if best_area > 30000:
            dist = "near"
        elif best_area > 10000:
            dist = "middle"
        else:
            dist = "far"

        return VisionResult(
            found=True,
            center_x=cx,
            center_y=cy,
            area=best_area,
            distance_level=dist,
            label=best_label,
        )

    # ---- A4 白纸文字检测 ----

    def read_task_text(self) -> VisionResult:
        """
        检测白色 A4 纸上的文字，返回 VisionResult。
        如果识别到有效任务文字，found=True，label 为原始文本。
        """
        frame = self.read_frame()
        if frame is None:
            return VisionResult(found=False)

        # 1) 检测白色区域（A4 纸）
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 宽松的白色范围：低饱和度 + 高亮度
        white_low = np.array([0, 0, 140])
        white_high = np.array([180, 60, 255])
        white_mask = cv2.inRange(hsv, white_low, white_high)

        # 形态学去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 找面积最大的白色四边形区域
        best_cnt = None
        best_area = 0
        total_white_blobs = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1500:  # 放宽最小面积
                continue
            total_white_blobs += 1

            # 近似矩形检查
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            if len(approx) < 4 or len(approx) > 8:
                continue

            if area > best_area:
                best_area = area
                best_cnt = cnt

        if best_cnt is None:
            self._paper_stable_count = 0
            if self._debug_frame_count < 10 and self.debug_dir:
                self._save_debug_pair(frame, white_mask, "paper_none")
                print(f"[vision] white blobs found: {total_white_blobs}, best_area: {best_area}")
            return VisionResult(found=False)

        # 2) 提取白色区域并做 OCR（只做一次，避免多次 OCR 卡顿）
        x, y, w, h = cv2.boundingRect(best_cnt)
        margin = 15
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(frame.shape[1] - x, w + 2 * margin)
        h = min(frame.shape[0] - y, h + 2 * margin)
        roi = frame[y : y + h, x : x + w]

        if roi.size == 0:
            self._paper_stable_count = 0
            return VisionResult(found=False)

        # 跳帧优化：不是每帧都 OCR
        self._ocr_frame_counter += 1
        if self._ocr_frame_counter % self._ocr_skip_interval != 0:
            return VisionResult(found=False)

        # 中央裁切：只保留画面中央 3/4，裁掉边缘噪声
        rh, rw = roi.shape[:2]
        crop_t = rh // 8
        crop_b = rh - rh // 8
        crop_l = rw // 8
        crop_r = rw - rw // 8
        roi_cropped = roi[crop_t:crop_b, crop_l:crop_r] if rh > 40 and rw > 40 else roi

        raw_text = _ocr_text(roi_cropped)

        if raw_text:
            print(f"[vision] OCR raw: \"{raw_text[:60]}\"")
        else:
            self._paper_stable_count = 0
            return VisionResult(found=False)

        valid, position, attack = parse_task_text(raw_text)
        if not valid:
            self._paper_stable_count = 0
            return VisionResult(found=False)

        self._paper_stable_count += 1
        if self._paper_stable_count >= self._paper_stable_threshold:
            return VisionResult(
                found=True,
                center_x=x + w // 2,
                center_y=y + h // 2,
                area=best_area,
                label=raw_text,
            )

        return VisionResult(found=False)

        return VisionResult(found=False)

    # ---- 二维码检测 ----

    def detect_qr(self) -> VisionResult:
        """
        检测二维码，解析内容（如 POS=1）。
        返回 VisionResult，成功时 label 为二维码内容。
        """
        frame = self.read_frame()
        if frame is None:
            return VisionResult(found=False)

        detector = cv2.QRCodeDetector()
        data, bbox, _ = detector.detectAndDecode(frame)

        if data and bbox is not None:
            # bbox shape: (1, 4, 2) 或 (4, 2)
            pts = bbox.reshape(-1, 2).astype(int)
            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))
            return VisionResult(
                found=True,
                center_x=cx,
                center_y=cy,
                area=0,
                label=data.strip(),
            )

        return VisionResult(found=False)

    # ---- 调试辅助 ----

    def save_debug_frame(self, filepath: str = "debug_frame.jpg") -> None:
        """保存当前帧用于调试。"""
        if self._last_frame is not None:
            cv2.imwrite(filepath, self._last_frame)
            print(f"[vision] debug frame saved: {filepath}")
        else:
            print("[vision] no frame available")

    def _save_debug_pair(
        self, frame: np.ndarray, mask: np.ndarray, tag: str, roi: Optional[np.ndarray] = None
    ) -> None:
        """保存调试帧：原图 + 白色掩码 + 可选的 ROI 区域。"""
        import os
        os.makedirs(self.debug_dir, exist_ok=True)
        idx = self._debug_frame_count
        self._debug_frame_count += 1
        cv2.imwrite(os.path.join(self.debug_dir, f"{tag}_{idx}_frame.jpg"), frame)
        cv2.imwrite(os.path.join(self.debug_dir, f"{tag}_{idx}_mask.jpg"), mask)
        if roi is not None and roi.size > 0:
            cv2.imwrite(os.path.join(self.debug_dir, f"{tag}_{idx}_roi.jpg"), roi)
        print(f"[vision] debug frame saved: {self.debug_dir}/{tag}_{idx}_*.jpg")


# ---------------------------------------------------------------------------
# 颜色标定工具
# ---------------------------------------------------------------------------

def calibrate_color(camera_id: int = 0):
    """
    交互式颜色标定：在窗口中点击要检测的颜色区域，输出 HSV 范围。
    按 'q' 退出。
    """
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"无法打开摄像头 /dev/video{camera_id}")
        return

    print("颜色标定工具")
    print("  用鼠标点击画面中要检测的颜色区域")
    print("  按 'q' 退出")

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            hsv = cv2.cvtColor(param["frame"], cv2.COLOR_BGR2HSV)
            pixel = hsv[y, x]
            print(f"  点击位置 ({x}, {y}) → HSV = ({pixel[0]}, {pixel[1]}, {pixel[2]})")
            # 给出建议范围
            h_low = max(0, int(pixel[0]) - 15)
            h_high = min(180, int(pixel[0]) + 15)
            s_low = max(0, int(pixel[1]) - 40)
            v_low = max(0, int(pixel[2]) - 40)
            print(
                f"  建议范围: lower=({h_low}, {s_low}, {v_low}), "
                f"upper=({h_high}, 255, 255)"
            )

    cv2.namedWindow("color_calibrate")
    frame_data = {"frame": None}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_data["frame"] = frame
        cv2.setMouseCallback("color_calibrate", on_mouse, frame_data)
        cv2.imshow("color_calibrate", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def _draw_detections(frame: np.ndarray, result: VisionResult, mode: str) -> np.ndarray:
    """在帧上绘制检测结果叠加层。"""
    display = frame.copy()

    if mode == "block" and result.found:
        cx, cy = result.center_x, result.center_y
        cv2.drawMarker(display, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 30, 2)
        label = f"{result.label} area={result.area:.0f} {result.distance_level}"
        cv2.putText(display, label, (cx - 60, cy - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    elif mode == "text":
        if result.found:
            cv2.drawMarker(display, (result.center_x, result.center_y),
                           (255, 0, 0), cv2.MARKER_CROSS, 20, 2)
            cv2.putText(display, f"TEXT: {result.label[:30]}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        else:
            cv2.putText(display, "scanning for A4 paper...",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    elif mode == "qr":
        if result.found:
            cv2.putText(display, f"QR: {result.label}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            cv2.putText(display, "scanning for QR...",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return display


# ---------------------------------------------------------------------------
# 独立测试入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="摄像头设备 ID")
    parser.add_argument("--mode", choices=["block", "text", "qr", "calibrate"], default="block")
    parser.add_argument("--no-display", action="store_true", help="不弹窗显示画面（Atlas 无头模式）")
    parser.add_argument("--debug-dir", default=".", help="调试帧保存目录")
    args = parser.parse_args()

    if args.mode == "calibrate":
        calibrate_color(args.camera)
        exit(0)

    with VisionPerception(camera_id=args.camera) as vp:
        print(f"Mode: {args.mode}, press 'q' to quit")
        if not args.no_display:
            cv2.namedWindow(f"Vision - {args.mode}", cv2.WINDOW_NORMAL)

        frame_idx = 0

        try:
            while True:
                frame_idx += 1

                if args.mode == "block":
                    result = vp.detect_block()
                    if result.found:
                        print(
                            f"  [{frame_idx}] [OK] block found: color={result.label}, "
                            f"area={result.area:.0f}, center=({result.center_x},{result.center_y}), "
                            f"dist={result.distance_level}"
                        )
                    else:
                        if frame_idx % 10 == 1:
                            print(f"  [{frame_idx}] [--] block not found")

                elif args.mode == "text":
                    result = vp.read_task_text()
                    if result.found:
                        valid, pos, att = parse_task_text(result.label)
                        print(
                            f"  [{frame_idx}] [OK] A4 text: \"{result.label}\" "
                            f"-> position={pos}, attack={att}"
                        )
                    else:
                        if frame_idx % 5 == 1:
                            print(f"  [{frame_idx}] [--] A4 paper or text not found")

                elif args.mode == "qr":
                    result = vp.detect_qr()
                    if result.found:
                        print(f"  [{frame_idx}] [OK] QR code: {result.label}")
                    else:
                        if frame_idx % 10 == 1:
                            print(f"  [{frame_idx}] [--] QR code not found")

                # 显示画面
                if not args.no_display and vp._last_frame is not None:
                    display = _draw_detections(vp._last_frame, result, args.mode)
                    cv2.imshow(f"Vision - {args.mode}", display)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                time.sleep(0.05)

        except KeyboardInterrupt:
            print("\nExiting.")

        if not args.no_display:
            cv2.destroyAllWindows()
