"""
视觉感知模块 —— 基于 OpenCV + EasyOCR。

功能：
1. detect_block()      – 仅依据几何形状检测竖直四方体柱子，不使用颜色
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
# OCR 引擎 —— 仅使用 EasyOCR
# ---------------------------------------------------------------------------
_easyocr_reader = None


def _init_ocr():
    """延迟初始化 EasyOCR，避免导入时的启动开销。"""
    global _easyocr_reader
    if _easyocr_reader is not None:
        return

    try:
        import easyocr  # type: ignore
        _easyocr_reader = easyocr.Reader(["ch_sim", "en"], gpu=False)
        print("[vision] OCR engine: EasyOCR")
    except ImportError:
        print("[vision] [WARN] EasyOCR not installed, text recognition disabled!")
    except Exception as exc:
        print(f"[vision] [WARN] EasyOCR init failed: {exc}")


def _ocr_text(image: np.ndarray) -> str:
    """对图像执行 EasyOCR，返回识别出的文本（拼接所有行）。"""
    _init_ocr()
    if _easyocr_reader is None:
        return ""

    results = _easyocr_reader.readtext(image)
    texts = [item[1] for item in results if item[2] > 0.2]
    return " ".join(texts)


# ---------------------------------------------------------------------------
# 颜色范围配置（HSV）—— 保留用于颜色标定工具 calibrate_color()
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# 四方体柱几何检测说明
# ---------------------------------------------------------------------------
# 目标柱检测完全不使用 HSV 颜色。候选由灰度边缘产生，再依据：
# 面积、竖直高宽比、矩形度、凸度、长轴竖直度和落地位置综合评分。


@dataclass
class VisionResult:
    found: bool
    center_x: int = 0
    center_y: int = 0
    area: float = 0.0
    distance_level: str = "unknown"  # "near" | "middle" | "far"
    label: str = ""                  # cuboid / OCR 原文 / QR 内容
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    aspect_ratio: float = 0.0
    rectangularity: float = 0.0
    solidity: float = 0.0
    verticality: float = 0.0
    score: float = 0.0


# ---------------------------------------------------------------------------
# 任务文字解析
# ---------------------------------------------------------------------------

def parse_task_text(raw_text: str) -> Tuple[bool, int, str]:
    """
    从 OCR 原始文本解析 (valid, position, attack)。

    标准任务文字仅使用：
      - 位置1/位置2
      - 劈砍/刺击

    内部仍使用 attack="chop" / "stab" 与动作编号映射对接。
    """
    text = raw_text.strip()
    if not text:
        return False, 0, ""

    import re

    # OCR 可能在汉字之间插入空格或换行，先生成紧凑文本用于中文短语匹配。
    compact = re.sub(r"\s+", "", text)
    lower = text.lower()

    pos_match = re.search(r"(?:pos|位置)\s*[=:：]?\s*([12])", text, re.IGNORECASE)
    if pos_match is None:
        pos_match = re.search(r"\b([12])\b", text)
    position = int(pos_match.group(1)) if pos_match else 0

    if position not in (1, 2):
        return False, position, ""

    # 先匹配完整动作短语，避免把任意"击"误判成刺击。
    if "刺击" in compact or "刺擊" in compact:
        return True, position, "stab"
    if "劈砍" in compact:
        return True, position, "chop"

    # 英文调试文本兼容。
    if any(word in lower for word in ("stab", "thrust", "pierce")):
        return True, position, "stab"
    if any(word in lower for word in ("chop", "slash", "cut")):
        return True, position, "chop"

    return False, position, ""


# ---------------------------------------------------------------------------
# 视觉感知主类
# ---------------------------------------------------------------------------

class VisionPerception:
    """真实摄像头视觉感知。目标柱按四方体几何特征检测，与颜色无关。"""

    def __init__(
        self,
        camera_id: int = 0,
        frame_width: int = 640,
        frame_height: int = 480,
        block_min_area: int = 1500,
        block_max_area: int = 250000,
        cuboid_min_aspect: float = 0.55,
        cuboid_max_aspect: float = 3.0,
        cuboid_min_rectangularity: float = 0.35,
        cuboid_min_solidity: float = 0.60,
        cuboid_min_score: float = 0.38,
        cuboid_min_bottom_ratio: float = 0.35,
        block_middle_area: float = 10000.0,
        block_near_area: float = 30000.0,
    ):
        self.camera_id = camera_id
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.block_min_area = block_min_area
        self.block_max_area = block_max_area
        self.cuboid_min_aspect = cuboid_min_aspect
        self.cuboid_max_aspect = cuboid_max_aspect
        self.cuboid_min_rectangularity = cuboid_min_rectangularity
        self.cuboid_min_solidity = cuboid_min_solidity
        self.cuboid_min_score = cuboid_min_score
        self.cuboid_min_bottom_ratio = cuboid_min_bottom_ratio
        self.block_middle_area = block_middle_area
        self.block_near_area = block_near_area

        self.cap: Optional[cv2.VideoCapture] = None
        self._last_frame: Optional[np.ndarray] = None
        self._last_annotated_frame: Optional[np.ndarray] = None
        self._last_block_candidates = []
        self._display_window_name = "Robot Mission Vision"

        self._ocr_frame_counter = 0
        self._ocr_skip_interval = 5
        self._ocr_max_width = 200  # ROI 缩放到此宽度再送 OCR，ARM 上提速
        self._paper_stable_count = 0
        self._paper_stable_threshold = 1  # 首次识别到就立即返回，避免绕行时丢帧

    # ---- 摄像头控制 ----

    def open(self) -> None:
        """打开摄像头，启动时可能需等待设备初始化，最多重试 10 次。"""
        import time as _time
        for attempt in range(10):
            self.cap = cv2.VideoCapture(self.camera_id)
            if self.cap.isOpened():
                break
            self.cap.release()
            self.cap = None
            if attempt < 9:
                print(f"[vision] camera not ready, retry {attempt+1}/10...")
                _time.sleep(2)
        if self.cap is None or not self.cap.isOpened():
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

    def discard_frames(self, count: int = 4, delay_s: float = 0.01) -> None:
        """底盘运动后丢弃摄像头缓存旧帧，确保下一次判断使用新画面。"""
        if self.cap is None:
            return
        for _ in range(max(0, count)):
            self.cap.grab()
            if delay_s > 0:
                time.sleep(delay_s)

    @staticmethod
    def _long_axis_verticality(rect) -> float:
        """返回最小外接矩形长轴接近竖直方向的程度，范围 0~1。"""
        box = cv2.boxPoints(rect)
        edges = []
        for i in range(4):
            p1 = box[i]
            p2 = box[(i + 1) % 4]
            dx = float(p2[0] - p1[0])
            dy = float(p2[1] - p1[1])
            length = (dx * dx + dy * dy) ** 0.5
            edges.append((length, dx, dy))
        _, dx, dy = max(edges, key=lambda item: item[0])
        denom = (dx * dx + dy * dy) ** 0.5
        return abs(dy) / denom if denom > 1e-6 else 0.0

    @staticmethod
    def _closeness(value: float, target: float, tolerance: float) -> float:
        return max(0.0, 1.0 - abs(value - target) / max(tolerance, 1e-6))

    # ---- 四方体柱检测（纯形状） ----

    def detect_block(self) -> VisionResult:
        """
        检测落地四方体目标。

        检测不使用固定颜色范围，也不判断 red/yellow 等颜色类别。
        为避免目标与背景在灰度上接近导致边缘消失，同时构造两类边缘：

        1. Lab-L 亮度边缘：覆盖灰色、白色等低彩度四方体；
        2. Lab-a/b 色度梯度边缘：仅用于恢复物体边界，不限定具体色相。

        候选最终仍按完整轮廓、落地位置、宽高比、矩形度、凸度、
        竖直性和是否接触画面边界进行几何筛选。
        """
        frame = self.read_frame()
        if frame is None:
            return VisionResult(found=False)

        frame_h, frame_w = frame.shape[:2]
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        channel_l, channel_a, channel_b = cv2.split(lab)

        # 亮度边缘：放宽阈值以捕获更多轮廓。
        l_blur = cv2.GaussianBlur(channel_l, (5, 5), 0)
        l_edges = cv2.Canny(l_blur, 25, 80)
        l_edges = cv2.morphologyEx(
            l_edges,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 7)),
            iterations=1,
        )
        l_edges = cv2.dilate(
            l_edges,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        )

        # 色度梯度边缘：不选择某种颜色，只检测 a/b 通道中的色度突变。
        # 这可以恢复"彩色目标与灰色地面亮度接近"时在灰度图中消失的轮廓。
        a_blur = cv2.GaussianBlur(channel_a, (5, 5), 0)
        b_blur = cv2.GaussianBlur(channel_b, (5, 5), 0)
        a_edges = cv2.Canny(a_blur, 10, 30)
        b_edges = cv2.Canny(b_blur, 10, 30)
        chroma_edges = cv2.bitwise_or(a_edges, b_edges)
        chroma_edges = cv2.morphologyEx(
            chroma_edges,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
            iterations=2,
        )
        chroma_edges = cv2.dilate(
            chroma_edges,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        )

        # 调试图显示两类边缘的并集，但候选分别提取，避免背景边缘与目标粘连。
        candidates = []

        def collect_candidates(edge_map: np.ndarray, source: str) -> None:
            contours, _ = cv2.findContours(
                edge_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for cnt in contours:
                area = float(cv2.contourArea(cnt))
                if area < self.block_min_area or area > self.block_max_area:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)
                if w < 25 or h < 35:
                    continue

                # 目标应完整出现在画面中。柜体、门框等背景结构通常接触图像边界。
                border_margin = 4
                touches_border = (
                    x <= border_margin
                    or y <= border_margin
                    or x + w >= frame_w - border_margin
                    or y + h >= frame_h - border_margin
                )
                if touches_border:
                    continue

                # 排除黑色区域（墙壁、阴影），L 通道均值 < 35 视为黑色
                roi_l = channel_l[y:y + h, x:x + w]
                if roi_l.size > 0 and np.mean(roi_l) < 35:
                    continue

                aspect = h / float(w)
                if not (self.cuboid_min_aspect <= aspect <= self.cuboid_max_aspect):
                    continue

                bbox_area = float(w * h)
                rectangularity = area / bbox_area if bbox_area > 0 else 0.0
                if rectangularity < self.cuboid_min_rectangularity:
                    continue

                hull = cv2.convexHull(cnt)
                hull_area = float(cv2.contourArea(hull))
                solidity = area / hull_area if hull_area > 0 else 0.0
                if solidity < self.cuboid_min_solidity:
                    continue

                peri = cv2.arcLength(cnt, True)
                if peri <= 1e-6:
                    continue
                approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)
                if len(approx) < 4 or len(approx) > 16:
                    continue

                rect = cv2.minAreaRect(cnt)
                verticality = self._long_axis_verticality(rect)
                if verticality < 0.45:
                    continue

                bottom_ratio = (y + h) / float(frame_h)
                center_y_ratio = (y + h / 2.0) / float(frame_h)
                if bottom_ratio < self.cuboid_min_bottom_ratio:
                    continue
                if center_y_ratio < 0.30:
                    continue

                # 当前实物在摄像头中的投影接近 h/w=1.1~1.4，
                # 不再使用旧版"细长立柱 h/w≈2.5"的错误先验。
                aspect_score = self._closeness(aspect, 1.25, 1.05)
                rect_score = min(1.0, max(0.0, (rectangularity - 0.42) / 0.48))
                solidity_score = min(1.0, max(0.0, (solidity - 0.72) / 0.28))
                floor_score = min(
                    1.0, max(0.0, (bottom_ratio - self.cuboid_min_bottom_ratio) / 0.35)
                )
                area_score = min(1.0, area / max(self.block_near_area, 1.0))
                vertex_score = 1.0 if 4 <= len(approx) <= 10 else 0.65
                source_score = 1.0 if source == "chroma" else 0.72

                score = (
                    0.22 * aspect_score
                    + 0.22 * rect_score
                    + 0.16 * solidity_score
                    + 0.14 * verticality
                    + 0.12 * floor_score
                    + 0.06 * area_score
                    + 0.04 * vertex_score
                    + 0.04 * source_score
                )
                if score < self.cuboid_min_score:
                    continue

                candidates.append({
                    "contour": cnt,
                    "bbox": (x, y, w, h),
                    "area": area,
                    "aspect": aspect,
                    "rectangularity": rectangularity,
                    "solidity": solidity,
                    "verticality": verticality,
                    "score": score,
                    "source": source,
                })

        # 先从色度梯度中恢复完整目标轮廓，再以亮度边缘作为无彩色目标的后备。
        collect_candidates(chroma_edges, "chroma")
        collect_candidates(l_edges, "luma")

        # 去除两类边缘产生的重复框。
        def bbox_iou(a, b) -> float:
            ax, ay, aw, ah = a
            bx, by, bw, bh = b
            x1 = max(ax, bx)
            y1 = max(ay, by)
            x2 = min(ax + aw, bx + bw)
            y2 = min(ay + ah, by + bh)
            iw = max(0, x2 - x1)
            ih = max(0, y2 - y1)
            inter = float(iw * ih)
            union = float(aw * ah + bw * bh) - inter
            return inter / union if union > 0 else 0.0

        candidates.sort(key=lambda item: item["score"], reverse=True)
        unique_candidates = []
        for cand in candidates:
            if any(bbox_iou(cand["bbox"], kept["bbox"]) > 0.55 for kept in unique_candidates):
                continue
            unique_candidates.append(cand)
        candidates = unique_candidates
        self._last_block_candidates = candidates

        annotated = frame.copy()
        cv2.line(
            annotated,
            (frame_w // 2, 0),
            (frame_w // 2, frame_h),
            (255, 255, 0),
            1,
        )
        for idx, cand in enumerate(candidates[:8]):
            x, y, w, h = cand["bbox"]
            color = (0, 255, 0) if idx == 0 else (0, 165, 255)
            thickness = 3 if idx == 0 else 1
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)
            cv2.putText(
                annotated,
                f"S={cand['score']:.2f} AR={cand['aspect']:.2f} {cand['source']}",
                (x, max(18, y - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
            )
        self._last_annotated_frame = annotated

        if not candidates:
            return VisionResult(found=False)

        best = candidates[0]
        x, y, w, h = best["bbox"]
        cx = x + w // 2
        cy = y + h // 2
        area = best["area"]

        if area >= self.block_near_area:
            dist = "near"
        elif area >= self.block_middle_area:
            dist = "middle"
        else:
            dist = "far"

        return VisionResult(
            found=True,
            center_x=cx,
            center_y=cy,
            area=area,
            distance_level=dist,
            label="cuboid",
            bbox=(x, y, w, h),
            aspect_ratio=best["aspect"],
            rectangularity=best["rectangularity"],
            solidity=best["solidity"],
            verticality=best["verticality"],
            score=best["score"],
        )

    def show_debug(self, state: str, result: Optional[VisionResult] = None) -> bool:
        """显示自动任务实时画面。返回 False 表示用户按下 q。"""
        is_block_state = any(key in state for key in ("SEARCH_BLOCK", "ALIGN_BLOCK", "APPROACH_BLOCK", "ORBIT_AND_SCAN"))
        base = (self._last_annotated_frame if is_block_state and self._last_annotated_frame is not None
                else self._last_frame)
        if base is None:
            return True
        display = base.copy()
        cv2.putText(display, f"STATE: {state}", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        if result is not None:
            text = f"found={result.found} area={result.area:.0f} dist={result.distance_level}"
            cv2.putText(display, text, (10, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            if result.found:
                error_x = result.center_x - self.frame_width // 2
                text2 = f"err_x={error_x} AR={result.aspect_ratio:.2f} R={result.rectangularity:.2f} S={result.score:.2f}"
                cv2.putText(display, text2, (10, 76),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)
                cv2.drawMarker(display, (result.center_x, result.center_y),
                               (0, 255, 0), cv2.MARKER_CROSS, 26, 2)
        cv2.imshow(self._display_window_name, display)
        return (cv2.waitKey(1) & 0xFF) != ord("q")

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
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1500:  # 放宽最小面积
                continue

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

        # 缩小 ROI 加速 OCR（ARM 上 EasyOCR 对大图很慢）
        rh2, rw2 = roi_cropped.shape[:2]
        if rw2 > self._ocr_max_width:
            scale = self._ocr_max_width / rw2
            roi_cropped = cv2.resize(roi_cropped, (self._ocr_max_width, int(rh2 * scale)))

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

    # ---- 二维码检测 ----

    def detect_qr(self) -> VisionResult:
        """
        检测二维码，解析内容（如 POS=1）。
        返回 VisionResult，成功时 label 为二维码内容。
        """
        frame = self.read_frame()
        if frame is None:
            return VisionResult(found=False)

        # 转灰度提高 QR 识别率
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detector = cv2.QRCodeDetector()

        # 方法1：单码检测
        data, bbox, _ = detector.detectAndDecode(gray)
        if not data:
            # 方法2：多码检测
            data_list, bbox_list, _ = detector.detectAndDecodeMulti(gray)
            if data_list and len(data_list) > 0:
                data = data_list[0]
                bbox = bbox_list[0] if bbox_list is not None else None

        if data and bbox is not None:
            pts = bbox.reshape(-1, 2).astype(int)
            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))
            qr_w = int(np.max(pts[:, 0]) - np.min(pts[:, 0]))
            qr_h = int(np.max(pts[:, 1]) - np.min(pts[:, 1]))
            print(f"[vision] QR detected: \"{data.strip()}\" area={qr_w*qr_h}")
            return VisionResult(
                found=True,
                center_x=cx,
                center_y=cy,
                area=float(qr_w * qr_h),
                label=data.strip(),
            )

        return VisionResult(found=False)


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
        label = (f"cuboid area={result.area:.0f} {result.distance_level} "
                 f"AR={result.aspect_ratio:.2f} score={result.score:.2f}")
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
    parser.add_argument("--mode", choices=["block", "text", "qr"], default="block")
    parser.add_argument("--no-display", action="store_true", help="不弹窗显示画面（Atlas 无头模式）")
    args = parser.parse_args()

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
                            f"  [{frame_idx}] [OK] cuboid found: label={result.label}, "
                            f"area={result.area:.0f}, center=({result.center_x},{result.center_y}), "
                            f"dist={result.distance_level}, AR={result.aspect_ratio:.2f}, "
                            f"rect={result.rectangularity:.2f}, score={result.score:.2f}"
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
                    base = (vp._last_annotated_frame
                            if args.mode == "block" and vp._last_annotated_frame is not None
                            else vp._last_frame)
                    display = _draw_detections(base, result, args.mode)
                    cv2.imshow(f"Vision - {args.mode}", display)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                time.sleep(0.05)

        except KeyboardInterrupt:
            print("\nExiting.")

        if not args.no_display:
            cv2.destroyAllWindows()
