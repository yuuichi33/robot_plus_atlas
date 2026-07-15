from __future__ import annotations

import numpy as np

from .constants import CLASS_NAMES, MAX_DETECTIONS, MODEL_OUTPUT_SHAPE
from .preprocess import restore_boxes
from .types import Detection, LetterboxInfo


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    result = np.empty_like(boxes, dtype=np.float32)
    result[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    result[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    result[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    result[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return result


def _box_iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    inter_x1 = np.maximum(box[0], boxes[:, 0])
    inter_y1 = np.maximum(box[1], boxes[:, 1])
    inter_x2 = np.minimum(box[2], boxes[:, 2])
    inter_y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, inter_x2 - inter_x1) * np.maximum(0.0, inter_y2 - inter_y1)
    area_a = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    area_b = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0, boxes[:, 3] - boxes[:, 1]
    )
    union = area_a + area_b - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def class_agnostic_nms(
    boxes: np.ndarray, scores: np.ndarray, iou_threshold: float, max_det: int
) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or scores.shape != (boxes.shape[0],):
        raise ValueError("boxes/scores shapes are incompatible")
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int64)
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    while order.size and len(keep) < max_det:
        current = int(order[0])
        keep.append(current)
        remaining = order[1:]
        if remaining.size == 0:
            break
        ious = _box_iou_one_to_many(boxes[current], boxes[remaining])
        order = remaining[ious <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def postprocess_yolov5(
    prediction: np.ndarray,
    info: LetterboxInfo,
    *,
    confidence_threshold: float,
    iou_threshold: float,
    max_det: int = MAX_DETECTIONS,
    timestamp: float = 0.0,
) -> list[Detection]:
    raw = np.asarray(prediction)
    if raw.shape != MODEL_OUTPUT_SHAPE:
        raise ValueError(f"expected model output {MODEL_OUTPUT_SHAPE}, got {raw.shape}")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1")
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be between 0 and 1")
    if max_det <= 0:
        raise ValueError("max_det must be positive")

    rows = raw[0].astype(np.float32, copy=False)
    valid = np.isfinite(rows).all(axis=1) & (rows[:, 2] > 0) & (rows[:, 3] > 0)
    rows = rows[valid]
    if rows.size == 0:
        return []
    class_ids = np.argmax(rows[:, 5:], axis=1)
    scores = rows[:, 4] * rows[np.arange(rows.shape[0]), 5 + class_ids]
    selected = scores >= confidence_threshold
    if not np.any(selected):
        return []
    boxes = xywh_to_xyxy(rows[selected, :4])
    scores = scores[selected]
    class_ids = class_ids[selected]
    keep = class_agnostic_nms(boxes, scores, iou_threshold, max_det)
    boxes = restore_boxes(boxes[keep], info)
    scores = scores[keep]
    class_ids = class_ids[keep]
    nonempty = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    detections: list[Detection] = []
    for box, score, class_id in zip(
        boxes[nonempty], scores[nonempty], class_ids[nonempty], strict=True
    ):
        cid = int(class_id)
        detections.append(
            Detection(
                class_id=cid,
                label=CLASS_NAMES[cid],
                confidence=float(score),
                xyxy=tuple(float(value) for value in box),
                timestamp=float(timestamp),
            )
        )
    return detections
