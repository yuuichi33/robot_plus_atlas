"""Self-contained one-frame YOLOv5 task recognizer for Atlas devices."""

from .acl_backend import AclBackend, AclError
from .camera import CameraError, OpenCVCameraSource, parse_camera
from .constants import CLASS_NAMES, RecognitionCode
from .interfaces import FrameSource, InferenceBackend
from .recognizer import AtlasTaskRecognizer, select_result_code
from .types import Detection, LetterboxInfo

__all__ = [
    "AclBackend",
    "AclError",
    "AtlasTaskRecognizer",
    "CameraError",
    "CLASS_NAMES",
    "Detection",
    "FrameSource",
    "InferenceBackend",
    "LetterboxInfo",
    "OpenCVCameraSource",
    "RecognitionCode",
    "parse_camera",
    "select_result_code",
]
