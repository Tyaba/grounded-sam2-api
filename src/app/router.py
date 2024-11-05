from fastapi import APIRouter

from src.app.injector import inject_grounded_sam
from src.user_interface.grounded_sam import (
    DetectRequest,
    DetectResponse,
    SegmentRequest,
    SegmentResponse,
)

router = APIRouter()
interface = inject_grounded_sam()


@router.post("/segment")
def segment(
    request: SegmentRequest,
) -> SegmentResponse:
    return interface.segment(request)


@router.post("/detect")
def detect(
    request: DetectRequest,
) -> DetectResponse:
    return interface.detect(request)
