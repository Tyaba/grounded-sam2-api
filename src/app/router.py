from fastapi import APIRouter, Depends

from src.app.injector import inject_grounded_sam
from src.user_interface.grounded_sam import (
    SegmentRequest,
    SegmentResponse,
    SegmentUserInterface,
)

router = APIRouter()


@router.post("/segment")
def segment(
    request: SegmentRequest,
    interface: SegmentUserInterface = Depends(inject_grounded_sam),
) -> SegmentResponse:
    return interface.segment(request)
