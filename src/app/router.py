from fastapi import APIRouter, Depends

from src.app.injector import inject_grounded_sam
from src.user_interface.grounded_sam import (
    SegmentResponse,
    SegmentReuqest,
    SegmentUserInterface,
)

router = APIRouter()


@router.post("/segment")
def segment(
    request: SegmentReuqest,
    interface: SegmentUserInterface = Depends(inject_grounded_sam),
) -> SegmentResponse:
    return interface.segment(request)
