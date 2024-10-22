from pydantic import BaseModel, ConfigDict, Field

from src.usecase.grounded_sam import GroundedSAM
from src.utils.image import base642pil
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SegmentRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    image: str
    text: str


class SegmentResponse(BaseModel):
    masks: list[list[list[int]]] = Field(
        default=...,
        title="segment mask",
        description="mask = masks[idx_segment][x][y]のmask",
    )


class SegmentUserInterface:
    def __init__(self, use_case: GroundedSAM) -> None:
        self.use_case = use_case

    def segment(self, request: SegmentRequest) -> SegmentResponse:
        logger.info("segment request received")
        image = base642pil(image_base64=request.image)
        sam2_output, gdino_output, _ = self.use_case.segment(
            image=image,
            text=request.text,
        )
        logger.info("segment done")
        return SegmentResponse(masks=sam2_output.mask_arrays.tolist())
