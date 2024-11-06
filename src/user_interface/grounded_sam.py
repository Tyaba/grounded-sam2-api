from pydantic import BaseModel, ConfigDict, Field

from src.usecase.grounded_sam import GroundedSAM
from src.utils.image import base642pil
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DetectRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    image: str
    text: str


class DetectResponse(BaseModel):
    boxes: list[tuple[int, int, int, int]] = Field(
        default=...,
        title="detected boxes",
        description="box = boxes[idx_box]",
    )


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


class GSAM2UserInterface:
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

    def detect(self, request: DetectRequest) -> DetectResponse:
        logger.info("detect request received")
        image = base642pil(image_base64=request.image)
        gdino_output = self.use_case.detect(image=image, text=request.text)
        logger.info("detect done")
        boxes = [tuple(int(x) for x in box) for box in gdino_output.boxes.tolist()]
        return DetectResponse(boxes=boxes)
