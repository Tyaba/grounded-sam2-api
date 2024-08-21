from pydantic import BaseModel, ConfigDict

from src.usecase.grounded_sam import GroundedSAM
from src.utils.image import base642pil


class SegmentReuqest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    image: str
    text: str


class SegmentResponse(BaseModel):
    masks: list[list[list[int]]]


class SegmentUserInterface:
    def __init__(self, use_case: GroundedSAM) -> None:
        self.use_case = use_case

    def segment(self, request: SegmentReuqest) -> SegmentResponse:
        image = base642pil(image_base64=request.image)
        use_case_output, _ = self.use_case.segment(
            image=image,
            text=request.text,
        )
        return SegmentResponse(masks=use_case_output.mask_arrays.tolist())
