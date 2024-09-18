import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field


class SAM2Input(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    image: Image.Image
    input_boxes: np.ndarray
    crop_bbox: bool = Field(
        default=True, description="透過背景部分を消すかどうか。default to True"
    )


class SAM2Output(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    segments: list[Image.Image]
    mask_images: list[Image.Image]
    mask_arrays: np.ndarray
    scores: np.ndarray
    logits: np.ndarray
