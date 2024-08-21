import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict


class SAM2Input(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    image: Image.Image
    input_boxes: np.ndarray


class SAM2Output(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    segments: list[Image.Image]
    mask_images: list[Image.Image]
    mask_arrays: np.ndarray
    scores: np.ndarray
    logits: np.ndarray
