import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field


class GDINOInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    image: Image.Image
    text: str
    box_threshold: float = Field(
        default=0.2,
        description="https://huggingface.co/docs/transformers/model_doc/grounding-dino#transformers.GroundingDinoProcessor.post_process_grounded_object_detection",
    )
    text_threshold: float = Field(
        default=0.2,
        description="https://huggingface.co/docs/transformers/model_doc/grounding-dino#transformers.GroundingDinoProcessor.post_process_grounded_object_detection",
    )


class GDINOOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    boxes: np.ndarray
    class_ids: np.ndarray
    class_names: list[str]
    confidences: list[float]
    visualize_labels: list[str]
