import numpy as np
import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from src.domain.model.gdino import GDINOInput, GDINOOutput
from src.settings import Settings
from src.utils.cuda import setup_cuda

settings = Settings()


class GDINO:
    def __init__(
        self, model_id: str = settings.gdino_model_id, device: str = settings.device
    ) -> None:
        setup_cuda()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id
        ).to(device)
        self.device = device

    def detect(self, gdino_input: GDINOInput):
        inputs = self.processor(
            images=gdino_input.image, text=gdino_input.text, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            gdino_outputs = self.grounding_model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            gdino_outputs,
            inputs.input_ids,
            box_threshold=gdino_input.box_threshold,
            text_threshold=gdino_input.text_threshold,
            target_sizes=[gdino_input.image.size[::-1]],
        )
        boxes = results[0]["boxes"].cpu().numpy()
        confidences = results[0]["scores"].cpu().numpy().tolist()
        class_names = results[0]["labels"]
        class_ids = np.array(list(range(len(class_names))))
        visualize_labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence in zip(class_names, confidences)
        ]
        gdino_output = GDINOOutput(
            boxes=boxes,
            class_ids=class_ids,
            class_names=class_names,
            confidences=confidences,
            visualize_labels=visualize_labels,
        )
        return gdino_output
