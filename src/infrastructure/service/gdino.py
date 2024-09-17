import numpy as np
import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from src.domain.model.gdino import GDINOInput, GDINOOutput
from src.domain.service.gdino import GDINOInterface
from src.settings import Settings
from src.utils.cuda import setup_cuda
from src.utils.logger import get_logger

logger = get_logger(__name__)

settings = Settings()


class GDINO(GDINOInterface):
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
        try:
            with torch.no_grad():
                gdino_outputs = self.grounding_model(**inputs)
        except RuntimeError as e:
            logger.warning(f"grounding dinoがerrorをraiseしました: {e}")
            logger.warning("画像全体のbbox responseを返します")
            return GDINOOutput(
                boxes=np.array([[0, 0, gdino_input.image.size[0], gdino_input.image.size[1]]]),
                class_ids=np.array([0]),
                class_names=[""],
                confidences=[0.0],
                visualize_labels=[""],
                success=False,
            )
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
            success=True,
        )
        return gdino_output
