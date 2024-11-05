"""python src/usecase/gsam.py \
    --image-path notebooks/images/abema_water.png \
    --text a 'product.'
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

import supervision as sv
from PIL import Image
from supervision.draw.color import ColorPalette

from src.domain.model.gdino import GDINOInput, GDINOOutput
from src.domain.model.sam2 import SAM2Input, SAM2Output
from src.domain.service.gdino import GDINOInterface
from src.domain.service.sam2 import SAM2Interface
from src.infrastructure.service.gdino import GDINO
from src.infrastructure.service.sam2 import SAM2
from src.utils.image import pil2cv
from src.utils.logger import get_logger
from utils.supervision_utils import CUSTOM_COLOR_MAP

logger = get_logger(__name__)


class GroundedSAM:
    def __init__(self, gdino: GDINOInterface, sam2: SAM2Interface) -> None:
        self.gdino = gdino
        self.sam2 = sam2

    def detect(self, image: Image.Image, text: str, **kwargs: Any) -> GDINOOutput:
        logger.info(f"画像(サイズ{image.size})から{text}のbboxを検出します")
        gdino_input = GDINOInput(image=image, text=text, **kwargs)
        return self.gdino.detect(gdino_input)

    def segment(
        self,
        image: Image.Image,
        text: str,
        visualize: bool = False,
        gdino_kwargs: dict[str, Any] = {},
        sam2_kwargs: dict[str, Any] = {},
    ) -> tuple[SAM2Output, GDINOOutput, Image.Image | None]:
        logger.info(f"画像(サイズ{image.size})から{text}のsegmentationをします")
        gdino_output = self.detect(image, text, **gdino_kwargs)
        sam2_input = SAM2Input(
            image=image,
            input_boxes=gdino_output.boxes,
            **sam2_kwargs,
        )
        sam2_output = self.sam2.segment(sam2_input)
        visualization = None
        if visualize:
            visualization = self._visualize_segment(
                image=image, sam2_output=sam2_output, gdino_output=gdino_output
            )
        return sam2_output, gdino_output, visualization

    def _visualize_segment(
        self, image: Image.Image, sam2_output: SAM2Output, gdino_output: GDINOOutput
    ) -> Image.Image:
        detections = sv.Detections(
            xyxy=gdino_output.boxes,  # (n, 4)
            mask=sam2_output.mask_arrays.astype(bool),  # (n, h, w)
            class_id=gdino_output.class_ids,
        )
        box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        annotated_frame = box_annotator.annotate(
            scene=pil2cv(image=image), detections=detections
        )
        label_annotator = sv.LabelAnnotator(
            color=ColorPalette.from_hex(CUSTOM_COLOR_MAP)
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=gdino_output.visualize_labels,
        )
        mask_annotator = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        annotated_frame = mask_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        annotated_image = Image.fromarray(annotated_frame)
        return annotated_image


def main(args: Namespace):
    gdino = GDINO()
    sam2 = SAM2()
    gsam = GroundedSAM(gdino=gdino, sam2=sam2)
    image = Image.open(args.image_path)
    sam2_output = gsam.segment(image, args.text)
    print(sam2_output)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--image-path",
        type=Path,
        help="path",
        required=True,
    )
    parser.add_argument(
        "--text",
        type=str,
        help="gdino input text. ex) 'a product.'",
        default="a product.",
    )
    args = parser.parse_args()
    main(args=args)
