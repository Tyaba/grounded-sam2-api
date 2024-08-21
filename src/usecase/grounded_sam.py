"""python src/usecase/gsam.py \
    --image-path notebooks/images/abema_water.png \
    --text a 'product.'
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path

import supervision as sv
from PIL import Image
from supervision.draw.color import ColorPalette

from src.domain.model.gdino import GDINOInput, GDINOOutput
from src.domain.model.sam2 import SAM2Input, SAM2Output
from src.infrastructure.service.gdino import GDINO
from src.infrastructure.service.sam2 import SAM2
from src.utils.image import pil2cv
from utils.supervision_utils import CUSTOM_COLOR_MAP


class GroundedSAM:
    def __init__(self, gdino: GDINO, sam2: SAM2) -> None:
        self.gdino = gdino
        self.sam2 = sam2

    def segment(
        self, image: Image.Image, text: str, visualize: bool = False
    ) -> tuple[SAM2Output, Image.Image | None]:
        gdino_input = GDINOInput(image=image, text=text)
        gdino_output: GDINOOutput = self.gdino.detect(gdino_input)
        sam2_input = SAM2Input(image=gdino_input.image, input_boxes=gdino_output.boxes)
        sam2_output = self.sam2.predict(sam2_input)
        visualization = None
        if visualize:
            visualization = self.visualize_segment(
                image=image, sam2_output=sam2_output, gdino_output=gdino_output
            )
        return sam2_output, visualization

    def visualize_segment(
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
