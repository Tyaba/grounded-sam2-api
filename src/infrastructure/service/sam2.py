import numpy as np
from PIL import Image

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from src.domain.model.sam2 import SAM2Input, SAM2Output
from src.domain.service.sam2 import SAM2Interface
from src.settings import Settings
from src.utils.cuda import setup_cuda

settings = Settings()


class SAM2(SAM2Interface):
    def __init__(
        self,
        model_cfg=settings.sam2_model_cfg,
        sam2_checkpoint=settings.sam2_checkpoint,
    ):
        setup_cuda()
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
        self.sam2_predictor = SAM2ImagePredictor(sam2_model)

    def segment(self, sam2_input: SAM2Input) -> SAM2Output:
        self.sam2_predictor.set_image(np.array(sam2_input.image.convert("RGB")))
        # results of SAM 2
        masks, scores, logits = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=sam2_input.input_boxes,
            multimask_output=False,
        )
        # convert the shape to (n, H, W)
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        masks = np.uint8(masks) * 255
        segments = []
        mask_images = []
        for mask in masks:
            mask_image = Image.fromarray(mask)
            segment = Image.composite(
                image1=sam2_input.image,
                image2=Image.new("RGB", sam2_input.image.size, color=(0, 0, 0)),
                mask=mask_image,
            )
            mask_images.append(mask_image)
            segments.append(segment)

        sam2_output = SAM2Output(
            segments=segments,
            mask_images=mask_images,
            mask_arrays=masks,
            scores=scores,
            logits=logits,
        )
        return sam2_output
