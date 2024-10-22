"""example script to segment images in a directory
Usage:
    python -m src.app.client_example.segment_dir \
        --api-url http://localhost:58080 \
        --src-dir path/to/src_dir \
        --tgt-dir path/to/tgt_dir \
        --prompt "product."
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from tqdm import tqdm

from src.user_interface.grounded_sam import SegmentRequest, SegmentResponse
from src.utils.image import get_image_paths, pil2base64
from src.utils.logger import get_logger

logger = get_logger(__name__)


def segment_image(image: Image.Image, prompt: str) -> list[Image.Image]:
    segment_request = SegmentRequest(
        image=pil2base64(image),
        text=prompt,
    )
    response = requests.post(
        url=f"{args.api_url.remove_suffix('/')}/segment",
        json=segment_request.model_dump(),
    )
    segment_response = SegmentResponse.model_validate(response.json())
    mask_arrays = np.uint8(np.array(segment_response.masks))
    segmented_images = []
    for mask_array in mask_arrays:
        mask = Image.fromarray(mask_array)
        segmented_image = Image.composite(
            image1=image.convert("RGBA"),
            image2=Image.new("RGBA", image.size),
            mask=mask,
        )
        segmented_image = segmented_image.crop(segmented_image.getbbox())
        segmented_images.append(segmented_image)
    return segmented_images


def segment_dir(
    src_dir: Path,
    tgt_dir: Path,
    prompt: str,
    overwrite: bool = False,
) -> None:
    tgt_dir.mkdir(parents=True, exist_ok=True)
    image_paths = get_image_paths(src_dir)
    for image_path in tqdm(image_paths):
        relative_path = image_path.relative_to(src_dir)
        save_dir = tgt_dir / relative_path.stem
        if save_dir.exists() and not overwrite:
            logger.info(f"{save_dir} exists. skip")
            continue
        save_dir.mkdir(parents=True, exist_ok=True)
        image = Image.open(image_path)
        segmented_images = segment_image(image=image, prompt=prompt)
        for i, segmented_image in enumerate(segmented_images):
            save_path = save_dir / f"{i}.png"
            segmented_image.save(save_path)


def main(args: Namespace) -> None:
    segment_dir(
        src_dir=args.src_dir,
        tgt_dir=args.tgt_dir,
        prompt=args.prompt,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--src-dir",
        type=Path,
        help="src images dir",
        required=True,
    )
    parser.add_argument(
        "--tgt-dir",
        type=Path,
        help="tgt images dir",
        required=True,
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="prompt",
        required=False,
        default="product.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="既存のファイルを上書きする",
    )
    args = parser.parse_args()
    main(args=args)
