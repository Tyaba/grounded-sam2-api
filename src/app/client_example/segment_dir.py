"""example script to segment images in a directory
Usage:
    python -m src.app.client_example.segment_dir \
        --src-dir path/to/src_dir \
        --tgt-dir path/to/tgt_dir \
        --prompt "product."
"""

import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from tqdm import tqdm

from src.user_interface.grounded_sam import SegmentReuqest
from src.utils.image import get_image_paths, pil2base64

PORT = os.getenv("PORT", 58080)


def segment_image(image: Image.Image, prompt: str) -> Image.Image:
    segment_request = SegmentReuqest(
        image=pil2base64(image),
        text=prompt,
    )
    segment_response = requests.post(
        url=f"http://localhost:{PORT}/segment",
        json=segment_request.model_dump(),
    )

    mask_ar = np.uint8(np.array(segment_response.json()["masks"]))
    mask = Image.fromarray(mask_ar[0])

    segmented_image = Image.composite(
        image1=image,
        image2=Image.new("RGBA", image.size),
        mask=mask,
    )
    return segmented_image


def segment_dir(src_dir: Path, tgt_dir: Path, prompt: str) -> None:
    tgt_dir.mkdir(parents=True, exist_ok=True)
    image_paths = get_image_paths(src_dir)
    for image_path in tqdm(image_paths):
        relative_path = image_path.relative_to(src_dir)
        save_path = tgt_dir / relative_path.with_suffix(".png")
        image = Image.open(image_path)
        segmented_image = segment_image(image=image, prompt=prompt)
        segmented_image.save(save_path)


def main(args: Namespace) -> None:
    segment_dir(
        src_dir=args.src_dir,
        tgt_dir=args.tgt_dir,
        prompt=args.prompt,
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
    args = parser.parse_args()
    main(args=args)
