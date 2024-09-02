import base64
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def pil2cv(image: Image.Image) -> np.ndarray:
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image


def base642pil(image_base64: str) -> Image.Image:
    image_bytes = base64.b64decode(image_base64)
    image = Image.open(BytesIO(image_bytes))
    return image


def pil2base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def get_image_paths(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        print(f"{image_dir} does not exist. Please check the path and try again.")
    image_suffixes = {".jpg", ".png", ".jpeg", ".gif"}
    if image_dir.suffix in image_suffixes:
        print(f"{image_dir} is a file, not a directory. Use the file as an image.")
        return [image_dir]
    image_paths = []
    for image_suffix in image_suffixes:
        # recursive search for images in child directories
        image_paths += list(image_dir.glob(f"**/*{image_suffix}"))
    image_paths = sorted(image_paths)
    print(f"Loaded {len(image_paths)} images from {image_dir}")
    return image_paths
