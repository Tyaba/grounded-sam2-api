from PIL import Image

from src.utils.image import base642pil, pil2base64


def test_pil_base64_convert(sample_image: Image.Image):
    image_base64 = pil2base64(sample_image)
    image = base642pil(image_base64)
    assert image == sample_image
