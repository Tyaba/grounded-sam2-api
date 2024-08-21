import pytest
from PIL import Image


@pytest.fixture
def sample_image():
    return Image.open("tests/data/sample_image.png")
