import pytest
from PIL import Image

from src.domain.model.gdino import GDINOInput, GDINOOutput
from src.domain.service.gdino import GDINOInterface


@pytest.mark.parametrize(
    "text_prompt, image, expected_num_item",
    [
        ("product.", Image.open("tests/data/one_bottle.png"), 1),
        ("product.", Image.open("tests/data/two_bottles.jpg"), 2),
        ("universe.", Image.open("tests/data/one_bottle.png"), 0),
        ("universe.", Image.open("tests/data/two_bottles.jpg"), 0),
    ],
)
def test_gdino_detect_all_objects(
    gdino: GDINOInterface, text_prompt: str, image: Image.Image, expected_num_item: int
):
    gdino_input = GDINOInput(
        image=image,
        text=text_prompt,
        box_threshold=0.75,
    )
    gdino_output: GDINOOutput = gdino.detect(gdino_input=gdino_input)
    assertion_error_msg = f"confidences: {gdino_output.confidences}, threshold: {gdino_input.box_threshold}"
    if expected_num_item == 0:
        assert not gdino_output.success
        assert len(gdino_output.boxes) == 1, assertion_error_msg
        assert len(gdino_output.class_ids) == 1, assertion_error_msg
        assert len(gdino_output.class_names) == 1, assertion_error_msg
        assert len(gdino_output.confidences) == 1, assertion_error_msg
        assert len(gdino_output.visualize_labels) == 1, assertion_error_msg
    else:
        assert len(gdino_output.boxes) == expected_num_item, assertion_error_msg
        assert len(gdino_output.class_ids) == expected_num_item, assertion_error_msg
        assert len(gdino_output.class_names) == expected_num_item, assertion_error_msg
        assert len(gdino_output.confidences) == expected_num_item, assertion_error_msg
        assert (
            len(gdino_output.visualize_labels) == expected_num_item
        ), assertion_error_msg
        assert gdino_output.success
