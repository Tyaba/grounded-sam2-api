import numpy as np
import pytest
from PIL import Image
from pytest_mock import MockFixture

from src.domain.model.gdino import GDINOOutput
from src.domain.model.sam2 import SAM2Output
from src.domain.service.gdino import GDINOInterface
from src.domain.service.sam2 import SAM2Interface
from src.usecase.grounded_sam import GroundedSAM


# 単体テスト（pytest-mockを使用）
@pytest.mark.parametrize(
    "text_prompt, image_path, expected_num_item",
    [
        ("product.", "tests/data/one_bottle.png", 1),
        ("product.", "tests/data/two_bottles.jpg", 2),
        ("universe.", "tests/data/one_bottle.png", 0),
    ],
)
def test_grounded_sam_segment_unit(
    mocker: MockFixture,
    text_prompt: str,
    image_path: str,
    expected_num_item: int,
):
    # モックの設定
    mock_gdino = mocker.Mock(spec=GDINOInterface)
    mock_sam2 = mocker.Mock(spec=SAM2Interface)

    # GDINOの出力をモック
    mock_gdino.detect.return_value = GDINOOutput(
        boxes=np.array([[0, 0, 100, 100]] * expected_num_item),
        class_ids=np.array([0] * expected_num_item),
        class_names=["product"] * expected_num_item,
        confidences=[0.9] * expected_num_item,
        visualize_labels=["product 0.9"] * expected_num_item,
        success=True,
    )

    # SAM2の出力をモック
    mock_sam2.segment.return_value = SAM2Output(
        segments=[Image.new("L", (100, 100), 0) for _ in range(expected_num_item)],
        mask_images=[Image.new("L", (100, 100), 0) for _ in range(expected_num_item)],
        mask_arrays=np.array([[1]] * expected_num_item),
        scores=np.array([0.9] * expected_num_item),
        logits=np.array([[0.1, 0.9]] * expected_num_item),
        success=True,
    )

    grounded_sam = GroundedSAM(gdino=mock_gdino, sam2=mock_sam2)
    image = Image.open(image_path)
    sam2_output, gdino_output, _ = grounded_sam.segment(image, text_prompt)

    assert len(sam2_output.mask_arrays) == expected_num_item

    # モックが正しく呼び出されたことを確認
    mock_gdino.detect.assert_called_once()
    mock_sam2.segment.assert_called_once()


# 統合テスト（実際のGDINOとSAM2を使用）
@pytest.mark.parametrize(
    "text_prompt, image_path, expected_num_item",
    [
        ("product.", "tests/data/one_bottle.png", 1),
        ("product.", "tests/data/two_bottles.jpg", 2),
        ("universe.", "tests/data/one_bottle.png", 0),
    ],
)
def test_grounded_sam_segment_integration(
    gdino: GDINOInterface,
    sam2: SAM2Interface,
    text_prompt: str,
    image_path: str,
    expected_num_item: int,
):
    grounded_sam = GroundedSAM(gdino=gdino, sam2=sam2)

    image = Image.open(image_path)
    sam2_output, gdino_output, visualization = grounded_sam.segment(
        image, text_prompt, visualize=True, gdino_kwargs={"box_threshold": 0.75}
    )
    if expected_num_item == 0:
        assert not gdino_output.success
    else:
        assert len(sam2_output.mask_arrays) == expected_num_item
    assert visualization is not None
    assert isinstance(visualization, Image.Image)
