import pytest

from src.domain.service.gdino import GDINOInterface
from src.domain.service.sam2 import SAM2Interface
from src.infrastructure.service.gdino import GDINO
from src.infrastructure.service.sam2 import SAM2


@pytest.fixture
def gdino() -> GDINOInterface:
    return GDINO()


@pytest.fixture
def sam2() -> SAM2Interface:
    return SAM2()
