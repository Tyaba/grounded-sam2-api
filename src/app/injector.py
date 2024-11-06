from functools import lru_cache

from src.user_interface.grounded_sam import GSAM2UserInterface
from src.utils.logger import get_logger

logger = get_logger(__name__)


@lru_cache()
def inject_grounded_sam() -> GSAM2UserInterface:
    from src.infrastructure.service.gdino import GDINO
    from src.infrastructure.service.sam2 import SAM2
    from src.usecase.grounded_sam import GroundedSAM

    logger.info("Initializing GroundedSAM2")
    gdino = GDINO()
    sam2 = SAM2()
    grounded_sam = GroundedSAM(gdino=gdino, sam2=sam2)
    user_interface = GSAM2UserInterface(use_case=grounded_sam)
    logger.info("GroundedSAM2 initialized")
    return user_interface
