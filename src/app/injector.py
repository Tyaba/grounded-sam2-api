from functools import lru_cache

from src.user_interface.grounded_sam import GSAM2UserInterface


@lru_cache()
def inject_grounded_sam() -> GSAM2UserInterface:
    from src.infrastructure.service.gdino import GDINO
    from src.infrastructure.service.sam2 import SAM2
    from src.usecase.grounded_sam import GroundedSAM

    gdino = GDINO()
    sam2 = SAM2()
    grounded_sam = GroundedSAM(gdino=gdino, sam2=sam2)
    user_interface = GSAM2UserInterface(use_case=grounded_sam)
    return user_interface
