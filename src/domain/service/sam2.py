from abc import ABCMeta, abstractmethod

from src.domain.model.sam2 import SAM2Input, SAM2Output


class SAM2Interface(metaclass=ABCMeta):
    @abstractmethod
    def segment(self, sam2_input: SAM2Input) -> SAM2Output:
        pass
