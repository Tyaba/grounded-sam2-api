from abc import ABCMeta, abstractmethod

from src.domain.model.gdino import GDINOInput, GDINOOutput


class GDINOInterface(metaclass=ABCMeta):
    @abstractmethod
    def detect(self, gdino_input: GDINOInput) -> GDINOOutput:
        pass
