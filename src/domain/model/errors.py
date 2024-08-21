from http import HTTPStatus

from pydantic import ConfigDict


class BaseApiError(Exception):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    status_code: HTTPStatus

    def __init__(self, msg: str) -> None:
        self.msg = msg

    def __str__(self) -> str:
        return f"{self.__class__.__name__} has occurred. Message: {self.msg}"
