import json
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path
from types import TracebackType

DATEFMT = "%Y-%m-%d %H:%M:%S"
LOG_DIR = Path("/tmp/log/python")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILENAME = LOG_DIR / f"{datetime.now().strftime(DATEFMT)}.log"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record=record)
        body = {
            "severity": record.levelname,
            "time": self.formatTime(record, self.datefmt),
            "message": message,
        }

        # traceback
        if record.exc_info:
            body["traceback"] = traceback.format_exc()

        return json.dumps(body, ensure_ascii=False)


handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JsonFormatter(datefmt="%Y-%m-%d %H:%M:%S"))


def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


def log_uncaught_exceptions(
    exception_cls: type[BaseException] | None,
    exception: BaseException | None,
    trace_back: TracebackType | None,
):
    """未処理の例外が上がった時の処理をオーバーライドする

    Args:
        exception_cls (type[BaseException] | None): 例外クラス
        exception (BaseException | None): 例外インスタンス
        trace_back (TracebackType | None): エラーのトレースバック
    """
    logger = get_logger(__name__)
    text = "".join(
        traceback.format_exception(exception_cls, value=exception, tb=trace_back)
    )
    logger.error("未キャッチ例外を検出 \n%s", text)
    if exception_cls is not None and exception is not None:
        # 通常の例外処理を実行
        sys.__excepthook__(exception_cls, exception, trace_back)
    return None


sys.excepthook = log_uncaught_exceptions
