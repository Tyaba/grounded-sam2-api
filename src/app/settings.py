from __future__ import annotations

import os
from functools import lru_cache

from pydantic import BaseSettings


class APIBaseSettings(BaseSettings):
    env: str

    # Cloud Storage
    bucket_name: str
    upload_dir: str = "face_motion_generator"

    @property
    def device(self) -> str:
        return os.getenv("DEVICE", "cpu")


class LocalAPISettings(APIBaseSettings):
    env = "loc"
    bucket_name = "dews-human-dev-390"


class DevAPISettings(APIBaseSettings):
    env = "dev"
    bucket_name = "dews-human-dev-390"


class StgAPISettings(APIBaseSettings):
    env = "stg"
    bucket_name = "dews-human-stg-389"


class PrdAPISettings(APIBaseSettings):
    env = "prd"
    bucket_name = "dews-human-prd-349"


SETTINGS: list[APIBaseSettings] = [
    LocalAPISettings(),
    DevAPISettings(),
    StgAPISettings(),
    PrdAPISettings(),
]


@lru_cache()
def get_settings() -> APIBaseSettings:
    env = os.getenv("SERVICE_ENV", "loc")

    for settings in SETTINGS:
        if env == settings.env:
            return settings

    raise ValueError(f"Unsupported environment: {env}.")
