from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    sam2_checkpoint: str = "./checkpoints/sam2.1_hiera_large.pt"
    sam2_model_cfg: str = "sam2.1_hiera_l.yaml"
    gdino_model_id: str = "IDEA-Research/grounding-dino-tiny"
    device: str = "cuda"
