# app/core/config.py
from dataclasses import dataclass
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]

@dataclass
class Settings:
    MODEL_DIR: Path = ROOT / "models" / "clinicalbert_risk_model"
    MAX_LEN: int = 128
    USE_SIGMOID_SCALE: bool = True
    DEVICE: str = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

settings = Settings()
