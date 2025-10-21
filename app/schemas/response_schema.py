from pydantic import BaseModel, validator
from typing import List

class DiseaseOut(BaseModel):
    name: str
    probability_percent: float

    @validator("probability_percent", pre=True, always=True)
    def clamp_pct(cls, v):
        try:
            v = float(v)
        except Exception:
            v = 0.0
        return max(0.0, min(100.0, v))

class PredictOut(BaseModel):
    fall_risk_percent: float
    bedsore_risk_percent: float
    diseases: List[DiseaseOut]  # ← Top5 질병 리스트

    @validator("fall_risk_percent", "bedsore_risk_percent", pre=True, always=True)
    def clamp_pct(cls, v):
        try:
            v = float(v)
        except Exception:
            v = 0.0
        return max(0.0, min(100.0, v))
