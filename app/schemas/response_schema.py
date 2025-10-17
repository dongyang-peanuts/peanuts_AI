# app/schemas/response_schema.py
from pydantic import BaseModel, validator

class PredictOut(BaseModel):
    fall_risk_percent: float
    bedsore_risk_percent: float

    @validator("fall_risk_percent", "bedsore_risk_percent", pre=True, always=True)
    def clamp_pct(cls, v):
        try:
            v = float(v)
        except Exception:
            v = 0.0
        return max(0.0, min(100.0, v))
