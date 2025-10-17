# app/schemas/request_schema.py
from pydantic import BaseModel, Field, field_validator
from typing import Any

class PredictIn(BaseModel):
    text: Any = Field(..., description="환자 상태 서술 텍스트")

    @field_validator("text")
    @classmethod
    def to_str(cls, v):
        if v is None:
            raise ValueError("text is required")
        # list/dict/숫자 등도 문자열로
        if isinstance(v, str):
            s = v.strip()
        elif isinstance(v, (int, float, bool)):
            s = str(v)
        elif isinstance(v, (list, tuple)):
            s = " ".join(map(lambda x: str(x).strip(), v))
        else:
            # dict/기타 → JSON/문자열화
            try:
                import json
                s = json.dumps(v, ensure_ascii=False)
            except Exception:
                s = str(v)
        if not s:
            raise ValueError("text must be non-empty string")
        return s
