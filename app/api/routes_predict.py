# app/api/routes_predict.py
from fastapi import APIRouter, HTTPException, Request, Query
from app.schemas.request_schema import PredictIn
from app.schemas.response_schema import PredictOut
from app.services.inference_service import (
    infer_once,
    predict_from_text,
    fetch_user_from_spring,   # Spring GET으로 유저 데이터 가져오기
    user_data_to_text,        # 유저 데이터 -> 모델 입력 텍스트 생성
)

router = APIRouter()

# 1) 텍스트를 직접 받아서 추론 (표준)
@router.post("/infer", response_model=PredictOut)
async def infer(req: PredictIn):
    try:
        return await infer_once(req)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Bad input: {e}")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Inference error: {type(e).__name__}: {e}")

# 2) 관대 모드: 스키마 검사 우회, 원시 바디에서 text 뽑아 추론
@router.post("/infer_raw")
async def infer_raw(req: Request):
    try:
        try:
            body = await req.json()
        except Exception:
            body = {"text": (await req.body()).decode("utf-8", errors="ignore")}
        raw_text = body.get("text", body)
        out = predict_from_text(raw_text)
        return out
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Bad input: {e}")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Inference error: {type(e).__name__}: {e}")

# 3) GET: Spring에서 환자 데이터 가져와 -> 텍스트 생성 -> AI 추론 -> 결과(PredictOut)만 반환
@router.get("/analyze/{user_key}", response_model=dict)
async def analyze_by_user(
    user_key: int,
    patient_index: int = Query(0, description="patients 배열 인덱스(기본 0)")
):
    """
    /api/risk/analyze/{user_key}
    예: /api/risk/analyze/14
    """
    try:
        user_json = fetch_user_from_spring(user_key)
        spring_raw = user_json
        text = user_data_to_text(user_json, patient_index=patient_index)

        if not text:
            raise HTTPException(status_code=400, detail="No patient data to analyze")
        
        result = await infer_once(PredictIn(text=text))
        return {
            "ai_result": {
                "fall_risk_percent": result.fall_risk_percent,
                "bedsore_risk_percent": result.bedsore_risk_percent,
                "diseases": result.diseases
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Analyze failed: {e}")

