from fastapi import APIRouter, HTTPException, Request
from app.services.inference_service import infer_once, predict_from_text
from app.schemas.request_schema import PredictIn
from app.schemas.response_schema import PredictOut

router = APIRouter()

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

# ✅ 관대 모드: Pydantic 검사 우회, 원시 바디에서 text 뽑아 바로 추론
@router.post("/infer_raw")
async def infer_raw(req: Request):
    try:
        # JSON이면 그대로, 아니면 원시 바이트
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
