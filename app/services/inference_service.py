# app/services/inference_service.py
import os, json, logging, traceback
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from torch import nn

from app.core.config import settings
from app.schemas.request_schema import PredictIn
from app.schemas.response_schema import PredictOut

log = logging.getLogger(__name__)

BASE_MODEL_ID = "emilyalsentzer/Bio_ClinicalBERT"

class ClinicalBERT_Regressor(nn.Module):
    def __init__(self, model_id_or_path, num_labels=2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_id_or_path)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(0.3)
        self.regressor = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls = outputs.last_hidden_state[:, 0]
        x = self.dropout(cls)
        logits = self.regressor(x)
        return logits

_tokenizer = None
_model = None
_device = torch.device(settings.DEVICE)
_has_ft_weights = False  # 파인튜닝 가중치 존재 여부

def _ensure_loaded():
    """토크나이저/모델 준비. 가중치 없으면 휴리스틱 모드로 동작."""
    global _tokenizer, _model, _has_ft_weights
    if _tokenizer is not None and (_model is not None or not _has_ft_weights):
        return

    model_dir = str(settings.MODEL_DIR)
    log.info(f"Loading model & tokenizer from: {model_dir}")

    # --- 토크나이저: 로컬에 없으면 허브에서 로드 ---
    vocab_path = os.path.join(model_dir, "vocab.txt")
    try:
        if os.path.exists(vocab_path):
            _tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
            log.info("Tokenizer loaded from local dir.")
        else:
            _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
            log.warning("Tokenizer loaded from HF hub (local tokenizer files not found).")
    except Exception:
        log.error("Tokenizer load failed:\n" + traceback.format_exc())
        raise

    # --- 파인튜닝 가중치 확인 ---
    weight_path = os.path.join(model_dir, "pytorch_model.bin")
    _has_ft_weights = os.path.exists(weight_path)

    if _has_ft_weights:
        try:
            # 베이스 구조 로드 후 state_dict 적용
            _model = ClinicalBERT_Regressor(BASE_MODEL_ID)
            state = torch.load(weight_path, map_location=_device)
            missing, unexpected = _model.load_state_dict(state, strict=False)
            if missing or unexpected:
                log.warning(f"load_state_dict: missing={missing}, unexpected={unexpected}")
            _model.to(_device)
            _model.eval()
            log.info("Model ready with fine-tuned weights.")
        except Exception:
            log.error("Model load failed:\n" + traceback.format_exc())
            raise
    else:
        # 가중치 없으면 모델을 로드하지 않고 휴리스틱 경로만 사용
        _model = None
        log.warning("No fine-tuned weights found. Fallback to heuristic scoring.")

def _to_text(v) -> str:
    if isinstance(v, str): return v.strip()
    if v is None: return ""
    if isinstance(v, (int, float, bool)): return str(v)
    if isinstance(v, (list, tuple)):
        return " ".join(map(lambda x: str(x).strip(), v)).strip()
    try:
        return json.dumps(v, ensure_ascii=False)
    except Exception:
        return str(v)

def _scale_to_percent(arr):
    if settings.USE_SIGMOID_SCALE:
        arr = 100.0 * (1.0 / (1.0 + np.exp(-arr)))
    return arr

# ---------- 규칙 기반(휴리스틱) 폴백 ----------
def _heuristic_scores_kor(text: str) -> tuple[float, float]:
    """간단 한국어 휴리스틱: 키워드 기반 점수 (0~100). 임시/보수적."""
    t = text
    score_fall = 0.0
    score_bed = 0.0

    # 나이
    import re
    age = None
    m = re.search(r'(\d+)\s*세', t)
    if m:
        age = int(m.group(1))
        if age >= 80: score_fall += 25; score_bed += 15
        elif age >= 70: score_fall += 18; score_bed += 10
        elif age >= 60: score_fall += 10; score_bed += 5

    # 보행/이동
    if any(k in t for k in ["보행 보조", "워커", "보행 불안정", "부축", "휠체어", "침상안정"]):
        score_fall += 25
        score_bed += 10
    if "자가보행 가능" in t or "독립 보행" in t:
        score_fall += 0

    # 인지/의식
    if any(k in t for k in ["치매", "섬망", "의식저하"]):
        score_fall += 15

    # 과거력
    if any(k in t for k in ["낙상 1회", "낙상 한 번", "최근 낙상"]):
        score_fall += 15
    if any(k in t for k in ["욕창 1회", "욕창 과거력", "욕창 병력"]):
        score_bed += 20

    # 만성질환 / 저영양 / 당뇨 / 순환
    if any(k in t for k in ["당뇨", "고혈압", "뇌졸중", "파킨슨", "심부전"]):
        score_fall += 7
    if any(k in t for k in ["저영양", "체중감소", "침상", "장기부동"]):
        score_bed += 15

    # 피부/압박 위험
    if any(k in t for k in ["피부손상", "발적", "압박", "욕창 의심"]):
        score_bed += 20

    # 범위 보정
    score_fall = max(0.0, min(100.0, score_fall))
    score_bed  = max(0.0, min(100.0, score_bed))
    return score_fall, score_bed

# ---------- 공개 API ----------
async def infer_once(req: PredictIn) -> PredictOut:
    _ensure_loaded()

    text = _to_text(req.text)
    if not text:
        raise ValueError("text is empty")

    if _has_ft_weights and _model is not None:
        # 파인튜닝된 모델 경로
        enc = _tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=settings.MAX_LEN)
        tti = enc.get("token_type_ids")
        with torch.no_grad():
            logits = _model(
                input_ids=enc["input_ids"].to(_device),
                attention_mask=enc["attention_mask"].to(_device),
                token_type_ids=tti.to(_device) if tti is not None else None,
            )
            preds = logits.detach().cpu().numpy()[0]
        preds = _scale_to_percent(preds)
        fall = float(np.round(preds[0], 2))
        bed  = float(np.round(preds[1], 2))
        return PredictOut(fall_risk_percent=fall, bedsore_risk_percent=bed)
    else:
        # 휴리스틱 경로
        fall, bed = _heuristic_scores_kor(text)
        return PredictOut(fall_risk_percent=fall, bedsore_risk_percent=bed)

# (옵션) 로우 텍스트로 바로 추론하는 관대 엔드포인트용
def predict_from_text(raw_text: object) -> dict:
    _ensure_loaded()
    text = _to_text(raw_text)
    if not text:
        raise ValueError("text is empty")

    if _has_ft_weights and _model is not None:
        enc = _tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=settings.MAX_LEN)
        tti = enc.get("token_type_ids")
        with torch.no_grad():
            logits = _model(
                input_ids=enc["input_ids"].to(_device),
                attention_mask=enc["attention_mask"].to(_device),
                token_type_ids=tti.to(_device) if tti is not None else None,
            )
            preds = logits.detach().cpu().numpy()[0]
        preds = _scale_to_percent(preds)
        fall = float(np.round(preds[0], 2))
        bed  = float(np.round(preds[1], 2))
    else:
        fall, bed = _heuristic_scores_kor(text)

    return {"fall_risk_percent": fall, "bedsore_risk_percent": bed}
