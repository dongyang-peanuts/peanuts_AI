import os
import json
import logging
import traceback
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel

from app.core.config import settings
from app.schemas.request_schema import PredictIn
from app.schemas.response_schema import PredictOut

log = logging.getLogger(__name__)

BASE_MODEL_ID = "emilyalsentzer/Bio_ClinicalBERT"


# =========================
#   모델 정의
# =========================
class ClinicalBERT_Regressor(nn.Module):
    def __init__(self, model_id_or_path: str, num_labels: int = 2):
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


# =========================
#   전역 상태
# =========================
_tokenizer = None
_model = None
_device = torch.device(settings.DEVICE)
_has_ft_weights = False


# =========================
#   로딩
# =========================
def _ensure_loaded():
    global _tokenizer, _model, _has_ft_weights
    if _tokenizer is not None and (_model is not None or not _has_ft_weights):
        return

    model_dir = str(settings.MODEL_DIR)
    log.info(f"Loading model & tokenizer from: {model_dir}")

    vocab_path = os.path.join(model_dir, "vocab.txt")
    try:
        if os.path.exists(vocab_path):
            _tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        else:
            _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    except Exception:
        log.error("Tokenizer load failed:\n" + traceback.format_exc())
        raise

    weight_path = os.path.join(model_dir, "pytorch_model.bin")
    _has_ft_weights = os.path.exists(weight_path)

    if _has_ft_weights:
        try:
            _model = ClinicalBERT_Regressor(BASE_MODEL_ID)
            state = torch.load(weight_path, map_location=_device)
            missing, unexpected = _model.load_state_dict(state, strict=False)
            if missing or unexpected:
                log.warning(f"load_state_dict: missing={missing}, unexpected={unexpected}")
            _model.to(_device)
            _model.eval()
        except Exception:
            log.error("Model load failed:\n" + traceback.format_exc())
            raise
    else:
        _model = None
        log.warning("No fine-tuned weights found. Using heuristic mode.")


def _to_text(v) -> str:
    if isinstance(v, str):
        return v.strip()
    if v is None:
        return ""
    if isinstance(v, (int, float, bool)):
        return str(v)
    if isinstance(v, (list, tuple)):
        return " ".join(map(lambda x: str(x).strip(), v)).strip()
    try:
        return json.dumps(v, ensure_ascii=False)
    except Exception:
        return str(v)


def _scale_to_percent(arr: np.ndarray) -> np.ndarray:
    if settings.USE_SIGMOID_SCALE:
        arr = 100.0 * (1.0 / (1.0 + np.exp(-arr)))
    return arr


# =========================
#   휴리스틱: 낙상/욕창
# =========================
def _heuristic_scores_kor(text: str) -> Tuple[float, float]:
    t = text
    score_fall, score_bed = 5.0, 5.0

    import re
    age = None
    m = re.search(r"(\d+)\s*세", t)
    if m:
        try:
            age = int(m.group(1))
        except Exception:
            pass

    if age is not None:
        if age >= 85:
            score_fall += 28; score_bed += 18
        elif age >= 80:
            score_fall += 25; score_bed += 15
        elif age >= 70:
            score_fall += 18; score_bed += 10
        elif age >= 60:
            score_fall += 10; score_bed += 6

    m_bmi = re.search(r"BMI\s*([0-9]+(?:\.[0-9]+)?)", t)
    if m_bmi:
        try:
            bmi = float(m_bmi.group(1))
            if bmi < 16:
                score_bed += 20
            elif bmi < 18.5:
                score_bed += 10
        except Exception:
            pass

    mobility_strong = ["침상안정", "휠체어", "완전 부동", "전신 부동"]
    mobility_mid = ["보행 보조", "워커", "보행 불안정", "부축", "지팡이", "보행 보조기"]
    if any(k in t for k in mobility_strong):
        score_fall += 28; score_bed += 35
    elif any(k in t for k in mobility_mid):
        score_fall += 22; score_bed += 14

    if any(k in t for k in ["치매", "섬망", "의식저하"]):
        score_fall += 15

    fall_count = 0
    bed_count = 0
    m_fall = re.search(r"낙상\s*(?:이력\s*)?(\d+)\s*회", t)
    if m_fall: fall_count = int(m_fall.group(1))
    m_bed = re.search(r"욕창\s*(?:이력\s*)?(\d+)\s*회", t)
    if m_bed: bed_count = int(m_bed.group(1))

    if fall_count >= 1:
        if fall_count == 1: score_fall += 20
        elif fall_count <= 3: score_fall += 30
        elif fall_count <= 6: score_fall += 40
        else: score_fall += 55
        score_fall += min(fall_count * 2, 20)

    if bed_count >= 1:
        base = 20 if bed_count == 1 else 26
        score_bed += base + min((bed_count - 1) * 3, 15)

    if "당뇨" in t:
        score_fall += 4; score_bed += 5
    if "파킨슨" in t or "뇌졸중" in t or "중풍" in t:
        score_fall += 12
    if "심부전" in t or "부정맥" in t:
        score_fall += 6
    if "고혈압" in t:
        score_fall += 3
    if any(k in t for k in ["피부손상", "발적", "압박", "욕창 의심", "궤양"]):
        score_bed += 18

    return min(score_fall, 100.0), min(score_bed, 100.0)


# =========================
#   휴리스틱: 질병 Top5
# =========================
def _disease_ranking_kor(text: str, age_hint: Optional[int] = None) -> List[Tuple[str, float]]:
    t = text
    import re
    if age_hint is None:
        m = re.search(r"(\d+)\s*세", t)
        if m:
            try:
                age_hint = int(m.group(1))
            except Exception:
                age_hint = None

    CANDIDATES = {
        "고혈압": [3.0, ["혈압", "고혈압", "두통", "어지럼"]],
        "당뇨병": [3.0, ["당뇨", "혈당", "고혈당", "저혈당", "다뇨", "다음", "피로"]],
        "치매": [3.0, ["기억력 저하", "인지 저하", "섬망", "혼돈", "방향감 상실"]],
        "골다공증": [3.0, ["골다공증", "골절", "낙상", "허리 통증"]],
        "뇌졸중(중풍)": [3.0, ["편측마비", "언어장애", "구음장애", "시야장애", "갑작스러운 마비"]],
        "심부전": [3.0, ["호흡곤란", "부종", "야간호흡곤란", "체중 증가", "피로"]],
        "관상동맥질환(협심증/심근경색)": [3.0, ["가슴통증", "흉통", "압박감", "호흡곤란"]],
        "만성폐쇄성폐질환(COPD)": [3.0, ["기침", "가래", "호흡곤란", "천명음"]],
        "폐렴": [3.0, ["기침", "가래", "열", "호흡곤란", "산소포화도 저하"]],
        "요로감염": [3.0, ["배뇨통", "빈뇨", "절박뇨", "탁뇨", "소변 냄새", "열"]],
        "신부전(만성콩팥병)": [3.0, ["신부전", "부종", "소변 감소", "혈크레아티닌 상승"]],
        "파킨슨병": [3.0, ["손 떨림", "경직", "느린 움직임", "보행 불안정"]],
        "우울증": [3.0, ["우울", "의욕 저하", "불면", "식욕 저하", "피로"]],
        "불안장애": [3.0, ["불안", "긴장", "불면", "가슴 답답"]],
        "빈혈": [3.0, ["피로", "창백", "어지럼", "호흡곤란"]],
        "변비": [3.0, ["배변", "변비", "딱딱한 변", "배불름"]],
        "탈수": [3.0, ["탈수", "구갈", "피부 건조", "저혈압"]],
        "욕창": [3.0, ["피부손상", "발적", "압박", "궤양", "욕창"]],
        "감염성 질환(비특이적)": [3.0, ["발열", "오한", "CRP 상승", "백혈구 증가"]],
    }

    scores = {}
    keyword_hit = False
    used_keywords = set()

    for name, (base, keywords) in CANDIDATES.items():
        s = float(base)
        for k in keywords:
            if k in t and k not in used_keywords:
                s += 2.5
                keyword_hit = True
                used_keywords.add(k)
        scores[name] = max(0.0, s)

    # ✅ 키워드가 하나도 없을 경우 → 균등 확률 분포
    if not keyword_hit:
        total_diseases = len(CANDIDATES)
        equal_prob = round(100.0 / total_diseases, 2)
        return [(k, equal_prob) for k in list(CANDIDATES.keys())[:5]]

    total = sum(scores.values()) or 1.0
    probs = {k: (v / total * 100.0) for k, v in scores.items()}

    ordered = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
    ordered = [(k, float(round(v, 2))) for k, v in ordered]
    ssum = sum(v for _, v in ordered)
    if ssum > 0:
        k0, v0 = ordered[0]
        ordered[0] = (k0, float(round(v0 + (100.0 - ssum), 2)))
    return ordered


# =========================
#   공개 API
# =========================
async def infer_once(req: PredictIn) -> PredictOut:
    _ensure_loaded()
    text = _to_text(req.text)
    if not text:
        raise ValueError("text is empty")

    diseases = _disease_ranking_kor(text)
    diseases_out = [{"name": n, "probability_percent": p} for n, p in diseases]

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
        fall, bed = float(np.round(preds[0], 2)), float(np.round(preds[1], 2))
    else:
        fall, bed = _heuristic_scores_kor(text)

    return PredictOut(fall_risk_percent=fall, bedsore_risk_percent=bed, diseases=diseases_out)


def predict_from_text(raw_text: object) -> dict:
    _ensure_loaded()
    text = _to_text(raw_text)
    if not text:
        raise ValueError("text is empty")

    diseases = _disease_ranking_kor(text)
    diseases_out = [{"name": n, "probability_percent": p} for n, p in diseases]

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
        fall, bed = float(np.round(preds[0], 2)), float(np.round(preds[1], 2))
    else:
        fall, bed = _heuristic_scores_kor(text)

    return {"fall_risk_percent": fall, "bedsore_risk_percent": bed, "diseases": diseases_out}
