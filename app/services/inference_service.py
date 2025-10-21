# app/services/inference_service.py
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

# 원본 ClinicalBERT 식별자 (토크나이저/베이스 구조 로딩용)
BASE_MODEL_ID = "emilyalsentzer/Bio_ClinicalBERT"


# =========================
#   모델 정의 (회귀 헤드)
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
        cls = outputs.last_hidden_state[:, 0]  # [CLS]
        x = self.dropout(cls)
        logits = self.regressor(x)
        return logits


# =========================
#   전역 상태
# =========================
_tokenizer = None
_model = None
_device = torch.device(settings.DEVICE)
_has_ft_weights = False  # 파인튜닝 가중치 존재 여부


# =========================
#   로딩 & 유틸
# =========================
def _ensure_loaded():
    """
    토크나이저/모델 준비.
    - 로컬 토크나이저 없으면 HF 허브에서 가져옴
    - 파인튜닝 가중치 없으면 휴리스틱 모드로 동작
    """
    global _tokenizer, _model, _has_ft_weights
    if _tokenizer is not None and (_model is not None or not _has_ft_weights):
        return

    model_dir = str(settings.MODEL_DIR)
    log.info(f"Loading model & tokenizer from: {model_dir}")

    # --- 토크나이저 로드 (로컬 우선, 없으면 허브) ---
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
        # 베이스 구조 로드 후 state_dict 적용
        try:
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
        # 가중치 없으면 모델 로드 생략 → 휴리스틱으로만 동작
        _model = None
        log.warning("No fine-tuned weights found. Fallback to heuristic scoring.")


def _to_text(v) -> str:
    """입력 어떤 타입이 와도 문자열로 정규화."""
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
    """학습 출력 스케일 보정(옵션)."""
    if settings.USE_SIGMOID_SCALE:
        arr = 100.0 * (1.0 / (1.0 + np.exp(-arr)))
    return arr


# =========================
#   휴리스틱: 낙상/욕창
# =========================
def _heuristic_scores_kor(text: str) -> Tuple[float, float]:
    """
    간단 한국어 휴리스틱: 키워드 기반 점수 (0~100). 임시/보수적.
    """
    t = text
    score_fall = 0.0
    score_bed = 0.0

    # 나이
    import re
    age = None
    m = re.search(r"(\d+)\s*세", t)
    if m:
        try:
            age = int(m.group(1))
        except Exception:
            age = None
    if age is not None:
        if age >= 80:
            score_fall += 25
            score_bed += 15
        elif age >= 70:
            score_fall += 18
            score_bed += 10
        elif age >= 60:
            score_fall += 10
            score_bed += 5

    # 보행/이동
    if any(k in t for k in ["보행 보조", "워커", "보행 불안정", "부축", "휠체어", "침상안정"]):
        score_fall += 25
        score_bed += 10
    if "자가보행 가능" in t or "독립 보행" in t:
        score_fall += 0  # 완충

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
    score_bed = max(0.0, min(100.0, score_bed))
    return score_fall, score_bed


# =========================
#   휴리스틱: 질병 Top5
# =========================
def _disease_ranking_kor(text: str, age_hint: Optional[int] = None) -> List[Tuple[str, float]]:
    """
    입력 텍스트에서 키워드/힌트를 기반으로 질병 후보에 가중치를 부여하고
    확률(%)로 정규화해 Top5를 반환.
    추후 분류기 파인튜닝 시 이 함수를 대체하면 됨.
    """
    t = text

    # 나이 힌트 추출
    import re
    if age_hint is None:
        m = re.search(r"(\d+)\s*세", t)
        if m:
            try:
                age_hint = int(m.group(1))
            except Exception:
                age_hint = None

    # 후보 사전: {질병명: (기본가중치, [키워드들])}
    CANDIDATES = {
        "요로감염(UTI)": [4.0, ["배뇨통", "빈뇨", "절박뇨", "탁뇨", "소변 냄새", "열"]],
        "폐렴": [5.0, ["기침", "가래", "호흡곤란", "발열", "청색증", "산소포화도 저하"]],
        "섬망/의식변화": [5.0, ["섬망", "의식저하", "혼미", "급격한 인지 변화", "낮밤바뀜"]],
        "탈수/전해질 이상": [4.0, ["탈수", "구갈", "저혈압", "어지럼", "전해질 이상", "식욕부진"]],
        "저혈당/고혈당": [4.0, ["저혈당", "고혈당", "어지럼", "식은땀", "혼미"]],
        "심부전 악화": [3.5, ["호흡곤란", "부종", "야간호흡곤란", "체중증가", "부정맥"]],
        "뇌졸중(뇌혈관 사건)": [3.0, ["편측마비", "구음장애", "시야장애", "급성 발현"]],
        "치매 진행": [3.0, ["기억력 저하", "인지 저하", "일상기능 저하"]],
        "욕창/피부손상": [4.0, ["욕창", "피부손상", "발적", "압박", "궤양"]],
        "낙상 후 손상": [4.0, ["낙상", "골절", "통증", "부종", "멍"]],
        "감염 의증(비특이)": [2.5, ["발열", "오한", "CRP 상승", "백혈구 증가"]],
    }

    # 키워드 매칭으로 점수 누적
    scores = {}
    for name, (base, keywords) in CANDIDATES.items():
        s = float(base)
        for k in keywords:
            if k in t:
                s += 2.5  # 매칭 가중치 (튜닝 가능)
        # 추가 규칙 (나이 보정)
        if age_hint is not None:
            if age_hint >= 80 and name in [
                "폐렴",
                "섬망/의식변화",
                "욕창/피부손상",
                "낙상 후 손상",
                "탈수/전해질 이상",
            ]:
                s += 2.0
        # 동반질환 보정
        if "당뇨" in t and name == "저혈당/고혈당":
            s += 2.0
        if "심부전" in t and name == "심부전 악화":
            s += 2.0
        scores[name] = max(0.0, s)

    # 정규화 → 확률(%)
    total = sum(scores.values()) or 1.0
    probs = {k: (v / total * 100.0) for k, v in scores.items()}

    # Top5
    ordered = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]

    # 반올림 & 총합 보정
    ordered = [(k, float(round(v, 2))) for k, v in ordered]
    ssum = sum(v for _, v in ordered)
    if ssum > 0:
        k0, v0 = ordered[0]
        ordered[0] = (k0, float(round(v0 + (100.0 - ssum), 2)))
    return ordered


# =========================
#   공개 API 로직
# =========================
async def infer_once(req: PredictIn) -> PredictOut:
    """
    - 파인튜닝 가중치가 있으면 모델 추론
    - 없으면 휴리스틱 점수
    - 항상 질병 Top5도 함께 반환
    """
    _ensure_loaded()

    text = _to_text(req.text)
    if not text:
        raise ValueError("text is empty")

    # 질병 Top5 먼저 계산 (둘 다 공통)
    diseases = _disease_ranking_kor(text)
    diseases_out = [{"name": n, "probability_percent": p} for n, p in diseases]

    if _has_ft_weights and _model is not None:
        # 파인튜닝된 모델 경로
        enc = _tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=settings.MAX_LEN
        )
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
        bed = float(np.round(preds[1], 2))
        return PredictOut(
            fall_risk_percent=fall,
            bedsore_risk_percent=bed,
            diseases=diseases_out,
        )
    else:
        # 휴리스틱 경로
        fall, bed = _heuristic_scores_kor(text)
        return PredictOut(
            fall_risk_percent=fall,
            bedsore_risk_percent=bed,
            diseases=diseases_out,
        )


# (옵션) 관대 엔드포인트용: 원시 텍스트로 바로 추론
def predict_from_text(raw_text: object) -> dict:
    _ensure_loaded()

    text = _to_text(raw_text)
    if not text:
        raise ValueError("text is empty")

    # 질병 Top5
    diseases = _disease_ranking_kor(text)
    diseases_out = [{"name": n, "probability_percent": p} for n, p in diseases]

    if _has_ft_weights and _model is not None:
        enc = _tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=settings.MAX_LEN
        )
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
        bed = float(np.round(preds[1], 2))
    else:
        fall, bed = _heuristic_scores_kor(text)

    return {
        "fall_risk_percent": fall,
        "bedsore_risk_percent": bed,
        "diseases": diseases_out,
    }

# app/services/inference_service.py (맨 아래 근처)
import requests

def fetch_user_from_spring(user_key: int) -> dict:
    url = f"http://kongback.kro.kr:8080/admin/users/{user_key}"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.json()

def user_data_to_text(user_json: dict, patient_index: int = 0) -> str:
    def get(d, k, default=""):
        v = d.get(k, default) if isinstance(d, dict) else default
        return v if v is not None else default

    patients = get(user_json, "patients", [])
    if not patients:
        return ""

    idx = max(0, min(patient_index, len(patients)-1))
    p = patients[idx]

    paAge  = get(p, "paAge", "")
    paHei  = get(p, "paHei", "")
    paWei  = get(p, "paWei", "")
    mobility = ""
    disease  = ""
    severity = ""
    meds     = ""
    exti     = ""

    infos = get(p, "infos", [])
    if infos:
        main = infos[0]
        mobility = get(main, "paBest", "")
        disease  = get(main, "paDi", "")
        severity = get(main, "paDise", "")
        meds     = get(main, "paMedi", "")
        exti     = get(main, "paExti", "")

    parts = []
    if paAge != "": parts.append(f"{paAge}세 환자")
    if disease:
        parts.append(f"{disease}" + (f"({severity})" if severity else ""))
    if mobility: parts.append(mobility)
    if meds: parts.append(f"복용 약: {meds}")
    if paHei: parts.append(f"키 {paHei}cm")
    if paWei: parts.append(f"체중 {paWei}kg")
    if exti: parts.append(f"운동/활동: {exti}")

    # 간단 과거력 힌트(키워드 기반)
    base = " ".join(parts)
    if "낙상" in base: parts.append("낙상 병력")
    if "욕창" in base: parts.append("욕창 병력")

    return ", ".join(parts).strip()

