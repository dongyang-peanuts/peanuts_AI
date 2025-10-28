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
    한국어 휴리스틱: 키워드/정규식 기반 점수 (0~100).
    낙상/욕창 이력 N회, 거동 상태, 만성질환, BMI 등을 반영.
    """
    t = text
    score_fall = 5.0   # 약간의 베이스
    score_bed  = 5.0

    import re
    # 나이
    age = None
    m = re.search(r"(\d+)\s*세", t)
    if m:
        try:
            age = int(m.group(1))
        except Exception:
            age = None
    if age is not None:
        if age >= 85:
            score_fall += 28; score_bed += 18
        elif age >= 80:
            score_fall += 25; score_bed += 15
        elif age >= 70:
            score_fall += 18; score_bed += 10
        elif age >= 60:
            score_fall += 10; score_bed += 6
        # 젊다고 0으로 깎지는 않음(낙상 이력이 큰 경우가 있어서)

    # BMI
    bmi = None
    m_bmi = re.search(r"BMI\s*([0-9]+(?:\.[0-9]+)?)", t)
    if m_bmi:
        try:
            bmi = float(m_bmi.group(1))
        except Exception:
            bmi = None
    if bmi is not None:
        if bmi < 16:
            score_bed += 20
        elif bmi < 18.5:
            score_bed += 10

    # 거동/이동 — 세분화
    mobility_strong = ["침상안정", "휠체어", "완전 부동", "전신 부동"]
    mobility_mid    = ["보행 보조", "워커", "보행 불안정", "부축", "지팡이", "보행 보조기"]
    if any(k in t for k in mobility_strong):
        score_fall += 28; score_bed += 35
    elif any(k in t for k in mobility_mid):
        score_fall += 22; score_bed += 14
    if "자가보행 가능" in t or "독립 보행" in t:
        score_fall += 0

    # 인지/의식
    if any(k in t for k in ["치매", "섬망", "의식저하"]):
        score_fall += 15

    # 낙상/욕창 이력 N회 패턴
    fall_count = 0
    bed_count  = 0
    m_fall = re.search(r"낙상\s*(?:이력\s*)?(\d+)\s*회", t)
    if m_fall:
        fall_count = int(m_fall.group(1))
    m_bed = re.search(r"욕창\s*(?:이력\s*)?(\d+)\s*회", t)
    if m_bed:
        bed_count = int(m_bed.group(1))

    # 낙상 이력 가중치 (강화)
    if fall_count >= 1:
        # 1회: +20, 2~3회: +30, 4~6회: +40, 7회 이상: +55, 추가로 회당 +2(최대 +20)
        if fall_count == 1:
            score_fall += 20
        elif fall_count <= 3:
            score_fall += 30
        elif fall_count <= 6:
            score_fall += 40
        else:
            score_fall += 55
        score_fall += min(fall_count * 2, 20)
    # 욕창 이력 가중치
    if bed_count >= 1:
        base = 20 if bed_count == 1 else 26
        score_bed += base + min((bed_count - 1) * 3, 15)

    # 만성질환/동반질환 키워드
    if "당뇨" in t:
        score_fall += 4   # 저혈당/어지럼 연관
        score_bed  += 5   # 상처 치유 지연/피부 위험
    if "파킨슨" in t or "뇌졸중" in t or "중풍" in t:
        score_fall += 12
    if "심부전" in t or "부정맥" in t:
        score_fall += 6
    if "고혈압" in t:
        score_fall += 3   # 직접보단 간접 영향

    # 피부/압박 위험 키워드
    if any(k in t for k in ["피부손상", "발적", "압박", "욕창 의심", "궤양"]):
        score_bed += 18

    # 범위 보정
    score_fall = max(0.0, min(100.0, score_fall))
    score_bed  = max(0.0, min(100.0, score_bed))
    return score_fall, score_bed


# =========================
#   휴리스틱: 질병 Top5
# =========================
def _disease_ranking_kor(text: str, age_hint: Optional[int] = None) -> List[Tuple[str, float]]:
    """
    입력 텍스트에서 키워드/힌트를 기반으로 질병 후보에 가중치를 부여하고
    확률(%)로 정규화해 Top5를 반환.
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


    # 키워드 매칭으로 점수 누적
    scores = {}
    for name, (base, keywords) in CANDIDATES.items():
        s = float(base)
        for k in keywords:
            if k in t:
                s += 2.5  # 매칭 가중치
        # 추가 규칙 (나이 보정)
        if age_hint is not None and age_hint >= 80 and name in [
            "폐렴", "섬망/의식변화", "욕창/피부손상", "낙상 후 손상", "탈수/전해질 이상"
        ]:
            s += 2.0
        # 동반질환 보정
        if "당뇨" in t and name == "저혈당/고혈당":
            s += 3.0
        if "고혈압" in t and name == "뇌졸중(뇌혈관 사건)":
            s += 2.0
        if any(k in t for k in ["지팡이", "워커", "보행 보조", "보행 불안정", "부축", "휠체어"]):
            if name in ["낙상 후 손상", "욕창/피부손상"]:
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

# =========================
#   Spring 연동 & 텍스트 생성
# =========================
import requests

def fetch_user_from_spring(user_key: int) -> dict:
    url = f"http://kongback.kro.kr:8080/admin/users/{user_key}"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.json()

def _try_float(x, default=None):
    try:
        if x is None: return default
        if isinstance(x, (int, float)): return float(x)
        s = str(x).strip()
        if s == "": return default
        return float(s)
    except Exception:
        return default

def user_data_to_text(user_json: dict, patient_index: int = 0) -> str:
    """
    - infos[*].paFact/paPrct 합산 → "낙상 N회, 욕창 M회"
    - paDi는 '낙상/욕창' 제외하고 만성질환만 나열(예: 고혈압, 당뇨)
    - 키/몸무게가 있으면 BMI 계산하여 추가 ("BMI 23.4")
    - 거동 상태 정규화 단어가 텍스트에 포함되도록 유지(지팡이/워커/휠체어/침상안정 등)
    """
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

    infos = get(p, "infos", [])
    mobility = ""
    meds     = ""
    diseases_raw = []
    if infos:
        # 대표 1개에서 mobility/meds
        main = infos[0]
        mobility = get(main, "paBest", "")
        meds     = get(main, "paMedi", "")
        # 질병들 모으기(모든 infos)
        for info in infos:
            di = get(info, "paDi", "")
            if not di: continue
            # 여러 개가 콤마/슬래시로 묶여 있을 수 있음
            for token in str(di).replace("/", ",").split(","):
                name = token.strip()
                if not name: continue
                # 낙상/욕창은 질병 목록에 넣지 않음(이력으로 처리)
                if any(x in name for x in ["낙상", "욕창"]):
                    continue
                diseases_raw.append(name)

    # 낙상/욕창 이력 합산
    fall_hist_total = 0
    bedsore_hist_total = 0
    for info in infos:
        f = get(info, "paFact", 0)
        b = get(info, "paPrct", 0)
        try:
            fall_hist_total += int(str(f).strip().split()[0])
        except Exception:
            pass
        try:
            bedsore_hist_total += int(str(b).strip().split()[0])
        except Exception:
            pass

    # BMI 계산
    h = _try_float(paHei, None)
    w = _try_float(paWei, None)
    bmi_str = ""
    if h and w and h > 0:
        bmi = w / ((h/100.0) ** 2)
        bmi_str = f"BMI {round(bmi, 1)}"

    parts = []
    if paAge != "": parts.append(f"{paAge}세 환자")
    if diseases_raw:
        parts.append(" / ".join(diseases_raw))
    if mobility: parts.append(mobility)
    if meds: parts.append(f"복용 약: {meds}")
    if paHei: parts.append(f"키 {paHei}cm")
    if paWei: parts.append(f"체중 {paWei}kg")
    if bmi_str: parts.append(bmi_str)

    if fall_hist_total > 0: parts.append(f"낙상 {fall_hist_total}회")
    if bedsore_hist_total > 0: parts.append(f"욕창 {bedsore_hist_total}회")

    return ", ".join(parts).strip()
