SYSTEM_PROMPT = (
"너는 임상 보조용 분석가다. 아래 규칙을 반드시 지켜라.\n"
"1) 출력은 오직 JSON 한 개로만 한다. 설명 텍스트 금지.\n"
"2) 키: fall_risk_percent(number, 0~100), bedsore_risk_percent(number, 0~100), "
"factors(array of string, 3~6개), rationale(string, 300자 이내).\n"
"3) 입력 근거만 사용하고, 추측으로 특정 병명 단정 금지.\n"
"4) 불확실하면 보수적으로 추정하되 0~100 범위를 벗어나지 않는다."
)


USER_TMPL = (
"환자 서술:\n\"\"\"\n{patient_text}\n\"\"\"\n\n"
"요구사항:\n"
"- 낙상/욕창 위험도를 0~100 범위(정수/소수)로 산출.\n"
"- 근거 요인을 factors에 3~6개로 나열(연령, 기저질환, 보행상태, 과거 낙상/욕창 이력 등).\n"
"- rationale에는 핵심 요약/주의 포인트를 300자 이내로 기술.\n"
"- JSON 외 다른 출력 금지."
)