# 동양미래대학교 웹응용소프트웨어공학과 4학년 졸업작품 AI 레포지토리입니다.

- 낙상 및 욕창 예측 AI
- 발생 질병 예측 AI

```
ai-service/
│
├── app/                        # 핵심 애플리케이션 코드
│   ├── api/                    # API 엔드포인트
│   │   ├── __init__.py
│   │   ├── routes_predict.py   # /predict 등 AI inference용
│   │   ├── routes_health.py    # 헬스체크
│   │
│   ├── core/                   # 설정 및 유틸
│   │   ├── config.py           # 환경변수, 로깅 설정
│   │   ├── logger.py
│   │   ├── exceptions.py
│   │
│   ├── models/                 # 모델 관련
│   │   ├── clinicalbert_model.py   # torch model 정의
│   │   ├── model_loader.py         # 모델 로딩/캐싱 로직
│   │
│   ├── services/               # 서비스 로직
│   │   ├── inference_service.py    # 실제 예측 수행 함수
│   │   ├── preprocessing.py        # 텍스트 전처리
│   │   ├── postprocessing.py       # 확률 스케일 조정 등
│   │
│   ├── schemas/                # Pydantic 스키마 (요청/응답 정의)
│   │   ├── request_schema.py
│   │   ├── response_schema.py
│   │
│   ├── main.py                 # FastAPI 진입점
│   └── __init__.py
│
├── tests/                      # 유닛 테스트 및 통합 테스트
│   ├── test_inference.py
│   ├── test_routes.py
│
├── data/                       # (선택) 샘플 데이터 / 모델 리소스
│   └── sample_inputs.json
│
├── scripts/                    # 학습, 모델 변환, 배포 스크립트
│   ├── train_model.py
│   ├── export_model.py
│
├── models/                     # 저장된 모델 가중치
│   └── clinicalbert_risk_model/
│       ├── pytorch_model.bin
│       ├── config.json
│       └── tokenizer/
│
├── Dockerfile
├── requirements.txt
├── .env                        # 환경 변수 (API_KEY, MODEL_PATH 등)
├── .gitignore
└── README.md
```
