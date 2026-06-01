# WM-811K Wafer Map Defect Pattern Classification

## 프로젝트 개요

WM-811K 데이터셋을 활용하여 반도체 웨이퍼 맵의 불량 패턴을 분류하는 딥러닝 모델을 구현합니다.
기존 연구를 보완하여 **Transfer Learning + 최소한의 데이터 증강** 으로
적은 계산 비용으로 높은 분류 성능을 달성하는 것을 목표로 합니다.

---

## 연구 동기

기존 연구:

- **A논문 (경북대, 2024)**: Transfer Learning만으로 94.8% 달성했으나 None(정상) 클래스를 제외한 실험으로 실제 현장과 조건이 다름
- **B논문 (인하대, 2025)**: 회전 증강 50%로 GM+ 0.9195 달성했으나 Transfer Learning 미적용

→ 두 논문에서 시도하지 않은 **"None 포함 + Transfer Learning + 최소 증강"** 조합을 실험합니다.

---

## 연구 목표

> **"Transfer Learning을 적용하면 B논문의 Scratch + 50% 증강과 동등한 성능을
> 더 적은 증강 비율(25% 이하)로 달성할 수 있다"**

---

## 데이터셋

- **WM-811K** (Kaggle): https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map
- 전체 811,457개 웨이퍼 맵 중 라벨링된 172,950개 사용
- 9개 클래스: None, Center, Donut, Edge-Loc, Edge-Ring, Loc, Near-Full, Random, Scratch

## 컬럼
- waferMap (2D 배열) 실제 웨이퍼 이미지 데이터 (0=빈공간, 1=정상, 2=불량)
- dieSize (숫자) 웨이퍼 크기 (가로×세로 다이 수)
- lotName (문자열) 같은 배치(lot)에서 생산된 웨이퍼 묶음 번호
- waferIndex (숫자) lot 내에서 웨이퍼 순서
- trainTestLabel (문자열) 원본 train/test 구분
- failureType (문자열) 불량 패턴 라벨

### 클래스별 데이터 분포

| 클래스 | 데이터 수 | 비율 |
|--------|----------|------|
| None | 147,431 | 85.25% |
| Edge-Ring | 9,682 | 5.60% |
| Edge-Loc | 5,199 | 3.00% |
| Center | 4,296 | 2.48% |
| Loc | 3,597 | 2.08% |
| Scratch | 1,194 | 0.69% |
| Random | 866 | 0.50% |
| Donut | 555 | 0.32% |
| Near-Full | 149 | 0.09% |

---

## 전처리

- 이미지 크기: **64×64** 로 리사이즈
- 데이터 분할: **Train : Validation : Test = 6 : 2 : 2**
- 분할 후 Train 데이터에만 증강 적용 (data leakage 방지)

---

## 데이터 증강
B논문에서의 방식을 가이드라인으로

- 방식: **회전(Rotation)** 기반 오버샘플링
- 범위: 1° ~ 359° (원본과 중복 방지)
- 기준: None 클래스 수를 기준으로 부족한 클래스를 증강
- 이유: 웨이퍼는 원형 구조라 회전해도 패턴 구조 보존됨

---

## 실험 설계

베이스라인은 B논문 결과를 인용합니다.

| 실험 | 증강 비율 | Transfer Learning | 비고 |
|------|----------|-------------------|------|
| B논문 (베이스라인) | 50% | ❌ Scratch | 인용 |
| **C-0** | 0% | ✅ ImageNet pretrained | TL 단독 효과 확인 |
| **C-25** | 25% | ✅ ImageNet pretrained | **핵심 실험** |
| **C-50** | 50% | ✅ ImageNet pretrained | B논문과 동일 증강, TL 추가 |

- 모델: **ResNet50** 
- 프레임워크: **PyTorch**
- 환경: AWS EC2

---

## 평가 지표

| 지표 | 설명 |
|------|------|
| **Accuracy** | 전체 정확도 |
| **F1-score** | Precision과 Recall의 조화 평균 |
| **GM** | 클래스별 Recall의 기하 평균 |
| **GM+** | 클래스 비율 가중치를 반영한 GM (메인 지표) |

> GM+를 메인 지표로 사용하는 이유: None 클래스가 85%를 차지하는 불균형 데이터에서
> Accuracy만으로는 실제 분류 성능을 정확히 반영하지 못하기 때문

---

## 프로젝트 구조

wafer-defect/
├── data/
│   └── LSWMD.pkl
├── src/
│   ├── dataset.py       # 데이터 로드 및 전처리
│   ├── augmentation.py  # 회전 기반 데이터 증강
│   ├── model.py         # ResNet50 모델 정의
│   ├── train.py         # 학습 코드
│   └── evaluate.py      # 평가 코드
├── logs/                # 학습 로그
├── checkpoints/         # 모델 가중치
├── requirements.txt
└── README.md


---

## 환경 설정

```bash
pip install -r requirements.txt
```
torch
torchvision
numpy
pandas
matplotlib
scikit-learn
tqdm
kaggle

---

## 실행 방법

```bash
# 데이터 다운로드
kaggle datasets download -d qingyi/wm811k-wafer-map
unzip wm811k-wafer-map.zip -d ./data/

# 학습 실행
python src/train.py --augment_ratio 0.25 --transfer_learning True

# 평가
python src/evaluate.py --checkpoint checkpoints/best_model.pth
```

---

## 참고 논문

- 박인영, 김지영, "데이터 클래스 불균형을 고려한 전이학습 기반의 반도체 웨이퍼 빈 맵 결함 패턴 분류", 한국산학기술학회논문지, Vol.25, No.5, 2024.
- 박상현, 김지성, 남춘성, "WM-811K 데이터셋의 클래스 불균형 방안에 관한 연구", 멀티미디어학회 논문지, Vol.28, No.10, 2025.

---

## TODO

- [ ] 데이터 전처리 코드 작성
- [ ] 증강 코드 작성
- [ ] C-0 실험
- [ ] C-25 실험
- [ ] C-50 실험
- [ ] 결과 분석 및 보고서 작성
