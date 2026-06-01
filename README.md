# WM811k

# WM-811K Wafer Map Defect Pattern Classification

## 프로젝트 개요

WM-811K 데이터셋을 활용하여 반도체 웨이퍼 맵의 불량 패턴을 분류하는 딥러닝 모델을 구현합니다.
기존 연구의 한계를 보완하여 **Transfer Learning + 최소 데이터 증강** 조합으로
적은 계산 비용으로 높은 분류 성능을 달성하는 것을 목표로 합니다.

---

## 연구 동기

기존 연구의 한계:

- **A논문 (경북대, 2024)**: Transfer Learning만으로 94.8% 달성했으나 None(정상) 클래스를 제외한 실험으로 실제 현장과 조건이 다름
- **B논문 (인하대, 2025)**: 회전 증강 50%로 GM+ 0.9195 달성했으나 Transfer Learning 미적용

→ 두 논문 모두 시도하지 않은 **"None 포함 + Transfer Learning + 최소 증강"** 조합을 실험합니다.

---

## 연구 목표

> **"Transfer Learning을 적용하면 B논문의 Scratch + 50% 증강과 동등한 성능을
> 더 적은 증강 비율(25% 이하)로 달성할 수 있다"**

---

## 데이터셋

- **WM-811K** (Kaggle): https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map
- 전체 811,457개 웨이퍼 맵 중 라벨링된 172,950개 사용
- 9개 클래스: None, Center, Donut, Edge-Loc, Edge-Ring, Loc, Near-Full, Random, Scratch

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

- 모델: **ResNet50** (고정)
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
