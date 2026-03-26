# 🧍 Human Segmentation — 배경 분리 및 합성

AIFFEL Exploration 3 — PixelLib의 DeepLabV3 모델을 활용한 Semantic Segmentation으로 인물을 배경에서 분리하고, 배경 블러 처리 및 크로마키 배경 합성을 구현하는 프로젝트입니다.

---

## 📌 프로젝트 개요

| 항목 | 내용 |
|------|------|
| 목적 | Semantic Segmentation으로 인물 분리 후 배경 블러 / 배경 교체 합성 |
| 모델 | DeepLabV3 (Xception 기반, Pascal VOC 학습) |
| 라이브러리 | PixelLib, OpenCV, NumPy, Matplotlib |
| 세그멘테이션 방식 | Semantic Segmentation (픽셀 단위 클래스 분류) |
| 언어 | Python 3 |

---

## 🗂️ 프로젝트 구조

```
human_segmentation/
├── images/
│   ├── KakaoTalk_20240109_113309155.jpg  # 인물 원본 이미지 1 (사람 + 의자)
│   ├── 107425756bbfd5102867ed0f02c69095.jpg  # 인물 원본 이미지 2 (사람 + 동물)
│   └── th.jpg                             # 크로마키 합성용 새 배경 이미지
├── models/
│   └── deeplabv3_xception_tf_dim_ordering_tf_kernels.h5  # 사전학습 모델
└── aiffelproject_exploration3.ipynb       # 메인 노트북
```

---

## ⚙️ 주요 구현 내용

### 1. 모델 로드 및 Semantic Segmentation
- PixelLib의 `semantic_segmentation` 클래스 사용
- Pascal VOC 사전학습 모델(`deeplabv3_xception`) 다운로드 및 로드
- `segmentAsPascalvoc()`으로 이미지 내 모든 객체를 픽셀 단위로 분류

**Pascal VOC 지원 클래스 (21종)**
```
background, aeroplane, bicycle, bird, boat, bottle, bus, car,
cat, chair, cow, diningtable, dog, horse, motorbike, person,
pottedplant, sheep, sofa, train, tv
```

### 2. 세그멘테이션 마스크 생성
- 컬러맵(256×3 배열)으로 클래스별 고유 색상 코드 생성
- `np.all(output == seg_color, axis=-1)`로 특정 클래스 픽셀만 True인 Boolean 마스크 추출
- `astype(np.uint8) * 255`로 마스크를 0/255 이진 이미지로 변환

### 3. 구현한 기능

| 기능 | 방법 |
|------|------|
| **배경 블러** | `cv2.blur(kernel=(26,26))`로 배경만 흐리게, 인물은 선명하게 유지 |
| **인물 블러** | 마스크 반전 후 인물 영역만 블러 처리 |
| **크로마키 배경 합성** | `np.where(mask==255, img_orig, img_new_background)` |
| **다중 클래스 마스크** | 사람 마스크 + 의자 마스크를 `logical_or`로 결합하여 함께 추출 |

### 4. 핵심 처리 파이프라인

```
원본 이미지 로드 (cv2.imread)
     ↓
Semantic Segmentation (PixelLib DeepLabV3)
     ↓
클래스별 컬러맵 기반 Boolean 마스크 생성
     ↓
마스크 이진화 (0 / 255)
     ↓
  ┌──────────────────────────────────────┐
  │  배경 블러            배경 교체       │
  │  cv2.blur + bitwise  np.where + 새   │
  │  연산으로 배경만 흐림  배경 이미지 합성 │
  └──────────────────────────────────────┘
```

---

## 🧪 시도한 실험들

| 실험 | 결과 |
|------|------|
| 인물(person)만 마스크 후 배경 블러 | 기본 분리 성공, 경계 일부 부정확 |
| 인물 블러 + 배경 선명 (반전 적용) | 인물만 흐리게 처리 성공 |
| 사람 + 의자 다중 클래스 마스크 결합 | `logical_or`로 두 클래스 동시 추출 성공 |
| 크로마키 배경 합성 | 새 배경 이미지로 교체 성공 (`cv2.resize`로 크기 통일) |
| 동물 이미지 적용 | 스카프 등 미정의 클래스는 분리 불가 확인 |

---

## ⚠️ 발견한 문제점

- **Semantic Segmentation의 한계**: 같은 클래스의 객체들이 하나로 묶여 개별 분리 불가
- **경계 처리**: 인물과 배경의 경계 부분이 깔끔하지 않음 (특히 스카프, 머리카락 등 세밀한 영역)
- **의자 오인식**: 의자 클래스 분류 시 의자가 아닌 영역도 의자로 인식하는 경우 발생
- **동물 스카프**: Pascal VOC에 스카프 클래스가 없어 동물과 함께 블러 처리됨
- **이미지 경로**: `os.getenv('HOME')` + 문자열 연결 방식에서 경로 오류 발생 → `os.path.join()`으로 해결

---

## 📈 개선 방향

- **Instance Segmentation** 적용으로 동일 클래스 내 개별 객체 분리
- 더 많은 클래스 / 더 큰 데이터셋으로 학습된 모델 사용
- 커널 사이즈 조절 및 다른 블러 함수(Gaussian Blur 등) 적용으로 경계 개선
- Morphological 연산을 활용한 마스크 경계 후처리

---

## 🛠️ 실행 환경

```bash
pip install pixellib opencv-python numpy matplotlib
```

모델 자동 다운로드:
```python
# 노트북 실행 시 아래 URL에서 자동으로 다운로드됩니다
# https://github.com/ayoolaolafenwa/PixelLib/releases/download/1.1/
# deeplabv3_xception_tf_dim_ordering_tf_kernels.h5
```

---

## 📝 회고

> Semantic Segmentation을 활용해 배경 교체와 블러 합성을 구현했습니다.
> 의자 마스크와 인물 마스크를 결합하여 원하는 클래스를 함께 분리하는 방법으로 문제를 해결했습니다.
> 경계 처리의 한계와 동일 클래스 내 개별 분리 불가 문제를 경험하며, Instance Segmentation의 필요성을 느꼈습니다.

---

## 📚 참고

- [PixelLib 공식 문서](https://pixellib.readthedocs.io/)
- [DeepLabV3 논문 (Chen et al., 2017)](https://arxiv.org/abs/1706.05587)
- [Pascal VOC Dataset](http://host.robots.ox.ac.uk/pascal/VOC/)
- AIFFEL Exploration 3
