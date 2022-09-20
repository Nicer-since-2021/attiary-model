# 아띠어리 NLP 모델 api 서버

## 간단 설명

파이썬의 웹 프레임워크 중 flask를 사용하였습니다.

우울한 문장의 비율을 알려줍니다.

입력한 문장이 8가지 감정(행복/희망/중립/슬픔/분노/불안/후회/피곤) 중 어떤 감정에 해당하는지 반환해줍니다.

kogpt2 기반 공감형 챗봇, kobert 기반 위로형 챗봇을 운영합니다.

## URL

학교에서 tencent cloud를 지원받아 gpu 서버에서 운영했습니다.

## API DOCS

### 1. 일기 전체 내용에 대한 감정 분석

#### 요청 방법

 `get` /diary?s=특별한 이유가 없는데 그냥 불안하고 눈물이 나. 이 세상에서 완전히 사라지고 싶어. 가슴이 답답해서 터질 것만 같아. 자존감이 낮아지는 것 같아.

#### response body

소수점 아래 둘째자리까지 표현

```json
{
    "anger": 0.0,
    "anxiety": 0.25,
    "depression": 0.25,
    "hope": 0.0,
    "joy": 0.0,
    "neutrality": 0.0,
    "regret": 0.0,
    "sadness": 0.75,
    "tiredness": 0.0
}
```

감정은 1문장이 불안으로, 3문장이 슬픔으로 분류되었으며, 4문장 중 1문장은 우울이 감지되었음을 알 수 있습니다.

### 2. 음악 변경을 위한 짧은 내용에 대한 빠른 감정 분류

#### 요청 방법

`get` /emotion?s=시험 잘 볼 수 있을 것 같아!

#### response body

0: 기쁨, 1: 희망, 2: 중립, 3: 분노, 4: 슬픔, 5: 불안, 6: 피곤, 7: 후회를 뜻합니다.

```json
{
    "emotion": "희망",
    "emotion_no": 1
}
```

### 3. kogpt2 기반 공감형 아띠 (반응봇)

#### 요청 방법

`get` /chatbot/g?s=자존감이 낮아지는 것 같아

#### response body

```json
{
    "answer": "당신은 태어난 그 자체만으로 축복과 사랑을 받을 자격이 있는 사람이에요."
}
```

### 4. Kobert 기반 위로형 아띠 (반응봇)

#### 요청 방법

`get` /chatbot/b?s=자존감이 낮아지는 것 같아

#### response body

```json
{
    "answer": "실패할 때도 있을 거예요. 하지만 그 잠깐의 실패가 당신의 모든 걸 말해주지는 않아요"
}
```

## 코드 실행 방법

### 라이브러리 설치

아래의 명령어를 입력한다.
```
pip install -r requirements.txt
```
설치가 너무 느리거나 오류가 날 경우 [아래의 명령어](https://uiandwe.tistory.com/1330)로 설치한다.
```
pip install --upgrade --no-deps --force-reinstall -r requirements.txt
```
참고로 저는 windows, python 3.8.5 버전 환경에서 돌렸습니다.

### 학습 시켜 모델 파일 얻기

코드를 실행시키기 위해서는 모델 파일이 필요합니다.

프로젝트 최상단에 `checkpoint` directory를 만들고 하위에 4개의 모델 파일이 아래의 이름으로 존재합니다.

각각의 학습 코드입니다.

- 긍정/중립/부정 분류에 쓰이는 `emotion_p.pth`: 
- 기쁨/희망 분류에 쓰이는 `emotion_pn.pth`: 
- 중립/슬픔/분노/불안/후회/피곤 분류에 쓰이고, 우울/비우울 분류에 쓰이고, 위로형 반응봇에 쓰이는 `kobert_chatbot.pth`: https://github.com/hoit1302/kobert-wellness-chatbot
- 공감형 반응봇에 쓰이는 `kogpt2_chatbot.ckpt`: https://github.com/hoit1302/kogpt2-wellness-chatbot


감정 분류와 관련된 학습은 [해당 블로그 링크](https://hoit1302.tistory.com/159)에서 설명과 함께 볼 수 있습니다. 

챗봇과 관련된 학습은 [해당 블로그 링크](https://hoit1302.tistory.com/162)에서 설명과 함께 볼 수 있습니다. 

## 배포

dockerhub에 image로 만들어 배포합니다.

[hoit1302/attiary_model](https://hub.docker.com/repository/docker/hoit1302/attiary_model) repository에서 만나보실 수 있습니다.

현재, 2.2 버전이 최신 버전입니다.

```
docker pull hoit1302/attiary_model:2.2
```
