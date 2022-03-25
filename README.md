# 아띠어리 NLP 모델 api 서버

파이썬의 웹 프레임워크 중 flask를 사용하였습니다.

입력한 문장이 8가지 감정(행복/희망/중립/슬픔/분노/불안/후회/피곤) 중 어떤 감정에 해당하는지 반환해줍니다.

입력한 문장에 대해 공감해주거나 반응해주는 문장을 반환해주는 기능을 추가할 예정입니다.

## URL
https://api.attiary.net 🎉

2024년 말까지 운영할 계획입니다.

## API DOCS

개발 후 수정할 예정입니다 

## 코드 실행 방법

### 0. 개발 IDE 설치
사용한 Python IDE: [PyCharm](https://www.jetbrains.com/pycharm/) by JetBrains

다른 IDE를 사용해도 됩니다. 

### 1. 가상 환경 구축하기
#### (1) command 창을 열어 virtualenv를 설치합니다.
```
pip install virtualenv
```
#### (2) 프로젝트를 생성하고 싶은 곳으로 이동 한 뒤 새로운 프로젝트 생성합니다. 해당 프로젝트명의 폴더가 생성됩니다.
```
virtualenv 프로젝트명
```
#### (3) 가상 환경 가동합니다.
**Mac**
```
cd 프로젝트명
source bin/activate
```
**Windows**
```
cd 프로젝트명
Scripts/activate
```
왼쪽에 (프로젝트 이름) 으로 현재 위치가 표시되면 활성화되어 있다는 뜻입니다.

<img alt="image" src="https://user-images.githubusercontent.com/68107000/158960095-6eb88b18-f5de-487b-a49e-d2077cfcb978.png">

참고: 개발이 끝난 후 가상 환경 비활성화 합니다.
```
deactivate
```
참고: 가상 환경 삭제는 아래의 명령어로 할 수 있습니다.
```
rm -rf 프로젝트명
```

### 2. 가상환경에 필요한 라이브러리 설치하기

아래의 명령어를 입력한다.
```
pip install -r requirements.txt
```
설치가 너무 느리거나 오류가 날 경우 [아래의 명령어](https://uiandwe.tistory.com/1330)로 설치한다.
```
pip install --upgrade --no-deps --force-reinstall -r requirements.txt
```

**설치 라이브러리 종류**
- flask
- [SKTBrain/KoBERT의 요구 라이브러리](https://github.com/SKTBrain/KoBERT/blob/master/requirements.txt)


### 3. 모델 준비하기

[model_weights.pth.zip](https://drive.google.com/file/d/1-6C1bst8WldaG4Pdx7PlI9oNtdlzWREF/view?usp=sharing)

`model_weights.pth.zip`은 kobert를 활용하여 저희 팀이 직접 재라벨링한 데이터셋으로 파인 튜닝한 모델의 가중치가 저장된 파일을 압축해둔 것입니다.

해당 파일을 다운 받아 압축을 풀어주시고 폴더 구조 최하단에 있는 model_weights.pth 을 프로젝트 내부에 추가해주세요.

참고로 현재, 적은 데이터셋으로 최소한만 학습시켜 둔 모델의 가중치 파일을 두었습니다.

모델을 학습시키는 코드는 [# 링크](https://github.com/Nicer-since-2021/multiclass-emotion-classification-using-KoBERT/blob/main/SOJIN/emotion_classification_kobert.ipynb) 에서 확인할 수 있습니다.

### 4. github의 코드를 다운 받아 실행시키기

해당 레포의 코드를 다운받아 프로젝트 폴더에 적절히 풀어주세요.

<img width="1094" alt="image" src="https://user-images.githubusercontent.com/68107000/158560343-f8c77654-a629-4c8e-bb7a-4ce1883156b7.png">

터미널 창에 아래의 명령어를 입력하여 flask 애플리케이션을 실행할 수 있습니다.

```
FLASK_APP=main.py flask run
```

로그에 올라오는 링크를 클릭하여 잘 접속되는지 확인합니다.

http://127.0.0.1:5000/ 

감정 추출해보기
```
호출 시
http://127.0.0.1:5000/api/emotion?s=프로젝트가 잘 마무리 되었으면 좋겠다

> 희망
을 반환값으로 알려줍니다.
```

**WARNING 해결하기**
```
WARNING: This is a development server. Do not use it in a production deployment.
```
이러한 오류가 발생한다면 터미널에 아래의 명령어를 입력해서 설정을 변경해주세요.
```
export FLASK_ENV=development
```
