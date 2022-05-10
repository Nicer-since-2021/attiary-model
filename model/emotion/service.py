from model.chatbot.kobert import chatbot as emotion_n
from model.emotion import emotion_p
from model.emotion import emotion_pn
from util.depression import Depression
from util.emotion import Emotion
from util.positive_negative import Positive_Negative


def predict(sent):
    result = emotion_pn.predict(sent)
    if result == Positive_Negative.POSITIVE:
        return emotion_p.predict(sent)
    elif result == Positive_Negative.NEUTRAL:
        return Emotion.NEUTRALITY
    elif result == Positive_Negative.NEGATIVE:
        return emotion_n.predict_emotion(sent)

def predict_depression(sent):
    return emotion_n.predict_depression(sent)

""" test """
Emotion = Emotion()
print("=" * 50)
print("[*] emotion classification logic test")
print("\'이대 합격했어!\' 분류 결과: " + Emotion.to_string(predict("이대 합격했어!")))
print("\'시험 잘 볼 수 있을 것 같아!\' 분류 결과: " + Emotion.to_string(predict("시험 잘 볼 수 있을 것 같아!")))
print("\'난 아이스크림이 정말 좋아\' 분류 결과: " + Emotion.to_string(predict("난 아이스크림이 정말 좋아")))
print("\'아 짜증나!\' 분류 결과: " + Emotion.to_string(predict("아 짜증나!")))
print("\'가슴이 답답해서 터질 것만 같아요.\' 분류 결과: " + Emotion.to_string(predict("가슴이 답답해서 터질 것만 같아요.")))
print("\'나 진짜 쪽팔려서 내일 회사 어떻게 가지?\' 분류 결과: " + Emotion.to_string(predict("나 진짜 쪽팔려서 내일 회사 어떻게 가지?")))
print("\'계속 밀려서 이젠 답이 없어\' 분류 결과: " + Emotion.to_string(predict("계속 밀려서 이젠 답이 없어")))
print("=" * 50)

Depression = Depression()
print("=" * 50)
print("[*] depression classification logic test")
print("\'진짜 어이없고 화가 나서 숨이 턱 막히는 기분이었어\' 분류 결과: " + Depression.to_string(predict_depression("진짜 어이없고 화가 나서 숨이 턱 막히는 기분이었어")))
print("\'가끔씩은 외로움이 몰려와\' 분류 결과: " + Depression.to_string(predict_depression("가끔씩은 외로움이 몰려와")))
print("\'하루라도 잠 좀 푹 자고 싶어\' 분류 결과: " + Depression.to_string(predict_depression("하루라도 잠 좀 푹 자고 싶어")))
print("\'그게 3년 전 일인데도 아직도 괴로워\' 분류 결과: " + Depression.to_string(predict_depression("그게 3년 전 일인데도 아직도 괴로워")))
print("\'먼지가 되고 싶어\' 분류 결과: " + Depression.to_string(predict_depression("먼지가 되고 싶어")))
print("\'스트레스를 받은 상태에서 무리해서 기절했었어\' 분류 결과: " + Depression.to_string(predict_depression("스트레스를 받은 상태에서 무리해서 기절했었어")))
print("=" * 50)
