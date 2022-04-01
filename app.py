import os
from model import emotion, chatbot
from util.emotion import Emotion
from flask import Flask, request, jsonify
from werkzeug.exceptions import BadRequest
from kss import split_sentences

app = Flask(__name__)
Emotion = Emotion()

@app.route('/')
def hello():
    return "deep learning server is running 💗"


@app.route('/emotion')
def classifyEmotion():
    sentence = request.args.get("s")
    if sentence is None or len(sentence) == 0:
        raise BadRequest()

    result = emotion.predict(sentence)
    print("[*] 감정 분석 결과: " + Emotion.to_string(result))
    return Emotion.to_string(result)  # result 로 반환 시 정수값 반환


@app.route('/diary')
def classifyEmotionDiary():
    sentence = request.args.get("s")
    if sentence is None or len(sentence) == 0:
        raise BadRequest()

    predict = predictDiary(sentence)
    return jsonify({
        "happiness": predict[Emotion.HAPPINESS],
        "hope": predict[Emotion.HOPE],
        "neutrality": predict[Emotion.NEUTRALITY],
        "anger": predict[Emotion.ANGER],
        "sadness": predict[Emotion.SADNESS],
        "anxiety": predict[Emotion.ANXIETY],
        "tiredness": predict[Emotion.TIREDNESS],
        "regret": predict[Emotion.REGRET]
    })


@app.route('/chatbot')
def reactChatbot():
    sentence = request.args.get("s")
    if sentence is None or len(sentence) == 0:
        raise BadRequest()

    result = chatbot.predict(sentence)
    return result


def predictDiary(s):
    total_cnt = 0.0
    predict = [0.0 for _ in range(8)]
    for sent in split_sentences(s):
        total_cnt += 1
        predict[emotion.predict(sent)] += 1

    for i in range(7):
        # float로 수정함..!
        predict[i] = float("{:.2f}".format(predict[i]/total_cnt))
    return predict


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5001)))
