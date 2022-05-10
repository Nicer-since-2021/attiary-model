import os
from model.chatbot.kogpt2 import chatbot as ch_v1
from model.chatbot.kobert import chatbot as ch_v2
from model.emotion import service as emotion
from util.emotion import Emotion
from util.depression import Depression
from flask import Flask, request, jsonify
from werkzeug.exceptions import BadRequest
from kss import split_sentences

app = Flask(__name__)
Emotion = Emotion()
Depression = Depression()

def hello():
    return "deep learning server is running 💗"


@app.route('/emotion')
def classifyEmotion():
    sentence = request.args.get("s")
    if sentence is None or len(sentence) == 0:
        raise BadRequest()

    result = emotion.predict(sentence)
    print("[*] 감정 분석 결과: " + Emotion.to_string(result))
    return jsonify({
        "emotion_no": int(result),
        "emotion": Emotion.to_string(result)
    })


@app.route('/diary')
def classifyEmotionDiary():
    sentence = request.args.get("s")
    if sentence is None or len(sentence) == 0:
        raise BadRequest()

    predict, dep_predict = predictDiary(sentence)
    return jsonify({
        "joy": predict[Emotion.JOY],
        "hope": predict[Emotion.HOPE],
        "neutrality": predict[Emotion.NEUTRALITY],
        "anger": predict[Emotion.ANGER],
        "sadness": predict[Emotion.SADNESS],
        "anxiety": predict[Emotion.ANXIETY],
        "tiredness": predict[Emotion.TIREDNESS],
        "regret": predict[Emotion.REGRET],
        "depression": dep_predict
    })


@app.route('/chatbot/v1')
def reactChatbotV1():
    sentence = request.args.get("s")
    if sentence is None or len(sentence) == 0:
        raise BadRequest()

    answer = ch_v1.predict(sentence)
    return jsonify({
        "answer": answer
    })


@app.route('/chatbot/v2')
def reactChatbotV2():
    sentence = request.args.get("s")
    if sentence is None or len(sentence) == 0:
        raise BadRequest()

    answer, category, desc, softmax = ch_v2.chat(sentence)
    return jsonify({
        "answer": answer,
        "category": category,
        "category_info": desc
    })


def predictDiary(s):
    total_cnt = 0.0
    dep_cnt = 0
    predict = [0.0 for _ in range(8)]
    for sent in split_sentences(s):
        total_cnt += 1
        predict[emotion.predict(sent)] += 1
        if emotion.predict_depression(sent) == Depression.DEPRESS:
            dep_cnt += 1

    for i in range(7):
        predict[i] = float("{:.2f}".format(predict[i] / total_cnt))
    dep_cnt /= total_cnt
    return predict, dep_cnt


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
