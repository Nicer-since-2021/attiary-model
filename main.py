from flask import Flask, request, jsonify, abort
from werkzeug.exceptions import BadRequest
import emotion_classification

app = Flask(__name__)


@app.route('/')
def hello():
    return "deep learning server is running 💗"


@app.route('/api/emotion')
def classifyEmotion():
    sentence = request.args.get("s")
    if sentence is None or len(sentence) == 0:
        raise BadRequest()

    result = emotion_classification.predict(sentence)
    print("[*] 감정 분석 결과: " + result)
    return result
