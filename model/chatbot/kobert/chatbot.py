import random

import torch
from kobert_transformers import get_tokenizer
from model.chatbot.kobert.classifier import KoBERTforSequenceClassfication, kobert_input
from util.depression import Depression
from util.emotion import Emotion


def load_wellness_data():
    root_path = '.'
    category_path = f"{root_path}/data/wellness_dialog_category.txt"
    answer_path = f"{root_path}/data/wellness_dialog_answer.txt"
    emotion_path = f"{root_path}/data/wellness_dialog_emotion.txt"
    depression_path = f"{root_path}/data/wellness_dialog_depression.txt"

    c_f = open(category_path, 'r')
    a_f = open(answer_path, 'r')
    e_f = open(emotion_path, 'r')
    d_f = open(depression_path, 'r')

    category_lines = c_f.readlines()
    answer_lines = a_f.readlines()
    emotion_lines = e_f.readlines()
    depression_lines = d_f.readlines()

    category = {}
    answer = {}
    emotion = {}
    depression = {}

    for line_num, line_data in enumerate(category_lines):
        data = line_data.split('    ')
        category[data[1][:-1]] = data[0]

    for line_num, line_data in enumerate(answer_lines):
        data = line_data.split('    ')
        keys = answer.keys()
        if (data[0] in keys):
            answer[data[0]] += [data[1][:-1]]
        else:
            answer[data[0]] = [data[1][:-1]]

    for line_num, line_data in enumerate(emotion_lines):
        data = line_data.split('\t')
        emotion[data[0]] = data[1]

    for line_num, line_data in enumerate(depression_lines):
        data = line_data.split('\t')
        depression[data[0]] = data[1]

    return category, answer, emotion, depression


def predict_emotion(sent):
    data = kobert_input(tokenizer, sent, device, 512)

    output = model(**data)

    logit = output[0]
    softmax_logit = torch.softmax(logit, dim=-1)
    softmax_logit = softmax_logit.squeeze()

    max_index = torch.argmax(softmax_logit).item()
    return Emotion.to_num(emotion[str(max_index)])


def predict_depression(sent):
    data = kobert_input(tokenizer, sent, device, 512)

    output = model(**data)

    logit = output[0]
    softmax_logit = torch.softmax(logit, dim=-1)
    softmax_logit = softmax_logit.squeeze()

    max_index = torch.argmax(softmax_logit).item()
    return Depression.to_num(depression[str(max_index)])


def chat(sent):
    data = kobert_input(tokenizer, sent, device, 512)

    output = model(**data)

    logit = output[0]
    softmax_logit = torch.softmax(logit, dim=-1)
    softmax_logit = softmax_logit.squeeze()

    max_index = torch.argmax(softmax_logit).item()
    max_index_value = softmax_logit[torch.argmax(softmax_logit)].item()

    answer_list = answer[category[str(max_index)]]
    answer_len = len(answer_list) - 1
    answer_index = random.randint(0, answer_len)
    return answer_list[answer_index], max_index, category[str(max_index)], max_index_value
    # print(f'Answer: {answer_list[answer_index]}, index: {max_index}, softmax_value: {max_index_value}')
    # print('-' * 50)

Emotion = Emotion()
Depression = Depression()

root_path = '.'
checkpoint_path = f"{root_path}/checkpoint"
save_ckpt_path = f"{checkpoint_path}/chatbot_kobert.pth"

# 답변과 카테고리 불러오기
category, answer, emotion, depression = load_wellness_data()

ctx = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(ctx)

# 저장한 Checkpoint 불러오기
checkpoint = torch.load(save_ckpt_path, map_location=device)

model = KoBERTforSequenceClassfication()
model.load_state_dict(checkpoint['model_state_dict'], strict=False)

model.eval()

tokenizer = get_tokenizer()

print("=" * 50)
print("[*] kobert chatbot test")
print("\'특별한 이유가 없는데 그냥 불안하고 눈물이 나와\' 챗봇 응답: ", end='')
print(chat("특별한 이유가 없는데 그냥 불안하고 눈물이 나와"))
print("\'이 세상에서 완전히 사라지고 싶어\' 챗봇 응답: ", end='')
print(chat("이 세상에서 완전히 사라지고 싶어"))
print("\'가슴이 답답해서 터질 것만 같아요.\' 챗봇 응답: ", end='')
print(chat("가슴이 답답해서 터질 것만 같아요."))
print("\'남들이 나를 어떻게 생각할지 신경쓰게 돼\' 챗봇 응답: ", end='')
print(chat("남들이 나를 어떻게 생각할지 신경쓰게 돼"))
print("\'자존감이 낮아지는 것 같아\' 챗봇 응답: ", end='')
print(chat("자존감이 낮아지는 것 같아"))
print("\'뭘 해도 금방 지쳐\' 챗봇 응답: ", end='')
print(chat("뭘 해도 금방 지쳐"))
print("\'걔한테 진짜 크게 배신 당했어\' 챗봇 응답: ", end='')
print(chat("걔한테 진짜 크게 배신 당했어"))
print("\'내일 놀이공원 갈건데 사람 별로 없었으면 좋겠다\' 챗봇 응답: ", end='')
print(chat("내일 놀이공원 갈건데 사람 별로 없었으면 좋겠다"))
print("\'오늘은 구름이랑 달이 너무너무 예쁘더라\' 챗봇 응답: ", end='')
print(chat("오늘은 구름이랑 달이 너무너무 예쁘더라"))
print("\'그래도 내가 머리는 좀 좋아\' 챗봇 응답: ", end='')
print(chat("그래도 내가 머리는 좀 좋아"))
print("=" * 50)
