import argparse

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.core.lightning import LightningModule
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel

parser = argparse.ArgumentParser(description='Simsimi based on KoGPT-2')

parser.add_argument('--chat',
                    action='store_true',
                    default=False,
                    help='response generation on given user input')

parser.add_argument('--sentiment',
                    type=str,
                    default='0',
                    help='sentiment for system. 0 is neutral, 1 is negative, 2 is positive.')

parser.add_argument('--model_params',
                    type=str,
                    default='./checkpoint/chatbot_kogpt2.ckpt',
                    help='model binary for starting chat')

parser.add_argument('--train',
                    action='store_true',
                    default=False,
                    help='for training')


U_TKN = '<usr>'
S_TKN = '<sys>'
BOS = '</s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'
PAD = '<pad>'

TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                                                    bos_token=BOS, eos_token=EOS, unk_token='<unk>',
                                                    pad_token=PAD, mask_token=MASK)


class KoGPT2Chat(LightningModule):
    def __init__(self, hparams, **kwargs):
        super(KoGPT2Chat, self).__init__()
        # self.hparams = hparams # read file
        self.neg = -1e18
        self.kogpt2 = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
        self.loss_function = torch.nn.CrossEntropyLoss(reduction='none')

    @staticmethod
    def add_model_specific_args(parent_parser):
        # add model specific args
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--max-len',
                            type=int,
                            default=32,
                            help='max sentence length on input (default: 32)')

        parser.add_argument('--batch-size',
                            type=int,
                            default=96,
                            help='batch size for training (default: 96)')
        parser.add_argument('--lr',
                            type=float,
                            default=5e-5,
                            help='The initial learning rate')
        parser.add_argument('--warmup_ratio',
                            type=float,
                            default=0.1,
                            help='warmup ratio')
        return parser

    def forward(self, inputs):
        output = self.kogpt2(inputs, return_dict=True)
        return output.logits

    def chat(self, input_sentence, sent='0'):
        tok = TOKENIZER
        sent_tokens = tok.tokenize(sent)
        with torch.no_grad():
            q = input_sentence.strip()
            a = ''
            while 1:
                input_ids = torch.LongTensor(tok.encode(U_TKN + q + SENT + sent + S_TKN + a)).unsqueeze(dim=0)
                pred = self(input_ids)
                gen = tok.convert_ids_to_tokens(torch.argmax(pred, dim=-1).squeeze().numpy().tolist())[-1]
                # print(gen) # <pad>
                if gen == EOS or gen == PAD: # PAD 무한 루프 에러 방지
                    break
                a += gen.replace('▁', ' ')
            a = a.strip()
            period_pos = a.rfind(".")
            question_pos = a.rfind("?")
            exclamation_pos = a.rfind("!")
            last_pos = len(a) - 1
            # print (str(period_pos) + " " + str(question_pos) + " " + str(exclamation_pos))
            if last_pos == period_pos or last_pos == question_pos or last_pos == exclamation_pos:
                return a
            mark_pos = max(max(period_pos, question_pos), exclamation_pos)
            a = a[:mark_pos + 1]
            if a == "":
                return "(끄덕끄덕) 듣고 있어요. 더 말씀해주세요!"
            return  a


parser = KoGPT2Chat.add_model_specific_args(parser)
parser = Trainer.add_argparse_args(parser)
args = parser.parse_args()

model = KoGPT2Chat(args)
model = model.load_from_checkpoint(args.model_params)


def predict(sent):
    return model.chat(sent)


print("=" * 50)
print("[*] kogpt2 chatbot test")
print("\'특별한 이유가 없는데 그냥 불안하고 눈물이 나와\' 챗봇 응답: " + predict("특별한 이유가 없는데 그냥 불안하고 눈물이 나와"))
print("\'이 세상에서 완전히 사라지고 싶어\' 챗봇 응답: " + predict("이 세상에서 완전히 사라지고 싶어"))
print("\'가슴이 답답해서 터질 것만 같아요.\' 챗봇 응답: " + predict("가슴이 답답해서 터질 것만 같아요."))
print("\'남들이 나를 어떻게 생각할지 신경쓰게 돼\' 챗봇 응답: " + predict("남들이 나를 어떻게 생각할지 신경쓰게 돼"))
print("\'자존감이 낮아지는 것 같아\' 챗봇 응답: " + predict("자존감이 낮아지는 것 같아"))
print("\'뭘 해도 금방 지쳐\' 챗봇 응답: " + predict("뭘 해도 금방 지쳐"))
print("\'걔한테 진짜 크게 배신 당했어\' 챗봇 응답: " + predict("걔한테 진짜 크게 배신 당했어"))
print("\'내일 놀이공원 갈건데 사람 별로 없었으면 좋겠다\' 챗봇 응답: " + predict("내일 놀이공원 갈건데 사람 별로 없었으면 좋겠다"))
print("\'오늘은 구름이랑 달이 너무너무 예쁘더라\' 챗봇 응답: " + predict("오늘은 구름이랑 달이 너무너무 예쁘더라"))
print("\'그래도 내가 머리는 좀 좋아\' 챗봇 응답: " + predict("그래도 내가 머리는 좀 좋아"))
print("=" * 50)
