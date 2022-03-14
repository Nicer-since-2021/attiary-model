import torch

import gluonnlp as nlp
import numpy as np
from torch.utils.data import Dataset, DataLoader

from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from bert_classifier import BERTClassifier

model = torch.load('./emotion_clsf_0311.pth', map_location=torch.device('cpu'))

kobert_model, vocab = get_pytorch_kobert_model()

tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)


class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len, pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return self.sentences[i] + (self.labels[i],)

    def __len__(self):
        return len(self.labels)


def predict(predict_sentence):
    max_len = 100
    batch_size = 64

    data = [predict_sentence, '0']
    dataset_another = [data]

    another_test = BERTDataset(dataset_another, 0, 1, tok, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=0)

    model.eval()

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        # print(batch_id + '\n')
        token_ids = token_ids.long()
        segment_ids = segment_ids.long()

        valid_length = valid_length
        label = label.long()

        out = model(token_ids, valid_length, segment_ids)

        for i in out:
            logits = i
            logits = logits.detach().cpu().numpy()

            if np.argmax(logits) == 0:
                return "기쁨"
            elif np.argmax(logits) == 1:
                return "희망"
            elif np.argmax(logits) == 2:
                return "중립"
            elif np.argmax(logits) == 3:
                return "슬픔"
            elif np.argmax(logits) == 4:
                return "분노"
            elif np.argmax(logits) == 5:
                return "불안"
            elif np.argmax(logits) == 6:
                return "피곤"
            elif np.argmax(logits) == 7:
                return "후회"


print(predict("아 짜증나!"))



