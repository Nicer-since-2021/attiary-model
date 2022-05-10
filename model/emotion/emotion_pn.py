import gluonnlp as nlp
import numpy as np
import torch
from kobert.pytorch_kobert import get_pytorch_kobert_model
from kobert.utils import get_tokenizer
from torch.utils.data import Dataset

from model.emotion import classifier
from util.emotion import Emotion

kobert_model, vocab = get_pytorch_kobert_model()
model = classifier.BERTClassifier(kobert_model, dr_rate=0.5, num_classes=3)

ctx = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(ctx)

emotion_clsf_weights_file = "./checkpoint/emotion_pn.pth"
model.load_state_dict(torch.load(emotion_clsf_weights_file, map_location=device))
model.eval()

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

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long()
        segment_ids = segment_ids.long()

        valid_length = valid_length
        label = label.long()

        out = model(token_ids, valid_length, segment_ids)

        for i in out:
            logits = i
            logits = logits.detach().cpu().numpy()

            return np.argmax(logits)
