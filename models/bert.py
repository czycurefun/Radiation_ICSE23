import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel
# from transformers import BertModel
from params import model_param as mp
import torch.nn.functional as F

model = BertModel.from_pretrained('bert-base-cased')


class BERTEncoder(nn.Module):

    def __init__(self):
        super(BERTEncoder, self).__init__()
        self.restored = False
        self.encoder = model
    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.encoder(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        hidden_state, feat = self.encoder(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        return feat, feat


class BERTClassifier(nn.Module):

    def __init__(self):
        super(BERTClassifier, self).__init__()
        self.restored = False
        self.classifier = nn.Sequential(nn.LeakyReLU(inplace=True),
                                        nn.Dropout(mp.dropout),
                                        nn.Linear(mp.c_input_dims, mp.c_hidden_dims),
                                        nn.Linear(mp.c_hidden_dims, mp.c_output_dims))
        self.apply(self.init_bert_weights)

    def forward(self, x):
        out = self.classifier(x)
        return out

    def init_bert_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.normal_(mean=0.0, std=0.02)
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias