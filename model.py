# model.py - BERT + knowledge attention + gated fusion
import torch
import torch.nn as nn
from transformers import AutoModel
 
class KGEnhancedABSA(nn.Module):
    def __init__(self, name="bert-base-uncased", n_cls=3, drop=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(name)
        d = self.bert.config.hidden_size
        self.q = nn.Linear(d, d)           # query projection
        self.gate = nn.Linear(2 * d, d)    # fusion gate
        self.drop = nn.Dropout(drop)
        self.cls = nn.Linear(d, n_cls)
 
    def concept_vectors(self, kg_ids, kg_mask):
        emb = self.bert.embeddings.word_embeddings(kg_ids)   # B,K,L,d
        m = kg_mask.unsqueeze(-1)
        return (emb * m).sum(2) / m.sum(2).clamp(min=1)      # B,K,d
 
    def forward(self, input_ids, attention_mask, token_type_ids,
                kg_ids, kg_mask, kg_valid):
        h = self.bert(input_ids=input_ids,
                      attention_mask=attention_mask,
                      token_type_ids=token_type_ids
                      ).last_hidden_state[:, 0]              # B,d
        c = self.concept_vectors(kg_ids, kg_mask)            # B,K,d
        e = torch.bmm(c, self.q(h).unsqueeze(-1)).squeeze(-1)
        e = e / (h.size(-1) ** 0.5)
        e = e.masked_fill(kg_valid == 0, -1e9)
        has_kg = (kg_valid.sum(-1, keepdim=True) > 0).float()
        alpha = torch.softmax(e, dim=-1) * has_kg            # B,K
        k = torch.bmm(alpha.unsqueeze(1), c).squeeze(1)      # B,d
        g = torch.sigmoid(self.gate(torch.cat([h, k], dim=-1)))
        z = g * h + (1 - g) * k
        return self.cls(self.drop(z))                        # logits
