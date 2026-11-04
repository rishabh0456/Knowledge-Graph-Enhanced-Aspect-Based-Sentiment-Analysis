# data.py - SemEval-2014 parser and batch collation
import xml.etree.ElementTree as ET
import torch
from kg import get_knowledge
 
LABELS = {"negative": 0, "neutral": 1, "positive": 2}
 
def load_semeval(path):
    """Return a list of (sentence, aspect_term, label) triples."""
    items = []
    for s in ET.parse(path).getroot().iter("sentence"):
        text = s.find("text").text
        terms = s.find("aspectTerms")
        if terms is None:
            continue
        for t in terms.findall("aspectTerm"):
            pol = t.get("polarity")
            if pol in LABELS:          # 'conflict' is dropped
                items.append((text, t.get("term"), LABELS[pol]))
    return items
 
def make_collate(tok, kg, K=5, max_len=128, max_c=4):
    def collate(batch):
        texts, aspects, labels = zip(*batch)
        enc = tok(list(texts), list(aspects), padding=True,
                  truncation=True, max_length=max_len,
                  return_tensors="pt")
        B = len(batch)
        ids = torch.zeros(B, K, max_c, dtype=torch.long)
        msk = torch.zeros(B, K, max_c)
        val = torch.zeros(B, K)
        for i, a in enumerate(aspects):
            for j, c in enumerate(get_knowledge(a, kg, K)):
                t = tok(c, add_special_tokens=False, truncation=True,
                        max_length=max_c)["input_ids"]
                if not t:
                    continue
                ids[i, j, :len(t)] = torch.tensor(t)
                msk[i, j, :len(t)] = 1
                val[i, j] = 1
        out = dict(enc)
        out.update(kg_ids=ids, kg_mask=msk, kg_valid=val,
                   labels=torch.tensor(labels))
        return out
    return collate
