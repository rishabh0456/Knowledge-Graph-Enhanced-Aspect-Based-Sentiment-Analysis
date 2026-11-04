# train.py - training, validation and testing loop
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score)
from sklearn.model_selection import train_test_split
from data import load_semeval, make_collate
from kg import load_kg
from model import KGEnhancedABSA
 
def set_seed(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
 
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for b in loader:
        y = b.pop("labels")
        b = {k: v.to(device) for k, v in b.items()}
        ps += model(**b).argmax(-1).cpu().tolist()
        ys += y.tolist()
    return ys, ps
 
def main(train_xml, test_xml, kg_path, use_kg=True,
         epochs=5, bs=32, lr=2e-5):
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    kg = load_kg(kg_path) if use_kg else {}
    data = load_semeval(train_xml)
    train, dev = train_test_split(
        data, test_size=0.1, stratify=[d[2] for d in data],
        random_state=42)
    test = load_semeval(test_xml)
    collate = make_collate(tok, kg)
    tl = DataLoader(train, bs, shuffle=True, collate_fn=collate)
    dl = DataLoader(dev, 64, collate_fn=collate)
    el = DataLoader(test, 64, collate_fn=collate)
 
    model = KGEnhancedABSA().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=0.01)
    total = epochs * len(tl)
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
    loss_fn = torch.nn.CrossEntropyLoss()
    best = 0.0
    for ep in range(epochs):
        model.train()
        for b in tl:
            y = b.pop("labels").to(device)
            b = {k: v.to(device) for k, v in b.items()}
            loss = loss_fn(model(**b), y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
        ys, ps = evaluate(model, dl, device)
        f1 = f1_score(ys, ps, average="macro")
        print(f"epoch {ep + 1}: dev macro-F1 = {f1:.4f}")
        if f1 > best:
            best = f1
            torch.save(model.state_dict(), "best.pt")
 
    model.load_state_dict(torch.load("best.pt"))
    ys, ps = evaluate(model, el, device)
    print(classification_report(
        ys, ps, target_names=["negative", "neutral", "positive"],
        digits=4))
    print(confusion_matrix(ys, ps))
 
if __name__ == "__main__":
    main("Laptop_Train.xml", "Laptops_Test.xml", "kg.json")

