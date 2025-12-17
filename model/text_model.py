# model/text_model.py

import os
import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# 🔒 Force CPU (VERY IMPORTANT for deployment)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.set_grad_enabled(False)


# =========================
# STRESS MODEL
# =========================
class StressModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def _load(self):
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "ValenHumano/roberta-base-bne-detector-de-stress"
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "ValenHumano/roberta-base-bne-detector-de-stress"
            )
            self.model.eval()

    def predict(self, texts):
        self._load()
        scores = []

        for t in texts:
            inputs = self.tokenizer(
                t, return_tensors="pt", truncation=True, max_length=256
            )
            with torch.no_grad():
                out = self.model(**inputs)
            scores.append(out.logits.softmax(dim=1)[0][1].item())

        return scores


# =========================
# DEPRESSION MODEL
# =========================
class DepressionModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def _load(self):
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "ShreyaR/finetuned-roberta-depression"
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "ShreyaR/finetuned-roberta-depression"
            )
            self.model.eval()

    def predict(self, texts):
        self._load()
        scores = []

        for t in texts:
            inputs = self.tokenizer(
                t, return_tensors="pt", truncation=True, max_length=256
            )
            with torch.no_grad():
                out = self.model(**inputs)
            scores.append(out.logits.softmax(dim=1)[0][1].item())

        return scores


# =========================
# EMOTION MODEL
# =========================
class EmotionModel:
    def __init__(self):
        self.pipe = None

    def _load(self):
        if self.pipe is None:
            self.pipe = pipeline(
                "text-classification",
                model="SamLowe/roberta-base-go_emotions",
                top_k=None,
                device=-1  # CPU
            )

    def predict(self, texts):
        self._load()
        rows = []

        for t in texts:
            res = self.pipe(t)[0]
            rows.append({r["label"]: r["score"] for r in res})

        return pd.DataFrame(rows)
