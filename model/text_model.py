from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch
import pandas as pd

class StressModel:
    def __init__(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "ValenHumano/roberta-base-bne-detector-de-stress"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "ValenHumano/roberta-base-bne-detector-de-stress"
        )

    def predict(self, texts):
        scores = []
        for t in texts:
            inputs = self.tokenizer(t, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                out = self.model(**inputs)
            scores.append(out.logits.softmax(dim=1)[0][1].item())
        return scores


class DepressionModel:
    def __init__(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "ShreyaR/finetuned-roberta-depression"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "ShreyaR/finetuned-roberta-depression"
        )

    def predict(self, texts):
        scores = []
        for t in texts:
            inputs = self.tokenizer(t, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                out = self.model(**inputs)
            scores.append(out.logits.softmax(dim=1)[0][1].item())
        return scores


class EmotionModel:
    def __init__(self):
        self.pipe = pipeline(
            "text-classification",
            model="SamLowe/roberta-base-go_emotions",
            top_k=None
        )

    def predict(self, texts):
        rows = []
        for t in texts:
            res = self.pipe(t)[0]
            rows.append({r["label"]: r["score"] for r in res})
        return pd.DataFrame(rows)
