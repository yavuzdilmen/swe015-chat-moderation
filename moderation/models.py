"""The model ladder. Every model exposes the same interface:

    predict_proba(texts: list[str]) -> list[float]   # P(offensive) for each message
    size_mb() -> float                               # size of its files on disk

  1. LexiconMatcher   (Area B)  rule baseline: data-driven offensive lexicon + minimum edit distance
  2. TfidfLR          (Area B)  tf-idf (word + char n-grams) + logistic regression
  3. W2VFFN           (Area A)  our own word2vec (trained on the training tweets) -> averaged
                                -> feedforward network trained with SGD + backpropagation
  4. BertClassifier   (Area C)  BERTurk fine-tuned for classification (FP32)
  5. BertClassifier(quantized=True) (Area D)  same model, dynamic INT8 quantization of all Linear layers
"""
import json
import math
import os
import re
from collections import Counter
from functools import lru_cache

import joblib
import numpy as np
import torch
import torch.nn as nn

from .normalize import normalize, tokenize


def _dir_size_mb(path):
    if os.path.isfile(path):
        return os.path.getsize(path) / 2**20
    total = 0
    for root, _, files in os.walk(path):
        total += sum(os.path.getsize(os.path.join(root, f)) for f in files)
    return total / 2**20


# --------------------------------------------------------------------------------------------
# Area B: minimum edit distance (Levenshtein, insert = delete = substitute = 1)
# --------------------------------------------------------------------------------------------
def min_edit_distance(a: str, b: str, max_d: int = 99) -> int:
    """Dynamic-programming minimum edit distance with an early exit when > max_d."""
    if abs(len(a) - len(b)) > max_d:
        return max_d + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1,            # deletion
                         cur[j - 1] + 1,         # insertion
                         prev[j - 1] + (ca != cb))  # substitution
        if min(cur) > max_d:
            return max_d + 1
        prev = cur
    return prev[-1]


class LexiconMatcher:
    """Rule baseline. The lexicon is learned from the TRAINING set only: tokens that appear
    >= min_count times and whose smoothed precision P(offensive | token) >= threshold.
    A message token matches a lexicon word if their edit distance is <= 1 (length 4-6)
    or <= 2 (length >= 7), which catches misspellings such as "gerizekali" / "gerzekalı"."""

    name = "Lexicon + edit distance"
    area = "B"

    def __init__(self, lexicon=None):
        self.lexicon = lexicon or {}          # word -> precision
        self._by_len = {}
        self._index()

    def _index(self):
        self._by_len = {}
        for w in self.lexicon:
            self._by_len.setdefault(len(w), []).append(w)
        self._match.cache_clear()

    @staticmethod
    def _allowed(n):
        return 0 if n < 4 else (1 if n < 7 else 2)

    @lru_cache(maxsize=50000)
    def _match(self, tok):
        if tok in self.lexicon:
            return self.lexicon[tok], tok, 0
        d_max = self._allowed(len(tok))
        best = (0.0, None, 99)
        for L in range(len(tok) - d_max, len(tok) + d_max + 1):
            for w in self._by_len.get(L, []):
                d = min_edit_distance(tok, w, d_max)
                if d <= d_max:
                    score = self.lexicon[w] * (0.85 ** d)
                    if score > best[0]:
                        best = (score, w, d)
        return best

    def explain(self, text):
        """Return the tokens of the (normalized) message that matched the lexicon."""
        hits = []
        for tok in tokenize(normalize(text)):
            score, w, d = self._match(tok)
            if w is not None:
                hits.append((tok, w, d, score))
        return hits

    def predict_proba(self, texts):
        out = []
        for t in texts:
            hits = self.explain(t)
            # noisy-OR combination of the matched words; 0.05 = prior for "no match"
            p_clean = 1.0
            for *_, s in hits:
                p_clean *= (1 - s)
            out.append(max(0.05, 1 - p_clean) if hits else 0.05)
        return out

    @classmethod
    def fit(cls, texts, labels, min_count=5, threshold=0.7, alpha=1.0):
        pos, tot = Counter(), Counter()
        for t, y in zip(texts, labels):
            toks = set(tokenize(normalize(t)))
            tot.update(toks)
            if y == 1:
                pos.update(toks)
        prior = sum(labels) / len(labels)
        lex = {}
        for w, n in tot.items():
            if n >= min_count and len(w) >= 3:
                prec = (pos[w] + alpha * prior) / (n + alpha)   # add-alpha smoothing
                if prec >= threshold:
                    lex[w] = round(prec, 4)
        return cls(lex)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.lexicon, f, ensure_ascii=False, indent=0)
        self.path = path

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            m = cls(json.load(f))
        m.path = path
        return m

    def size_mb(self):
        return _dir_size_mb(self.path)


# --------------------------------------------------------------------------------------------
# Area B: tf-idf + logistic regression
# --------------------------------------------------------------------------------------------
class TfidfLR:
    name = "tf-idf + LogReg"
    area = "B"

    def __init__(self, pipeline=None):
        self.pipe = pipeline

    @classmethod
    def fit(cls, texts, labels):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import FeatureUnion, Pipeline
        from sklearn.feature_extraction.text import TfidfVectorizer
        feats = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                     sublinear_tf=True, max_features=200000)),
        ])
        pipe = Pipeline([("tfidf", feats),
                         ("lr", LogisticRegression(C=4.0, max_iter=2000,
                                                   class_weight="balanced"))])
        pipe.fit([normalize(t) for t in texts], labels)
        return cls(pipe)

    def predict_proba(self, texts):
        return self.pipe.predict_proba([normalize(t) for t in texts])[:, 1].tolist()

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.pipe, path, compress=3)
        self.path = path

    @classmethod
    def load(cls, path):
        m = cls(joblib.load(path))
        m.path = path
        return m

    def size_mb(self):
        return _dir_size_mb(self.path)


# --------------------------------------------------------------------------------------------
# Area A: own word2vec + feedforward network (SGD + backprop)
# --------------------------------------------------------------------------------------------
class FFN(nn.Module):
    def __init__(self, dim_in=100, hidden=128, dropout=0.3):
        super().__init__()
        # feature standardization (mean / std of the training embeddings), saved with the weights
        self.register_buffer("mu", torch.zeros(dim_in))
        self.register_buffer("sigma", torch.ones(dim_in))
        self.net = nn.Sequential(nn.Linear(dim_in, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden // 2, 2))

    def forward(self, x):
        return self.net((x - self.mu) / self.sigma)


class W2VFFN:
    name = "word2vec + FFN"
    area = "A"

    def __init__(self, kv=None, ffn=None):
        self.kv, self.ffn = kv, ffn

    def embed(self, texts):
        dim = self.kv.vector_size
        X = np.zeros((len(texts), dim), dtype=np.float32)
        for i, t in enumerate(texts):
            vecs = [self.kv[w] for w in tokenize(normalize(t)) if w in self.kv]
            if vecs:
                X[i] = np.mean(vecs, axis=0)
        return X

    @classmethod
    def fit(cls, train_texts, train_labels, val_texts, val_labels, dim=100, epochs=40,
            lr=0.01, seed=42, log=print):
        from gensim.models import Word2Vec
        from sklearn.metrics import f1_score
        torch.manual_seed(seed)
        sents = [tokenize(normalize(t)) for t in train_texts]
        w2v = Word2Vec(sents, vector_size=dim, window=5, min_count=2, sg=1, negative=10,
                       epochs=15, workers=4, seed=seed)
        log(f"  word2vec: vocab={len(w2v.wv)}, dim={dim}")
        m = cls(w2v.wv, FFN(dim))
        Xtr, Xva = torch.tensor(m.embed(train_texts)), torch.tensor(m.embed(val_texts))
        ytr = torch.tensor(np.asarray(train_labels), dtype=torch.long)
        m.ffn.mu.copy_(Xtr.mean(0))
        m.ffn.sigma.copy_(Xtr.std(0) + 1e-6)
        # class weights because offensive messages are the minority class
        counts = torch.bincount(ytr, minlength=2).float()
        weights = counts.sum() / (2 * counts)
        loss_fn = nn.CrossEntropyLoss(weight=weights)
        opt = torch.optim.SGD(m.ffn.parameters(), lr=lr, momentum=0.9)
        best_f1, best_state = -1, None
        for ep in range(epochs):
            m.ffn.train()
            perm = torch.randperm(len(Xtr))
            for i in range(0, len(Xtr), 64):
                idx = perm[i:i + 64]
                opt.zero_grad()
                loss = loss_fn(m.ffn(Xtr[idx]), ytr[idx])
                loss.backward()            # backpropagation
                opt.step()                 # SGD update
            m.ffn.eval()
            with torch.no_grad():
                pred = m.ffn(Xva).argmax(1).numpy()
            f1 = f1_score(val_labels, pred, average="macro")
            if f1 > best_f1:
                best_f1 = f1
                best_state = {k: v.clone() for k, v in m.ffn.state_dict().items()}
            if (ep + 1) % 10 == 0:
                log(f"  FFN epoch {ep + 1}/{epochs}  loss={loss.item():.4f}  val macro-F1={f1:.4f}")
        m.ffn.load_state_dict(best_state)
        m.ffn.eval()
        log(f"  FFN best val macro-F1={best_f1:.4f}")
        return m

    def predict_proba(self, texts):
        with torch.no_grad():
            logits = self.ffn(torch.tensor(self.embed(texts)))
            return torch.softmax(logits, 1)[:, 1].tolist()

    def save(self, folder):
        os.makedirs(folder, exist_ok=True)
        self.kv.save(os.path.join(folder, "w2v.kv"))
        torch.save(self.ffn.state_dict(), os.path.join(folder, "ffn.pt"))
        self.path = folder

    @classmethod
    def load(cls, folder):
        from gensim.models import KeyedVectors
        kv = KeyedVectors.load(os.path.join(folder, "w2v.kv"))
        ffn = FFN(kv.vector_size)
        ffn.load_state_dict(torch.load(os.path.join(folder, "ffn.pt"), map_location="cpu"))
        ffn.eval()
        m = cls(kv, ffn)
        m.path = folder
        return m

    def size_mb(self):
        return _dir_size_mb(self.path)


# --------------------------------------------------------------------------------------------
# Area C + D: BERTurk classifier, FP32 and dynamic-INT8
# --------------------------------------------------------------------------------------------
_URL = re.compile(r"https?://\S+|www\.\S+")
_USER = re.compile(r"@\s?user\b|@\w+", re.IGNORECASE)
_REP = re.compile(r"(.)\1{2,}")


def light_normalize(text: str) -> str:
    """BERTurk is cased and has its own WordPiece tokenizer, so we only remove noise
    (URLs, @USER, long letter repetitions) and keep case + Turkish characters."""
    t = _URL.sub(" ", text)
    t = _USER.sub(" ", t)
    t = _REP.sub(r"\1\1", t)
    return " ".join(t.split())


def quantize_dynamic_int8(model):
    """Area D: post-training dynamic quantization. Weights of every nn.Linear are stored
    as INT8; activations are quantized on the fly at inference time (CPU only)."""
    engines = torch.backends.quantized.supported_engines
    torch.backends.quantized.engine = "fbgemm" if "fbgemm" in engines else "qnnpack"  # x86 / ARM
    return torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)


class BertClassifier:
    area = "C"

    def __init__(self, model, tokenizer, quantized=False, max_len=64):
        self.model, self.tok, self.quantized, self.max_len = model, tokenizer, quantized, max_len
        self.name = "BERTurk INT8 (quantized)" if quantized else "BERTurk FP32"
        if quantized:
            self.area = "C + D"
        self.model.eval()

    def predict_proba(self, texts, batch_size=64):
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                enc = self.tok([light_normalize(t) for t in texts[i:i + batch_size]],
                               truncation=True, max_length=self.max_len, padding=True,
                               return_tensors="pt")
                logits = self.model(**enc).logits
                out += torch.softmax(logits, -1)[:, 1].tolist()
        return out

    @classmethod
    def load(cls, folder, quantized=False):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(folder)
        if not quantized:
            model = AutoModelForSequenceClassification.from_pretrained(folder)
            m = cls(model, tok)
        else:
            fp32_dir = os.path.join(os.path.dirname(folder.rstrip("/")), "bert_fp32")
            model = AutoModelForSequenceClassification.from_pretrained(fp32_dir)
            model = quantize_dynamic_int8(model)
            model.load_state_dict(torch.load(os.path.join(folder, "model_int8.pt"),
                                             map_location="cpu"))
            m = cls(model, tok, quantized=True)
        m.path = folder
        return m

    def size_mb(self):
        if self.quantized:
            return _dir_size_mb(os.path.join(self.path, "model_int8.pt"))
        weights = [f for f in os.listdir(self.path) if f.endswith((".safetensors", ".bin"))]
        return sum(_dir_size_mb(os.path.join(self.path, f)) for f in weights)


# --------------------------------------------------------------------------------------------
MODEL_KEYS = ["lexicon", "tfidf", "w2v_ffn", "bert_fp32", "bert_int8"]


def load_all(art_dir="artifacts", keys=MODEL_KEYS):
    loaders = {
        "lexicon": lambda: LexiconMatcher.load(os.path.join(art_dir, "lexicon.json")),
        "tfidf": lambda: TfidfLR.load(os.path.join(art_dir, "tfidf_lr.joblib")),
        "w2v_ffn": lambda: W2VFFN.load(os.path.join(art_dir, "w2v_ffn")),
        "bert_fp32": lambda: BertClassifier.load(os.path.join(art_dir, "bert_fp32")),
        "bert_int8": lambda: BertClassifier.load(os.path.join(art_dir, "bert_int8"), quantized=True),
    }
    return {k: loaders[k]() for k in keys}
