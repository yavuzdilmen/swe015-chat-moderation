"""Loading OffensEval-TR 2020 (Çöltekin, 2020).

Labels: 0 = NOT (clean), 1 = OFF (offensive).
Splits used in this project:
  * train  -> 90 % of the official training set (stratified, seed 42)
  * val    -> 10 % of the official training set (model selection only)
  * test   -> the official OffensEval-TR test set = the SAME held-out set for every model

Three ways to get the data, tried in this order:
  1. a local folder (--data_dir) with train.csv / test.csv  (columns: text,label)
     or the original TSV files from the shared task
  2. the official zip of the dataset author, downloaded to ./data
  3. Hugging Face Hub: coltekin/offenseval2020_tr
  4. --smoke : a tiny synthetic data set, only to test that the code runs
"""
import csv
import os
import random

import pandas as pd
from sklearn.model_selection import train_test_split

HF_IDS = ["coltekin/offenseval2020_tr", "offenseval2020_tr"]
LABELS = ["clean", "offensive"]


def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    text_col = next(c for c in ["text", "tweet", "Tweet"] if c in df.columns)
    label_col = next(c for c in ["label", "subtask_a", "Subtask_a"] if c in df.columns)
    out = pd.DataFrame({"text": df[text_col].astype(str), "label": df[label_col]})
    if not pd.api.types.is_numeric_dtype(out["label"]):
        out["label"] = out["label"].str.strip().str.upper().map({"OFF": 1, "NOT": 0})
    out = out.dropna()
    out["label"] = out["label"].astype(int)
    return out.reset_index(drop=True)


ZIP_URL = "https://coltekin.github.io/offensive-turkish/offenseval2020-turkish.zip"


def _find(root, pred):
    for d, _, files in os.walk(root):
        for f in sorted(files):
            if pred(f):
                return os.path.join(d, f)
    return None


def _from_dir(data_dir: str):
    if os.path.exists(os.path.join(data_dir, "train.csv")):
        return (_standardize(pd.read_csv(os.path.join(data_dir, "train.csv"))),
                _standardize(pd.read_csv(os.path.join(data_dir, "test.csv"))))
    # original shared-task files (searched recursively, e.g. after unzipping)
    tr = _find(data_dir, lambda f: "training" in f and f.endswith(".tsv"))
    te = _find(data_dir, lambda f: "testset" in f and f.endswith(".tsv"))
    lab = _find(data_dir, lambda f: "labela" in f.lower())
    if not (tr and te and lab):
        raise FileNotFoundError(f"OffensEval-TR files not found in {data_dir}")
    train = pd.read_csv(tr, sep="\t", quoting=csv.QUOTE_NONE, dtype=str)
    test = pd.read_csv(te, sep="\t", quoting=csv.QUOTE_NONE, dtype=str)
    gold = pd.read_csv(lab, sep=None, engine="python", header=None, dtype=str)
    gold = pd.DataFrame({"id": gold.iloc[:, 0].str.strip(), "subtask_a": gold.iloc[:, -1]})
    test = test.drop(columns=[c for c in test.columns if c.lower() == "subtask_a"])
    test["id"] = test["id"].str.strip()
    test = test.merge(gold, on="id")
    return _standardize(train), _standardize(test)


def _from_zip(cache="data"):
    """Official release by the dataset author (Çöltekin, 2020)."""
    import urllib.request
    import zipfile
    os.makedirs(cache, exist_ok=True)
    zpath = os.path.join(cache, "offenseval2020-turkish.zip")
    if not os.path.exists(zpath):
        print("Downloading", ZIP_URL)
        urllib.request.urlretrieve(ZIP_URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(cache)
    return _from_dir(cache)


def _from_hub():
    from datasets import load_dataset
    last_err = None
    for ds_id in HF_IDS:
        for kwargs in ({"revision": "refs/convert/parquet"}, {}, {"trust_remote_code": True}):
            try:
                ds = load_dataset(ds_id, **kwargs)
                return (_standardize(ds["train"].to_pandas()),
                        _standardize(ds["test"].to_pandas()))
            except Exception as e:  # noqa: BLE001
                last_err = e
    raise RuntimeError(
        "Could not download OffensEval-TR from the Hugging Face Hub. "
        "Download it manually and pass --data_dir (see README). Last error: " + repr(last_err))


def _synthetic(n_train=1200, n_test=300, seed=0):
    """Fake data, ONLY for --smoke tests. Never report numbers from it."""
    rng = random.Random(seed)
    clean = ["bugün maç çok güzeldi", "takım iyi oynadı", "yarın görüşürüz kanka",
             "bu oyun harika", "güzel bir gün", "teşekkürler arkadaşlar", "level atladım",
             "akşam yemeği hazır", "yeni sezon başlıyor", "herkese iyi oyunlar"]
    bad = ["salak", "aptal", "gerizekalı", "mal", "şerefsiz", "ahmak", "dangalak"]
    fill = ["sen", "tam", "bir", "herif", "oyun", "bu", "takım", "ne", "kadar", "ya"]

    def make(n):
        rows = []
        for _ in range(n):
            if rng.random() < 0.3:
                w = rng.choice(bad)
                w = rng.choice([w, w.upper(), w.replace("a", "4"), w + w[-1] * 3])
                words = rng.sample(fill, 3) + [w]
                rng.shuffle(words)
                rows.append((" ".join(words), 1))
            else:
                rows.append((rng.choice(clean) + " " + " ".join(rng.sample(fill, 2)), 0))
        return pd.DataFrame(rows, columns=["text", "label"])
    return make(n_train), make(n_test)


def load_splits(data_dir: str | None = None, smoke: bool = False, seed: int = 42):
    if smoke:
        train_full, test = _synthetic()
    elif data_dir:
        train_full, test = _from_dir(data_dir)
    else:
        try:
            train_full, test = _from_zip()
        except Exception as e:  # noqa: BLE001
            print("Official zip failed (", repr(e), ") -> trying the Hugging Face Hub")
            train_full, test = _from_hub()
    train, val = train_test_split(train_full, test_size=0.1, stratify=train_full["label"],
                                  random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test


def describe(name, df):
    off = int(df["label"].sum())
    return f"{name}: {len(df)} messages, offensive={off} ({off / len(df):.1%})"
