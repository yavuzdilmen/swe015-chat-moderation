"""Train the whole model ladder and save it to ./artifacts

    python train.py                     # real run (OffensEval-TR from the HF Hub, BERTurk)
    python train.py --data_dir data/    # real run with manually downloaded data
    python train.py --smoke             # 1-minute test on synthetic data + a tiny random BERT
"""
import argparse
import json
import math
import os
import shutil
import time

import numpy as np
import torch
from sklearn.metrics import f1_score

from moderation.data import describe, load_splits
from moderation.models import (BertClassifier, LexiconMatcher, TfidfLR, W2VFFN,
                               light_normalize, quantize_dynamic_int8)


def log(*a):
    print(*a, flush=True)


def macro_f1(y, p, thr=0.5):
    return f1_score(y, (np.asarray(p) >= thr).astype(int), average="macro")


def make_tiny_bert(texts, out_dir):
    """--smoke only: a randomly initialised 2-layer BERT with a WordPiece tokenizer
    trained on the synthetic data (no download needed)."""
    from tokenizers import BertWordPieceTokenizer
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast
    os.makedirs(out_dir, exist_ok=True)
    wp = BertWordPieceTokenizer(lowercase=False, strip_accents=False)
    wp.train_from_iterator(texts, vocab_size=500, min_frequency=1)
    wp.save_model(out_dir)
    tok = BertTokenizerFast(os.path.join(out_dir, "vocab.txt"), do_lower_case=False)
    cfg = BertConfig(vocab_size=tok.vocab_size, hidden_size=64, num_hidden_layers=2,
                     num_attention_heads=2, intermediate_size=128, num_labels=2)
    return BertForSequenceClassification(cfg), tok


def train_bert(train, val, args, out_dir):
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"  device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    if args.smoke:
        model, tok = make_tiny_bert(train["text"].tolist(), out_dir + "_tok")
    else:
        tok = AutoTokenizer.from_pretrained(args.bert_model)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.bert_model, num_labels=2, id2label={0: "clean", 1: "offensive"},
            label2id={"clean": 0, "offensive": 1})
    model.to(device)

    def batches(df, bs, shuffle):
        idx = np.random.permutation(len(df)) if shuffle else np.arange(len(df))
        for i in range(0, len(df), bs):
            part = df.iloc[idx[i:i + bs]]
            enc = tok([light_normalize(t) for t in part["text"]], truncation=True,
                      max_length=args.max_len, padding=True, return_tensors="pt")
            yield {k: v.to(device) for k, v in enc.items()}, torch.tensor(
                part["label"].values, device=device)

    counts = np.bincount(train["label"], minlength=2)
    weights = torch.tensor(counts.sum() / (2 * counts), dtype=torch.float, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * math.ceil(len(train) / args.batch_size)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_f1 = -1
    for ep in range(args.epochs):
        model.train()
        t0, step = time.time(), 0
        for enc, y in batches(train, args.batch_size, True):
            opt.zero_grad()
            with torch.autocast(device_type=device, enabled=use_amp):
                loss = loss_fn(model(**enc).logits.float(), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 100 == 0:
                log(f"  epoch {ep + 1} step {step}  loss={loss.item():.4f}")
        model.eval()
        probs = []
        with torch.no_grad():
            for enc, _ in batches(val, 128, False):
                probs += torch.softmax(model(**enc).logits.float(), -1)[:, 1].tolist()
        f1 = macro_f1(val["label"], probs)
        log(f"  BERT epoch {ep + 1}/{args.epochs}: val macro-F1={f1:.4f} ({time.time() - t0:.0f}s)")
        if f1 > best_f1:
            best_f1 = f1
            model.save_pretrained(out_dir)
            tok.save_pretrained(out_dir)
    if args.smoke:
        shutil.rmtree(out_dir + "_tok", ignore_errors=True)
    return best_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--bert_model", default="dbmdz/bert-base-turkish-cased")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--skip_bert", action="store_true", help="re-train only models 1-3")
    args = ap.parse_args()
    if args.smoke:
        args.out, args.epochs, args.lr = "artifacts_smoke", 3, 5e-4
    np.random.seed(42)
    torch.manual_seed(42)
    os.makedirs(args.out, exist_ok=True)

    train, val, test = load_splits(args.data_dir, args.smoke)
    info = {"train": describe("train", train), "val": describe("val", val),
            "test": describe("test", test), "smoke": args.smoke}
    for v in list(info.values())[:3]:
        log(v)
    Xtr, ytr = train["text"].tolist(), train["label"].tolist()
    Xva, yva = val["text"].tolist(), val["label"].tolist()
    val_scores = {}

    log("\n[1/5] Lexicon + edit distance (rule baseline)")
    best = None
    for thr in [0.6, 0.7, 0.8, 0.9]:
        lex = LexiconMatcher.fit(Xtr, ytr, threshold=thr)
        f1 = macro_f1(yva, lex.predict_proba(Xva))
        log(f"  threshold={thr}: {len(lex.lexicon)} words, val macro-F1={f1:.4f}")
        if best is None or f1 > best[0]:
            best = (f1, thr, lex)
    val_scores["lexicon"], info["lexicon_threshold"] = best[0], best[1]
    best[2].save(os.path.join(args.out, "lexicon.json"))

    log("\n[2/5] tf-idf + logistic regression")
    tf = TfidfLR.fit(Xtr, ytr)
    val_scores["tfidf"] = macro_f1(yva, tf.predict_proba(Xva))
    log(f"  val macro-F1={val_scores['tfidf']:.4f}")
    tf.save(os.path.join(args.out, "tfidf_lr.joblib"))

    log("\n[3/5] own word2vec + feedforward network (SGD)")
    ffn = W2VFFN.fit(Xtr, ytr, Xva, yva, log=log)
    val_scores["w2v_ffn"] = macro_f1(yva, ffn.predict_proba(Xva))
    ffn.save(os.path.join(args.out, "w2v_ffn"))

    bert_dir = os.path.join(args.out, "bert_fp32")
    if not args.skip_bert:
        log(f"\n[4/5] fine-tuning {'tiny random BERT (smoke)' if args.smoke else args.bert_model}")
        val_scores["bert_fp32"] = train_bert(train, val, args, bert_dir)

    log("\n[5/5] dynamic INT8 quantization of BERT (torch.ao.quantization.quantize_dynamic)")
    fp32 = BertClassifier.load(bert_dir)
    int8_model = quantize_dynamic_int8(fp32.model.cpu())
    q_dir = os.path.join(args.out, "bert_int8")
    os.makedirs(q_dir, exist_ok=True)
    torch.save(int8_model.state_dict(), os.path.join(q_dir, "model_int8.pt"))
    fp32.tok.save_pretrained(q_dir)
    q = BertClassifier.load(q_dir, quantized=True)
    val_scores["bert_int8"] = macro_f1(yva, q.predict_proba(Xva))
    log(f"  INT8 val macro-F1={val_scores['bert_int8']:.4f}")

    info["val_macro_f1"] = {k: round(v, 4) for k, v in val_scores.items()}
    info["bert_model"] = "tiny-random (smoke)" if args.smoke else args.bert_model
    info["hparams"] = {k: getattr(args, k) for k in ["epochs", "batch_size", "lr", "max_len"]}
    with open(os.path.join(args.out, "train_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    log("\nDone. Validation macro-F1:", json.dumps(info["val_macro_f1"]))
    log(f"Artifacts saved in ./{args.out}. Next: python evaluate.py" + (" --smoke" if args.smoke else ""))


if __name__ == "__main__":
    main()
