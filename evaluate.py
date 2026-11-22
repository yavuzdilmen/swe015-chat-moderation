"""Evaluate all five models on the SAME held-out test set (official OffensEval-TR test split).

    python evaluate.py            # writes results/results.md, results.json, failures.md, chart
    python evaluate.py --smoke

Metric: macro-F1 (classes are imbalanced).  Size: model files on disk (MB).
Latency: one message at a time on CPU (batch size 1, like a live chat), averaged over
--n_latency messages (default 200, >= 100 as required) after 10 warm-up calls.
"""
import argparse
import json
import os
import platform
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

from moderation.data import describe, load_splits
from moderation.models import MODEL_KEYS, load_all

LATENCY_BUDGET_MS = 50.0


def cpu_name():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def measure_latency(model, texts, n):
    for t in texts[:10]:
        model.predict_proba([t])
    times = []
    for t in texts[:n]:
        t0 = time.perf_counter()
        model.predict_proba([t])
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.percentile(times, 95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--art", default=None)
    ap.add_argument("--n_latency", type=int, default=200)
    ap.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    args = ap.parse_args()
    art = args.art or ("artifacts_smoke" if args.smoke else "artifacts")
    out_dir = "results_smoke" if args.smoke else "results"
    os.makedirs(out_dir, exist_ok=True)
    torch.set_num_threads(args.threads)

    _, _, test = load_splits(args.data_dir, args.smoke)
    texts, y = test["text"].tolist(), test["label"].values
    hardware = f"{cpu_name()}, {args.threads} CPU threads, torch {torch.__version__}, no GPU"
    print(describe("test", test))
    print("Latency hardware:", hardware)

    models = load_all(art)
    rows, preds = [], {}
    # latency messages: a fixed random sample of the test set (same for every model)
    lat_texts = list(np.random.RandomState(0).choice(texts, size=min(len(texts), args.n_latency + 10),
                                                     replace=False))
    for key in MODEL_KEYS:
        m = models[key]
        p = np.asarray(m.predict_proba(texts))
        yhat = (p >= 0.5).astype(int)
        preds[key] = (p, yhat)
        lat, p95 = measure_latency(m, lat_texts, args.n_latency)
        row = {
            "key": key, "model": m.name, "area": m.area,
            "macro_f1": f1_score(y, yhat, average="macro"),
            "f1_offensive": f1_score(y, yhat, pos_label=1),
            "recall_offensive": recall_score(y, yhat, pos_label=1),
            "accuracy": accuracy_score(y, yhat),
            "size_mb": m.size_mb(), "latency_ms": lat, "latency_p95_ms": p95,
            "meets_budget": lat <= LATENCY_BUDGET_MS,
            "confusion": confusion_matrix(y, yhat).tolist(),
        }
        rows.append(row)
        print(f"{m.name:28s} macro-F1={row['macro_f1']:.4f}  size={row['size_mb']:.1f} MB  "
              f"latency={lat:.2f} ms (p95 {p95:.2f})")

    # ---------- results table (markdown) ----------
    md = ["# Results on the held-out test set", "",
          f"Test set: {describe('test', test)}  ",
          f"Latency: batch size 1, mean over {args.n_latency} messages after 10 warm-up calls. "
          f"Hardware: {hardware}.  ",
          f"Latency budget: {LATENCY_BUDGET_MS:.0f} ms per message on CPU.", "",
          "| # | Model | Area | Macro-F1 | F1 (offensive) | Accuracy | Size (MB) | Latency (ms) | p95 (ms) | ≤ budget |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        md.append(f"| {i} | {r['model']} | {r['area']} | {r['macro_f1']:.3f} | {r['f1_offensive']:.3f} | "
                  f"{r['accuracy']:.3f} | {r['size_mb']:.2f} | {r['latency_ms']:.2f} | "
                  f"{r['latency_p95_ms']:.2f} | {'yes' if r['meets_budget'] else 'no'} |")
    fp, q = rows[3], rows[4]
    md += ["", "## Quantization trade-off (BERTurk FP32 → INT8)", "",
           f"* size: {fp['size_mb']:.1f} MB → {q['size_mb']:.1f} MB "
           f"(×{fp['size_mb'] / max(q['size_mb'], 1e-9):.2f} smaller)",
           f"* latency: {fp['latency_ms']:.2f} ms → {q['latency_ms']:.2f} ms "
           f"(×{fp['latency_ms'] / max(q['latency_ms'], 1e-9):.2f} faster)",
           f"* macro-F1: {fp['macro_f1']:.3f} → {q['macro_f1']:.3f} "
           f"(Δ = {q['macro_f1'] - fp['macro_f1']:+.3f})",
           f"* FP32 and INT8 give the same label on "
           f"{(preds['bert_fp32'][1] == preds['bert_int8'][1]).mean():.1%} of the test messages", "",
           "## Confusion matrices  [[TN, FP], [FN, TP]]", ""]
    md += [f"* {r['model']}: {r['confusion']}" for r in rows]
    with open(os.path.join(out_dir, "results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"hardware": hardware, "test": describe("test", test), "rows": rows}, f, indent=2)

    # ---------- failure examples (for the "limitations" slide) ----------
    pq, yq = preds["bert_int8"]
    fn = np.where((y == 1) & (yq == 0))[0]
    fpos = np.where((y == 0) & (yq == 1))[0]
    lex_only = np.where((preds["lexicon"][1] != y) & (yq == y))[0]
    fail = ["# Failure examples of the quantized BERTurk (INT8)", "",
            "## Missed offensive messages (false negatives), most confident first", ""]
    for i in fn[np.argsort(pq[fn])][:10]:
        fail.append(f"* P(off)={pq[i]:.2f} — {texts[i]}")
    fail += ["", "## Clean messages flagged as offensive (false positives)", ""]
    for i in fpos[np.argsort(-pq[fpos])][:10]:
        fail.append(f"* P(off)={pq[i]:.2f} — {texts[i]}")
    fail += ["", "## Lexicon baseline wrong, BERT right (where context helps)", ""]
    for i in lex_only[:10]:
        fail.append(f"* gold={'OFF' if y[i] else 'NOT'} — {texts[i]}")
    with open(os.path.join(out_dir, "failures.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(fail) + "\n")

    # ---------- chart ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
        names = [r["model"].replace(" (quantized)", "") for r in rows]
        colors = ["#9aa5b1", "#9aa5b1", "#5b8def", "#e07a5f", "#2a9d8f"]
        for a, key, title in zip(ax, ["macro_f1", "size_mb", "latency_ms"],
                                 ["Macro-F1 (higher is better)", "Size MB (lower is better)",
                                  "CPU latency ms/message (lower is better)"]):
            vals = [r[key] for r in rows]
            a.barh(names, vals, color=colors)
            a.invert_yaxis()
            a.set_title(title, fontsize=10)
            for j, v in enumerate(vals):
                a.text(v, j, f" {v:.3f}" if key == "macro_f1" else f" {v:.1f}", va="center", fontsize=8)
            if key == "latency_ms":
                a.axvline(LATENCY_BUDGET_MS, ls="--", c="k", lw=1)
            a.spines[["top", "right"]].set_visible(False)
        for a in ax[1:]:
            a.set_yticklabels([])
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "results_chart.png"), dpi=160)
    except Exception as e:  # noqa: BLE001
        print("chart skipped:", e)
    print(f"\nWrote {out_dir}/results.md, results.json, failures.md, results_chart.png")


if __name__ == "__main__":
    main()
