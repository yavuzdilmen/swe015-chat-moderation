"""Real-time chat moderation demo (Gradio).

    python app.py              # uses ./artifacts (default model: quantized BERTurk INT8)
    python app.py --share      # also prints a public link (useful on Colab)
    python app.py --smoke      # uses ./artifacts_smoke (test models)

If ./artifacts is missing and the environment variable MODEL_REPO is set
(e.g. MODEL_REPO=yourname/swe015-chat-moderation), the weights are downloaded
from the Hugging Face Hub first.
"""
import argparse
import os
import time

import gradio as gr
import torch

from moderation.models import MODEL_KEYS, load_all
from moderation.normalize import normalize

LATENCY_BUDGET_MS = 50.0
DEFAULT_KEY = "bert_int8"


def ensure_artifacts(art):
    if os.path.isdir(art):
        return
    repo = os.environ.get("MODEL_REPO")
    if not repo:
        raise SystemExit(f"'{art}' not found. Run `python train.py` first or set MODEL_REPO.")
    from huggingface_hub import snapshot_download
    print(f"Downloading weights from https://huggingface.co/{repo} ...")
    snapshot_download(repo_id=repo, local_dir=art)


def mask_message(text, lexicon):
    """Mask the words that the lexicon recognises (with edit distance); if none of the
    words is in the lexicon (BERT flagged it from context), mask the whole message."""
    out, any_masked = [], False
    for tok in text.split():
        norm = normalize(tok)
        hit = any(lexicon._match(t)[1] is not None for t in norm.split()) if norm else False
        out.append(tok[0] + "*" * (len(tok) - 1) if hit and len(tok) > 1 else ("*" if hit else tok))
        any_masked |= hit
    return " ".join(out) if any_masked else "🚫 [mesaj gizlendi / message hidden]"


def build(models):
    names = {k: models[k].name for k in MODEL_KEYS}
    key_by_name = {v: k for k, v in names.items()}
    lexicon = models["lexicon"]

    def moderate(message, chat, log, model_name, compare):
        message = (message or "").strip()
        if not message:
            return "", chat, log, gr.update()
        if hasattr(log, "values"):          # Gradio passes the Dataframe as pandas
            log = log.values.tolist()
        log = [r for r in (log or []) if any(str(c).strip() for c in r)]
        chat = list(chat or [])
        key = key_by_name[model_name]
        t0 = time.perf_counter()
        p = models[key].predict_proba([message])[0]
        ms = (time.perf_counter() - t0) * 1000
        flagged = p >= 0.5
        shown = mask_message(message, lexicon) if flagged else message
        badge = (f"🟥 **offensive** ({p:.2f})" if flagged else f"🟩 **clean** ({1 - p:.2f})")
        badge += f" · {ms:.1f} ms · {names[key]}"
        chat = chat + [{"role": "user", "content": shown},
                       {"role": "assistant", "content": badge}]
        log = [[len(log) + 1, message[:40], names[key], "offensive" if flagged else "clean",
                round(float(p), 3), round(ms, 2), "✓" if ms <= LATENCY_BUDGET_MS else "✗"]] + log
        cmp_rows = []
        if compare:
            for k in MODEL_KEYS:
                t0 = time.perf_counter()
                pk = models[k].predict_proba([message])[0]
                mk = (time.perf_counter() - t0) * 1000
                cmp_rows.append([names[k], "offensive" if pk >= 0.5 else "clean",
                                 round(float(pk), 3), round(mk, 2)])
        return "", chat, log, gr.update(value=cmp_rows, visible=compare)

    with gr.Blocks(title="SWE015 · Real-Time Chat Moderation") as demo:
        gr.Markdown("## 🎮 Real-Time Chat Moderation (Türkçe)\n"
                    "Every message is labelled **clean / offensive** live. Flagged messages are "
                    "masked. The side panel logs the response time of every message "
                    f"(budget: {LATENCY_BUDGET_MS:.0f} ms on CPU).")
        with gr.Row():
            with gr.Column(scale=3):
                try:
                    chat = gr.Chatbot(type="messages", height=430, label="Game chat")
                except TypeError:  # newer Gradio: messages format is the default
                    chat = gr.Chatbot(height=430, label="Game chat")
                msg = gr.Textbox(placeholder="Bir mesaj yaz ve Enter'a bas…", show_label=False)
                with gr.Row():
                    model_dd = gr.Dropdown(list(names.values()), value=names[DEFAULT_KEY],
                                           label="Model")
                    compare = gr.Checkbox(label="Compare all 5 models", value=False)
                cmp = gr.Dataframe(headers=["Model", "Label", "P(offensive)", "ms"],
                                   visible=False, interactive=False, label="All models")
                gr.Examples(["Bugün maç harikaydı, iyi oyundu herkese!",
                             "s4l4aaak mısın sen, adam gibi oyna",
                             "Bu takımın savunması gerçekten berbat."], inputs=msg)
            with gr.Column(scale=2):
                log = gr.Dataframe(headers=["#", "Message", "Model", "Label", "P(off)", "ms",
                                            "≤50ms"], value=[], interactive=False,
                                   label="Response-time log", wrap=True)
                clear = gr.Button("Clear")
        msg.submit(moderate, [msg, chat, log, model_dd, compare], [msg, chat, log, cmp])
        clear.click(lambda: ([], [], gr.update(value=[], visible=False)), None, [chat, log, cmp])
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    art = "artifacts_smoke" if args.smoke else "artifacts"
    ensure_artifacts(art)
    print("Loading models (CPU) ...")
    models = load_all(art)
    for m in models.values():   # warm-up so the first live message is not slow
        m.predict_proba(["merhaba"])
    build(models).launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
