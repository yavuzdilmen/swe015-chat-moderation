# 2-minute demo video script (screen recording + voice)

Record the Gradio app (Colab `!python app.py --share` link) with OBS / Windows Game Bar (Win+Alt+R) / Loom.
Only the first 2:00 are watched. Speak slowly; ~260 words in total.

**0:00–0:15 · Problem** (show the app)
"This is my SWE015 homework, use case 4: real-time chat moderation. A gaming platform has to flag offensive
Turkish messages in under 50 milliseconds per message, on a CPU-only server."

**0:15–1:15 · Live demo** (type NEW messages; the default model is the quantized BERTurk)
1. A clean message, e.g. *"Akşam ranked atalım mı, 9'da online olurum"* → "It is labelled clean with its
   confidence, and the side panel logs the response time: about __ ms, inside the budget."
2. An obvious insult written with leetspeak/repeated letters, e.g. *"s4l4aaak mısın sen"* → "The regex
   normalizer turns this into 'salak', the word is masked in the chat and the message is flagged."
3. **Difficult case:** an insult without a swear word, or a clean sentence containing a "bad-looking" word
   (try a few beforehand, e.g. from failures.md). Tick **Compare all 5 models** → "Here the lexicon baseline
   is wrong but BERT gets it right" (or: "here even BERT fails; this is one of my limitations").
4. Switch the model dropdown to BERTurk FP32 and send one message → "The FP32 model gives the same answer but
   takes __ ms instead of __ ms."

**1:15–1:45 · Architecture and results** (show slides 4 and 8)
"Input is normalized with regular expressions and goes to five models: a lexicon with edit distance, tf-idf
with logistic regression, my own word2vec with a feedforward network, BERTurk, and BERTurk quantized to INT8,
which the app uses. On the official test set, macro-F1 went from __ for the lexicon to __ for BERTurk;
quantization made the model __ times smaller and __ times faster and cost __ points of F1."

**1:45–2:00 · Conclusion**
"The quantized Transformer gives the best trade-off: near-FP32 accuracy within the 50 ms budget. Its main
limitation is __. Next I would try distillation and ONNX Runtime. Thank you."
