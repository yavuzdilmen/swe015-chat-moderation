# Results on the held-out test set

Test set: test: 3528 messages, offensive=716 (20.3%)  
Latency: batch size 1, mean over 200 messages after 10 warm-up calls. Hardware: Intel(R) Xeon(R) CPU @ 2.00GHz, 2 CPU threads, torch 2.14.1+cu130, no GPU.  
Latency budget: 50 ms per message on CPU.

| # | Model | Area | Macro-F1 | F1 (offensive) | Accuracy | Size (MB) | Latency (ms) | p95 (ms) | ≤ budget |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Lexicon + edit distance | B | 0.605 | 0.388 | 0.724 | 0.00 | 0.12 | 0.28 | yes |
| 2 | tf-idf + LogReg | B | 0.771 | 0.636 | 0.851 | 1.94 | 2.14 | 2.60 | yes |
| 3 | word2vec + FFN | A | 0.666 | 0.514 | 0.736 | 11.85 | 0.46 | 0.73 | yes |
| 4 | BERTurk FP32 | C | 0.816 | 0.709 | 0.877 | 422.00 | 137.19 | 229.27 | no |
| 5 | BERTurk INT8 (quantized) | C + D | 0.817 | 0.709 | 0.881 | 177.42 | 58.92 | 109.57 | no |

## Quantization trade-off (BERTurk FP32 → INT8)

* size: 422.0 MB → 177.4 MB (×2.38 smaller)
* latency: 137.19 ms → 58.92 ms (×2.33 faster)
* macro-F1: 0.816 → 0.817 (Δ = +0.001)
* FP32 and INT8 give the same label on 97.8% of the test messages

## Confusion matrices  [[TN, FP], [FN, TP]]

* Lexicon + edit distance: [[2245, 567], [407, 309]]
* tf-idf + LogReg: [[2545, 267], [258, 458]]
* word2vec + FFN: [[2104, 708], [224, 492]]
* BERTurk FP32: [[2567, 245], [188, 528]]
* BERTurk INT8 (quantized): [[2596, 216], [204, 512]]
