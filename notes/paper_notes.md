# Paper notes — Radford et al. (2018), "Improving Language Understanding by Generative Pre-Training"

Source: `paper/language_understanding_paper.pdf` (12 pages; text dump in `paper/paper_text.txt`).
Every item below was checked against the PDF. Section/table/page numbers refer to the PDF.
Items marked **[not in paper]** come from the released weights / reference implementation and are
labelled as such in the report.

## 1. Problem and core idea (§1, §3)
- Labelled data for NLU tasks is scarce, unlabelled text is abundant.
- Two-stage semi-supervised recipe: (1) generative pre-training of a Transformer language model
  on unlabelled text, (2) discriminative fine-tuning on each target task.
- Transfer with *minimal architecture changes*: "traversal-style" input transformations turn
  structured inputs (pairs, triples) into one token sequence (§3.3, ref [52]).
- Only new fine-tuning parameters: the output layer W_y and embeddings for delimiter/special tokens (§3.2).

## 2. Equations (§3)
- Eq. (1) LM objective: L1(U) = Σ_i log P(u_i | u_{i−k}, …, u_{i−1}; Θ), k = context window.
- Eq. (2) forward pass:
  - h0 = U·We + Wp
  - h_l = transformer_block(h_{l−1}) for l ∈ [1, n]
  - P(u) = softmax(h_n · We^T)  → output projection re-uses the token embedding (tied weights).
- Eq. (3) classifier: P(y | x¹…x^m) = softmax(h_l^m · W_y), h_l^m = final block activation at the last input token.
- Eq. (4) L2(C) = Σ_(x,y) log P(y | x¹…x^m).
- Eq. (5) L3(C) = L2(C) + λ · L1(C). Note L1 here is computed on the *task* corpus C.
- Aux LM stated benefits: (a) better generalisation, (b) faster convergence (§3.2).

## 3. Input transformations (§3.3, Fig. 1)
- All transformations add randomly initialised start and end tokens ⟨s⟩, ⟨e⟩ (the text calls ⟨e⟩
  the "end" token; its final hidden state is what is "extracted" for the classifier).
- Classification: ⟨s⟩ text ⟨e⟩ (fine-tuned directly).
- Entailment: ⟨s⟩ premise $ hypothesis ⟨e⟩.
- Similarity: no inherent order → both orderings ⟨s⟩A$B⟨e⟩ and ⟨s⟩B$A⟨e⟩, processed independently,
  the two h_l^m **added element-wise** before the linear layer.
- QA / commonsense (multiple choice): [z; q; $; a_k] for each answer, processed independently,
  softmax over the k scores.

## 4. Pre-training setup (§4.1)
- Data: BooksCorpus, >7,000 unique unpublished books; long contiguous text. (1B Word Benchmark is
  sentence-shuffled, so it lacks long-range structure.)
- LM perplexity on BooksCorpus: **18.4** (token level).
- Model: 12-layer decoder-only Transformer, masked self-attention, 768-dim states, 12 heads,
  FFN inner size 3072.
- Adam, max lr 2.5e-4, linear warm-up from 0 over the first 2,000 updates, then cosine annealing to 0.
- 100 epochs, minibatches of 64 randomly sampled contiguous sequences of 512 tokens.
- Init N(0, 0.02) ("since layernorm is used extensively").
- BPE vocabulary with 40,000 merges (ref [53], Sennrich et al.).
- Dropout 0.1 on residual, embedding and attention.
- Modified L2 regularisation (ref [37], Loshchilov & Hutter — decoupled weight decay) with w = 0.01 on
  all non-bias / non-gain weights.
- GELU activation (ref [18]); learned position embeddings instead of sinusoidal.
- Text cleaned with ftfy; spaCy tokenizer.

## 5. Fine-tuning setup (§4.1)
- Reuse pre-training hyper-parameters unless specified.
- Classifier dropout 0.1; lr 6.25e-5; batch size 32; 3 epochs sufficient for most cases;
  linear lr decay with warm-up over 0.2% of training; λ = 0.5.

## 6. Results relevant to this assignment (Table 4, GLUE; Table 5)
Table 4 (all evaluated with the GLUE benchmark):
| Method | CoLA mc | SST-2 acc | MRPC F1 | STS-B pc | QQP F1 | GLUE |
|---|---|---|---|---|---|---|
| Single-task BiLSTM+ELMo+Attn | 35.0 | 90.2 | 80.2 | 55.5 | 66.1 | 64.8 |
| Multi-task BiLSTM+ELMo+Attn | 18.9 | 91.6 | 83.5 | 72.8 | 63.3 | 68.9 |
| Sparse byte mLSTM | – | 93.2 | – | – | – | – |
| TF-KLD | – | – | 86.0 | – | – | – |
| **Finetuned Transformer LM (GPT)** | **45.4** | **91.3** | **82.3** | **82.0** | **70.3** | **72.8** |

Table 5 ablations (same GLUE tasks):
| Method | Avg | CoLA | SST2 | MRPC | STSB | QQP | MNLI | QNLI | RTE |
|---|---|---|---|---|---|---|---|---|---|
| Transformer w/ aux LM (full) | 74.7 | 45.4 | 91.3 | 82.3 | 82.0 | 70.3 | 81.8 | 88.1 | 56.0 |
| Transformer w/o pre-training | 59.9 | 18.9 | 84.0 | 79.4 | 30.9 | 65.5 | 75.7 | 71.2 | 53.8 |
| Transformer w/o aux LM | 75.0 | 47.9 | 92.0 | 84.9 | 83.2 | 69.8 | 81.1 | 86.9 | 54.4 |
| LSTM w/ aux LM | 69.1 | 30.3 | 90.5 | 83.2 | 71.8 | 68.1 | 73.7 | 81.1 | 54.6 |

Important observations from the paper:
- Aux LM helps NLI and QQP; "larger datasets benefit from the auxiliary objective but smaller datasets do not".
  On SST-2 and MRPC specifically, w/o aux LM is *higher* (92.0, 84.9) than the full model.
- No pre-training: −14.8 average vs full model; hurts every task.
- LSTM: −5.6 average; LSTM only beats Transformer on MRPC.
- 9 of 12 datasets are new state-of-the-art; GLUE 72.8 vs previous best 68.9.
- Other headline numbers: Story Cloze +8.9, RACE +5.7, MNLI +1.5 (abstract); RTE 56.0 (below 61.7 of multi-task biLSTM).

## 7. Analysis section (§5)
- **Layer transfer** (Fig. 2 left): MultiNLI and RACE vs number of layers transferred; transferring
  embeddings helps and each layer adds more, "up to 9% for full transfer on MultiNLI".
- **Zero-shot** (Fig. 2 right): heuristics using the LM only, tracked over pre-training updates;
  performance rises steadily; LSTM shows higher variance.
  - **SST-2 heuristic (exact): append the token "very" to each example, restrict the LM output
    distribution to the words "positive" and "negative", predict whichever gets higher probability.**
  - CoLA: average token log-prob, thresholded. RACE: answer with highest avg token log-prob.
    DPRD/Winograd: substitute candidate referent, compare avg log-prob of the rest.

## 8. Things commonly quoted about GPT-1 that are NOT in the paper
- **~117M parameters** — not stated. We compute the count from our implementation.
- **512-token context length** — the paper states 512-token training sequences; the released
  model's `n_positions = 512` [not in paper, HF config].
- **Post-LN** — not stated explicitly; the paper says the model "largely follows the original
  transformer work" [62], which is post-LN. Confirmed empirically by our weight-match test against
  the released weights (LayerNorm applied after each residual add).
- **Tied input/output embeddings** — implied by Eq. (2) (softmax(h_n We^T)).
- Figure 1's "Start / Delim / Extract" labels are in the figure image (text layer only says start/end tokens).
- "Test set" for Table 4: the paper says evaluations were "done using the GLUE benchmark" (i.e. the GLUE
  server). Our numbers are on the public dev (validation) split — not directly comparable.
