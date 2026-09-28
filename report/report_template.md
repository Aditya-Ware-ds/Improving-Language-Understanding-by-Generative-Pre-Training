---
title: "Generative Pre-Training of a Transformer Language Model (GPT-1): Study, Re-implementation and Application to Sentiment and Paraphrase Detection"
subtitle: "Deep Learning Assignment: Model Study Report"
author:
  - "Aditya Ware, Roll No. 26215011118 (Full Time), M.Tech (Artificial Intelligence)"
  - "Maulana Azad National Institute of Technology (MANIT), Bhopal"
date: "[Submission date]"
lang: en-GB
---

|   |   |
|---|---|
| **Course** | [Course name and code] |
| **Submitted to** | [Faculty name, Department of ...] |
| **Submitted by** | Aditya Ware, 26215011118, M.Tech-AI (Full Time) |
| **Assigned model** | GPT (GPT-1) |
| **Base paper** | A. Radford, K. Narasimhan, T. Salimans, I. Sutskever, *Improving Language Understanding by Generative Pre-Training*, OpenAI, 2018 |
| **Code** | `gpt1-assignment/` (PyTorch; see Appendix A) |

# Abstract {-}

GPT-1 (Radford et al., 2018) showed that a single Transformer decoder, first trained as a language model on
unlabelled books and then fine-tuned with a small task-specific output layer, can match or beat architectures
designed separately for each language-understanding task. This report studies the model and the paper in detail.
It covers the idea, the decoder architecture, the mathematics of masked self-attention and of the three training
objectives, and the optimisation recipe. It then describes a from-scratch PyTorch re-implementation. The
implementation loads the released pre-trained weights, and a numerical test confirms it reproduces the reference
implementation's hidden states (maximum absolute difference {{wm_maxdiff}}). We apply the model to two real-world
problems: sentiment analysis of reviews (SST-2), as a stand-in for opinion mining on customer feedback, and paraphrase
detection (MRPC). We use the paper's input transformations and its fine-tuning hyper-parameters on a 4 GB laptop GPU,
keeping the effective batch of 32 through gradient accumulation and mixed precision. Beyond the main runs we
reproduce the paper's ablations (no auxiliary language-model loss, no pre-training), its layer-transfer analysis
and its zero-shot sentiment heuristic, and we add a TF-IDF + logistic-regression baseline.
<!--AUTO:abstract_results--> All our numbers are on the public GLUE validation splits and are compared with the
paper's test-set numbers with the necessary caveats.

\newpage

# 1. Introduction and historical context

Natural-language-understanding (NLU) tasks such as sentiment classification, paraphrase detection, textual entailment
and question answering all need a model that "understands" text, but the labelled data for each task is small.
SST-2 has about 67 thousand labelled phrases and MRPC fewer than four thousand sentence pairs. Unlabelled text, by
contrast, is practically unlimited. The central question behind GPT-1 is how to turn cheap unlabelled text into
better performance on expensive labelled tasks. The history of that question explains why the paper mattered.

**Word vectors (2013–2014).** word2vec (Mikolov et al., 2013) and GloVe (Pennington et al., 2014) learn one
fixed vector per word from co-occurrence statistics. Using them to initialise task models became standard practice
and gave consistent gains. However, only the first layer is transferred, and a word gets the same vector in every
context ("bank" of a river and a "bank" account share one vector).

**Contextual representations from language models (2015–2018).** Dai & Le (2015) pre-trained an LSTM as a
language model or auto-encoder and fine-tuned it for classification. ELMo (Peters et al., 2018) used the internal
states of a bidirectional LSTM language model as *features* that are added to a task-specific architecture.
ULMFiT (Howard & Ruder, 2018) fine-tuned a whole pre-trained LSTM language model for text classification using
careful schedules (discriminative learning rates, gradual unfreezing). These methods transfer much more than
word vectors. They still rely on recurrent networks with a limited effective memory, and (for ELMo) on a new
architecture for every task.

**The Transformer (2017).** Vaswani et al. (2017) replaced recurrence by self-attention. Every position can
attend directly to every other position, training parallelises across the sequence, and long-range dependencies
do not have to pass through a chain of recurrent states.

**GPT-1 (2018).** Radford et al. combined the two threads. They pre-trained a 12-layer Transformer *decoder* as a
left-to-right language model on BooksCorpus, then fine-tuned *all* its parameters on each target task with only a
linear output layer added. Structured inputs (pairs, triples) are serialised into a single token sequence with
special delimiter tokens rather than handled by new architecture. The model improved the state of the art on 9 of
the 12 datasets studied (Radford et al., 2018, §1, §4.2).

**What came next.** BERT (Devlin et al., 2019) kept the pre-train/fine-tune recipe and the input serialisation
but replaced the left-to-right objective by masked-language modelling with a bidirectional encoder. GPT-2
(Radford et al., 2019; 1.5 B parameters, pre-LayerNorm blocks) and GPT-3 (Brown et al., 2020; 175 B parameters)
kept the decoder-only language model and scaled it. They showed that the zero-shot behaviour already visible in
the GPT-1 paper's §5 turns into strong few-shot "in-context" learning at scale. The decoder-only generative
pre-training introduced in GPT-1 is the direct ancestor of today's large language models.

# 2. Summary of the paper

**Problem.** Labelled data for NLU tasks is scarce, and earlier transfer methods either transferred only
word-level information or needed task-specific architectures on top of the transferred representation
(Radford et al., 2018, §1–2).

**Core idea.** Semi-supervised learning in two stages (§3):

1. *Unsupervised pre-training*: maximise the likelihood of a large unlabelled corpus with a multi-layer
   Transformer decoder language model.
2. *Supervised fine-tuning*: adapt all parameters to a labelled task with a single added linear layer,
   optionally keeping language modelling as an auxiliary objective.

Task-aware *input transformations* (§3.3) turn pairs and triples into one contiguous token sequence the pre-trained
model can read, so the architecture does not change between tasks.

**Main contributions.**

- A task-agnostic model that outperforms discriminatively trained, task-specific architectures on 9 of 12
  benchmarks, including absolute gains of 8.9% on Story Cloze, 5.7% on RACE and 1.5% on MultiNLI (Abstract), and
  a GLUE score of 72.8 against a previous best of 68.9 (§4.2, Table 4).
- Evidence that the Transformer transfers better than an LSTM trained in the same framework (average score
  −5.6 for the LSTM, Table 5), and that pre-training matters most: removing it costs 14.8% on average (Table 5).
- An analysis showing that every transferred layer helps (Fig. 2, left) and that the pre-trained language model
  performs several tasks zero-shot, with performance rising steadily during pre-training (Fig. 2, right).

**Headline numbers relevant to this report** (GLUE benchmark, Radford et al., 2018, Table 4): SST-2 accuracy
91.3, MRPC F1 82.3. The ablations in Table 5 give, for SST-2 / MRPC: without auxiliary LM 92.0 / 84.9, without
pre-training 84.0 / 79.4, LSTM 90.5 / 83.2.

# 3. Architecture

GPT-1 is a decoder-only Transformer: the Transformer's encoder–decoder cross-attention is removed and only masked
self-attention blocks remain (Radford et al., 2018, §3.1, citing Liu et al., 2018). The paper's specification
(§4.1) is:

| Component | Value in the paper | Our implementation (`src/model.py`) |
|---|---|---|
| Layers (Transformer blocks) | 12 | `n_layer = 12` |
| Hidden size $d_{model}$ | 768 | `n_embd = 768` |
| Attention heads | 12 (so $d_k = 64$) | `n_head = 12` |
| Feed-forward inner size | 3072 | `n_inner = 3072` |
| Activation | GELU | `nn.GELU(approximate="tanh")` (the tanh form used by the released model) |
| Position encoding | learned (not sinusoidal) | `nn.Embedding(512, 768)` |
| Training sequence length | 512 tokens | `n_positions = 512` |
| Vocabulary | BPE, 40,000 merges | 40,478 BPE tokens + 4 new special tokens |
| Output layer | tied to input embedding: $\text{softmax}(h_n W_e^\top)$ (Eq. 2) | `F.linear(h, W_e)` |
| Dropout | 0.1 on residual, embedding, attention | 0.1 in all three places |

Two details that are often quoted are **not** stated in the paper. (i) *Post-LayerNorm*: the paper says the model
"largely follows the original transformer work", which applies LayerNorm after each residual addition. GPT-2 later
moved LayerNorm before each sub-layer. We implement post-LN, and the exact match with the released weights
(Section 6.2) confirms it. (ii) *"117 M parameters"*: the paper gives no count. Counting from the architecture:
token embedding $40478 \times 768 = 31{,}087{,}104$, position embedding $512 \times 768 = 393{,}216$, and per block
$1{,}771{,}776$ (QKV projection) $+ 590{,}592$ (output projection) $+ 2{,}362{,}368 + 2{,}360{,}064$ (feed-forward)
$+ 3{,}072$ (two LayerNorms) $= 7{,}087{,}872$. Twelve blocks give $85{,}054{,}464$, for a total of
**116,534,784** parameters, the figure usually rounded to "117 M". Our weight-loading test reports
{{n_params_hf}} parameters for the reference model. Our classifier adds 4 special-token embeddings and the
$768 \times 2 + 2$ classification layer, giving {{n_params_clf}} parameters.

<!--FIG:architecture.png|Figure 1. GPT-1 architecture (left) and the task-specific input transformations (right). Drawn with matplotlib by src/make_report_assets.py after Radford et al. (2018), Fig. 1.-->

**Data flow.** A sequence of token ids is embedded and summed with position embeddings. It then passes through 12
identical blocks, each made of masked multi-head self-attention followed by a position-wise feed-forward network,
each with a residual connection and LayerNorm. The final hidden states $h_n$ feed two heads. The *language-model
head* projects every position back onto the vocabulary through the transposed token-embedding matrix; the
*classification head* $W_y$ reads only the hidden state at the final ⟨e⟩ (extract) token. Because attention is
causal, the state at ⟨e⟩ is the only one that has seen the entire input, which is why it is used for
classification.

**Task-specific input transformations** (§3.3, Fig. 1). All inputs receive randomly initialised start ⟨s⟩ and end
(extract) ⟨e⟩ tokens:

- *Classification* (SST-2): `⟨s⟩ text ⟨e⟩`.
- *Entailment*: `⟨s⟩ premise ⟨$⟩ hypothesis ⟨e⟩`.
- *Similarity* (MRPC): the two sentences have no natural order, so both orderings `⟨s⟩ A ⟨$⟩ B ⟨e⟩` and
  `⟨s⟩ B ⟨$⟩ A ⟨e⟩` are processed independently and their final ⟨e⟩ states are **added element-wise** before
  $W_y$.
- *Multiple choice* (RACE, Story Cloze): one sequence `⟨s⟩ context ⟨$⟩ answer_k ⟨e⟩` per candidate, each scored
  by the linear layer, with a softmax over candidates.

We implement classification and similarity, the two used in our applications. In our tokenizer the delimiter is
a new token `<$>` rather than the character "\$", because "\$" is already an ordinary BPE token (id 289) that can
appear in review text.

# 4. Mathematical formulation

## 4.1 Byte-pair encoding

Text is first cleaned with *ftfy* and split into words with spaCy, then segmented into sub-word units by byte-pair
encoding (BPE; Sennrich et al., 2016), as in §4.1 of the paper. BPE starts from characters and repeatedly merges
the most frequent adjacent pair of symbols. After 40,000 merges, frequent words are single tokens (`very</w>`,
`positive</w>`), while rare words split into several pieces, so there is no out-of-vocabulary problem. Our runs
used the tokenizer in the mode: *{{tok_mode}}*.

## 4.2 Embeddings

For a context of $T$ token ids $U = (u_1, \dots, u_T)$, written as one-hot rows of a $T \times |V|$ matrix, the
input to the first block is (Eq. 2 of the paper)
$$h_0 = U W_e + W_p,$$
where $W_e \in \mathbb{R}^{|V| \times 768}$ is the token-embedding matrix and $W_p \in \mathbb{R}^{T \times 768}$
holds the first $T$ rows of the learned position-embedding table. Learned rather than sinusoidal position
embeddings are used (§4.1), so the model cannot represent positions beyond 512.

## 4.3 Scaled dot-product attention with a causal mask

Inside a block, the input $X \in \mathbb{R}^{T \times d}$ is projected to queries, keys and values,
$Q = XW^Q$, $K = XW^K$, $V = XW^V$, and attention is
$$\operatorname{Attention}(Q,K,V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V,
\qquad M_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i. \end{cases}$$
Row $i$ of the softmax is a probability distribution over the positions position $i$ may look at. The mask $M$
removes all future positions, so the prediction for position $i+1$ can only use tokens $1..i$. This is what makes
the network a valid left-to-right language model. The factor $1/\sqrt{d_k}$ keeps the dot products at unit
scale: if the components of $q$ and $k$ are independent with variance 1, then $q^\top k$ has variance $d_k$, and
without the scaling a large $d_k$ would push the softmax into saturated regions with vanishing gradients.

A useful consequence for fine-tuning: with right-padding, real tokens never attend to the padding that follows
them, so no separate padding mask is needed (Section 6.3).

## 4.4 Multi-head attention

With $h = 12$ heads of size $d_k = d/h = 64$,
$$\operatorname{MultiHead}(X) = \operatorname{Concat}(\text{head}_1, \dots, \text{head}_h)\,W^O, \qquad
\text{head}_j = \operatorname{Attention}(XW^Q_j, XW^K_j, XW^V_j).$$
Each head can specialise in a different relation (for example the previous token, a syntactic head word, or
sentence boundaries) at the same total cost as one full-width head. In the implementation the three projections
of all heads are fused into one $768 \times 2304$ matrix (`c_attn`).

## 4.5 Position-wise feed-forward network and GELU

$$\operatorname{FFN}(x) = \operatorname{GELU}(xW_1 + b_1)W_2 + b_2, \qquad W_1 \in \mathbb{R}^{768\times 3072},\;
W_2 \in \mathbb{R}^{3072 \times 768},$$
applied to every position independently. The Gaussian Error Linear Unit (Hendrycks & Gimpel, 2016) is
$$\operatorname{GELU}(x) = x\,\Phi(x) \approx 0.5x\left(1 + \tanh\!\left[\sqrt{2/\pi}\,(x + 0.044715x^3)\right]\right),$$
where $\Phi$ is the standard normal CDF. Unlike ReLU, it is smooth and weights its input by the probability that
a standard Gaussian falls below it. The released model uses the tanh approximation (the original OpenAI code, and
the Hugging Face port, which maps this model's `gelu` setting to the tanh form), so our implementation uses it as
well. The two forms differ by less than $10^{-3}$ per activation, but this difference compounds over 12 layers.
Our first weight-match attempt used the exact erf form and failed, with maximum hidden-state differences between
$1.1 \times 10^{-2}$ (short sentences) and $2.0 \times 10^{-1}$ (512 random tokens); switching to the tanh form is what the released weights require.

## 4.6 Residual connections and LayerNorm (post-LN block)

LayerNorm (Ba et al., 2016) normalises each position's feature vector:
$$\operatorname{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta, \qquad
\mu = \tfrac{1}{d}\textstyle\sum_i x_i,\; \sigma^2 = \tfrac{1}{d}\sum_i (x_i - \mu)^2,$$
with learned gain $\gamma$ and bias $\beta$ ($\epsilon = 10^{-5}$). A GPT-1 block is
$$n = \operatorname{LN}_1\big(x + \operatorname{MultiHead}(x)\big), \qquad h = \operatorname{LN}_2\big(n + \operatorname{FFN}(n)\big),$$
and $h_l = \text{block}_l(h_{l-1})$ for $l = 1..12$ (Eq. 2). Because the paper uses LayerNorm "extensively", a
simple $\mathcal{N}(0, 0.02^2)$ initialisation was enough (§4.1).

## 4.7 Output distribution, softmax and cross-entropy

The next-token distribution uses the transposed embedding matrix (weight tying, Eq. 2):
$$P(u) = \operatorname{softmax}(h_n W_e^\top), \qquad \operatorname{softmax}(z)_k = \frac{e^{z_k}}{\sum_j e^{z_j}}.$$
Maximising the log-probability of the correct token $y$ is the same as minimising the cross-entropy
$\ell = -\log \operatorname{softmax}(z)_y = -z_y + \log\sum_j e^{z_j}$. Its gradient with respect to the logits
is simply $\operatorname{softmax}(z) - \text{onehot}(y)$.

## 4.8 The three objectives

**Pre-training** (Eq. 1). For an unlabelled corpus $\mathcal{U} = \{u_1, \dots, u_n\}$ and context size $k$:
$$L_1(\mathcal{U}) = \sum_i \log P(u_i \mid u_{i-k}, \dots, u_{i-1}; \Theta).$$

**Supervised fine-tuning** (Eqs. 3–4). For a labelled dataset $\mathcal{C}$ of token sequences $x^1..x^m$ with
label $y$, the final block's state at the last (⟨e⟩) position, $h_l^m$, feeds a new linear layer:
$$P(y \mid x^1,\dots,x^m) = \operatorname{softmax}(h_l^m W_y), \qquad
L_2(\mathcal{C}) = \sum_{(x,y)} \log P(y \mid x^1, \dots, x^m).$$
For similarity tasks $h_l^m$ is replaced by $h_l^m(A;B) + h_l^m(B;A)$.

**Fine-tuning with an auxiliary LM objective** (Eq. 5):
$$L_3(\mathcal{C}) = L_2(\mathcal{C}) + \lambda \cdot L_1(\mathcal{C}), \qquad \lambda = 0.5.$$
Note that $L_1$ is evaluated here on the *task* data $\mathcal{C}$ (the ⟨s⟩ … ⟨$⟩ … ⟨e⟩ sequences), not on
BooksCorpus. The paper motivates it by better generalisation and faster convergence (§3.2). In code we minimise
the negatives, averaged rather than summed:
`loss = CE(classifier) + λ · CE(next token over all non-pad positions)`. The only new parameters during
fine-tuning are $W_y$ and the embeddings of the special tokens.

## 4.9 Perplexity

The LM loss per token is an average negative log-likelihood; its exponential is the perplexity,
$$\operatorname{PPL} = \exp\!\Big(-\tfrac{1}{N}\textstyle\sum_{i=1}^{N} \log P(u_i \mid u_{<i})\Big),$$
the effective number of equally likely choices per token. The paper reports a token-level perplexity of 18.4 on
BooksCorpus (§4.1). The auxiliary-loss curves in Section 8 give our per-token LM cross-entropy on the (very
different) SST-2 and MRPC text, whose exponential is the perplexity on that data.

# 5. Training and optimisation

## 5.1 Pre-training (paper, not repeated here)

| Setting | Value (Radford et al., 2018, §4.1) |
|---|---|
| Corpus | BooksCorpus: over 7,000 unpublished books, long contiguous text |
| Batch | 64 randomly sampled contiguous sequences of 512 tokens |
| Epochs | 100 |
| Optimiser | Adam, max learning rate $2.5 \times 10^{-4}$ |
| Schedule | linear warm-up from 0 over the first 2,000 updates, then cosine annealing to 0 |
| Regularisation | dropout 0.1 (residual, embedding, attention); modified L2 with $w = 0.01$ on all non-bias, non-gain weights |
| Initialisation | $\mathcal{N}(0, 0.02)$ |
| Result | token-level perplexity 18.4 |

BooksCorpus was chosen over the similar-sized 1B Word Benchmark because the latter is shuffled at the sentence
level, which destroys the long-range structure the model is meant to learn (§4.1). Pre-training at this scale is far
beyond a 4 GB laptop GPU, so, as the assignment allows, we start from the released weights.

## 5.2 Fine-tuning (paper settings used as our defaults)

| Setting | Paper (§4.1) | Ours |
|---|---|---|
| Learning rate | $6.25 \times 10^{-5}$ | same |
| Batch size | 32 | 32 effective = {{micro_batch}} (micro-batch) × {{grad_accum}} (accumulation) |
| Epochs | 3 | 3 |
| Schedule | linear decay, warm-up over 0.2% of training | same (warm-up steps = round(0.002 × total)) |
| $\lambda$ | 0.5 | 0.5 (E1), 0 (E2) |
| Classifier dropout | 0.1 | 0.1 |
| Weight decay | modified L2, $w = 0.01$ | AdamW decoupled decay 0.01 on all ≥2-D weights |
| Gradient clipping | not stated | global norm 1.0 |
| Precision | not stated | fp16 autocast + dynamic loss scaling |

## 5.3 Adam and decoupled weight decay

Adam (Kingma & Ba, 2015) keeps running averages of the gradient and of its square,
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t,\quad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2,\quad
\hat m_t = \frac{m_t}{1-\beta_1^t},\; \hat v_t = \frac{v_t}{1-\beta_2^t},$$
$$\theta_t = \theta_{t-1} - \eta_t\left(\frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon} + w\,\theta_{t-1}\right).$$
The last term is the "modified L2 regularisation" the paper cites (Loshchilov & Hutter, 2017; ref. [37]). The decay
is applied directly to the weights instead of being added to the gradient, where Adam's per-parameter scaling would
weaken it for parameters with large gradients. We use $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$,
$w = 0.01$, and exclude biases and LayerNorm gains, as the paper specifies.

## 5.4 Learning-rate schedules

With $S$ total updates and $S_w$ warm-up updates, pre-training used
$\eta_t = \eta_{\max}\, t/S_w$ for $t < S_w$ and
$\eta_t = \tfrac{1}{2}\eta_{\max}\big(1 + \cos(\pi \tfrac{t - S_w}{S - S_w})\big)$ afterwards, while fine-tuning uses
the linear version, $\eta_t = \eta_{\max}\,\frac{S - t}{S - S_w}$ for $t \ge S_w$. Warm-up matters for Adam,
because $v_t$ is estimated from very few gradients in the first steps, and for post-LN Transformers, whose
gradients near the output are large at initialisation. Decaying to zero ends training at a flat point of the loss.

## 5.5 Dropout

Dropout zeroes each activation with probability $p = 0.1$ during training and rescales the rest by $1/(1-p)$. It
is applied to the summed embeddings, to the attention probabilities, to each sub-layer's output before the residual
addition, and (in fine-tuning) to the ⟨e⟩ state before $W_y$.

## 5.6 Fitting fine-tuning into 4 GB of GPU memory

The released model in fp32 needs about 0.47 GB for weights, plus the same again for gradients and twice that for
Adam's two moment estimates, before any activations. We therefore used:

- **Gradient accumulation.** Gradients of {{grad_accum}} micro-batches of {{micro_batch}} examples are summed
  before each optimiser step, so the effective batch is 32, as in the paper. The loss of each micro-batch is divided
  by the number of accumulation steps.
- **Mixed precision.** Matrix products run in fp16 under `torch.autocast`, with a `GradScaler` to prevent fp16
  gradient underflow. Master weights and optimiser states stay in fp32.
- **Dynamic padding.** Each micro-batch is padded only to its own longest sequence. The length analysis
  (Table 7.1) guided `max_len`, which only guards against truncation.
- **Memory-aware auxiliary loss.** The LM head produces 40,482 logits per position, the largest tensor in the
  model. We first gather the hidden states of the non-pad positions and only then project them to the vocabulary,
  so pad positions never produce logits.
- **Automatic OOM recovery.** If CUDA runs out of memory, the run restarts with half the micro-batch and twice the
  accumulation, keeping the effective batch at 32, and the event is logged. Across all runs there were
  {{oom_total}} such events. Gradient checkpointing is available (`--grad-checkpoint`) but was not needed.

# 6. Implementation

## 6.1 Code structure

| File | Purpose |
|---|---|
| `src/model.py` | GPT-1 written from scratch in PyTorch: embeddings, causal multi-head attention (fused SDPA kernel, plus an explicit path that returns attention maps), post-LN block, GELU MLP, tied LM head, classifier on the ⟨e⟩ state with element-wise sum for pairs. No `transformers` model classes. |
| `src/load_pretrained.py` | Maps `openai-community/openai-gpt` tensors into our modules (HF `Conv1D` weights are transposed relative to `nn.Linear`) and contains the weight-match test. |
| `src/data.py` | GLUE loading, BPE tokenisation (cached), the paper's input transformations, special tokens, dynamic right-padding, the LM-loss mask, and the length analysis. |
| `src/train.py` | Fine-tuning with $L_3 = L_2 + \lambda L_1$; paper hyper-parameters as defaults; flags `--lambda --lr --epochs --max-len --seed --subset --no-pretrain --transfer-layers k --grad-checkpoint --quick --time-steps`; fp16, accumulation, OOM fallback, progress bars, curves, peak-VRAM and timing logging. |
| `src/evaluate.py` | Accuracy, precision, recall, macro-F1, positive-class F1, confusion matrix; plotting. |
| `src/baselines.py` | TF-IDF + logistic regression (E4). |
| `src/zero_shot.py` | The paper's zero-shot SST-2 heuristic (E5). |
| `src/predict.py` | Classify custom text with a fine-tuned checkpoint. |
| `src/run_all.py` | Resumable driver for the smoke tests and the whole experiment suite. |
| `src/make_report_assets.py`, `src/build_report.py` | Tables and figures from `results/`, then this report. |
| `notebooks/GPT1_Colab.ipynb` | The same pipeline end to end on a Colab T4. |

## 6.2 Verifying the port of the pre-trained weights

A re-implementation is only useful if it computes exactly the same function as the released model.
`src/load_pretrained.py --test` feeds identical token ids to our model and to Hugging Face's `OpenAIGPTModel` and
compares the final hidden states. It uses two real sentences, a random batch of 2 × 512 ids that exercises the
full context, and the tied LM-head logits against `OpenAIGPTLMHeadModel`. The test requires a maximum absolute
difference below $10^{-4}$. Status: **{{wm_status}}**; largest difference over all cases: **{{wm_maxdiff}}**. This
also confirms the post-LN block order, the exact GELU variant, the attention scaling and the weight tying
empirically.

## 6.3 Fine-tuning details worth noting

- The four new tokens `<s>`, `<$>`, `<e>`, `<pad>` get ids 40478–40481. The embedding matrix is resized and the
  new rows are initialised from $\mathcal{N}(0, 0.02)$. Because the LM head is tied, these tokens can also be
  predicted.
- Sequences are right-padded. Causal attention means the state at ⟨e⟩ never sees the pads after it, so no
  attention mask is needed, and the classifier reads position `len − 1` of each sequence.
- The auxiliary LM loss predicts every next token of the transformed sequence, including the delimiter and ⟨e⟩,
  and excludes pad targets.
- For E3 (no pre-training), the same architecture and hyper-parameters are used with a fresh
  $\mathcal{N}(0, 0.02)$ initialisation. For E6 we keep the embeddings and the first $k$ pre-trained blocks and
  re-initialise the rest.
- We report the model after the last epoch. No checkpoint is chosen by validation score, because the validation
  set is also our test set.

## 6.4 Hardware and software

GPU: {{gpu}}. Software: PyTorch {{torch_version}}, transformers {{tf_version}} (used only to download weights, to
tokenise and as the reference in the weight test). Platform: {{platform}}. Total GPU fine-tuning time over all
runs: {{gpu_hours}} h; highest peak allocated GPU memory in any run: {{peak_vram_max}} MiB.

<!--TABLE:compute-->

# 7. Datasets and application

**Primary application: opinion mining on customer feedback.** Companies receive large volumes of free-text reviews
and survey answers. Automatically labelling each one as positive or negative lets them track satisfaction over
time and route complaints. We use the **Stanford Sentiment Treebank, binary version (SST-2)** from GLUE (Socher et
al., 2013; Wang et al., 2018), which consists of movie-review sentences. The training split has
{{sst2_n_train}} examples, many of them short phrases taken from the parse trees (mean {{sst2_len_train_mean}} BPE
tokens after transformation). The validation split has {{sst2_n_val}} full sentences (mean
{{sst2_len_val_mean}} tokens). GLUE's test labels are hidden, so **all our numbers are on the validation split**.
`src/predict.py` applies the fine-tuned model to arbitrary review text.

**Secondary application: paraphrase / duplicate detection.** Detecting that two texts say the same thing is used
for de-duplicating support tickets or FAQ questions and for clustering news. We use the **Microsoft Research
Paraphrase Corpus (MRPC)** from GLUE (Dolan & Brockett, 2005): {{mrpc_n_train}} training and {{mrpc_n_val}}
validation sentence pairs from news sources, labelled equivalent / not equivalent. The positive class is the
majority (roughly two thirds). Following GLUE and the paper, we report F1 of the positive class, plus accuracy and
macro-F1. We apply the paper's symmetric similarity transformation.

**Sequence lengths.** To choose `max_len` we measured the transformed lengths (Figure 2, Table 7.1). SST-2
sequences are at most {{sst2_len_train_max}} tokens long and MRPC pair sequences at most {{mrpc_len_train_max}},
so we set `max_len` = {{sst2_maxlen}} and {{mrpc_maxlen}} respectively. No example is truncated, and with
dynamic padding the limit costs nothing.

**Table 7.1.** Transformed sequence lengths.

<!--TABLE:lengths-->

<!--FIG:lengths.png|Figure 2. Length distributions of the transformed training sequences; the dashed line is the chosen max_len.-->

**Experiments.** All use seed 42 unless stated otherwise.

| ID | Description |
|---|---|
| E1 | Pre-trained GPT-1 fine-tuned with $\lambda = 0.5$ (the paper's full method) |
| E2 | Pre-trained, $\lambda = 0$ (no auxiliary LM objective) |
| E3 | Same architecture, random initialisation (no pre-training) |
| E4 | TF-IDF (word 1–2-grams + character 2–5-grams) + logistic regression, $C$ chosen by cross-validation on the training set |
| E5 | Zero-shot SST-2 with the paper's LM heuristic (append "very", compare P(positive) and P(negative)) |
| E6 | Layer transfer on MRPC: embeddings + first $k \in \{0, 3, 6, 9, 12\}$ pre-trained blocks |

MRPC E1 and E2 are repeated over seeds {{mrpc_seeds}} and reported as mean ± sample standard deviation.

# 8. Results

## 8.1 Sentiment analysis (SST-2)

**Table 8.1.** SST-2 validation results.

<!--TABLE:main_sst2-->

<!--AUTO:findings_sst2-->

<!--FIG:comparison.png|Figure 3. All systems on both tasks; the dashed line is the paper's GLUE test-set result for the full model.-->

<!--FIG:cm_E1_sst2_seed42.png|Figure 4. Confusion matrix of E1 on SST-2 validation.-->

## 8.2 Paraphrase detection (MRPC)

**Table 8.2.** MRPC validation results.

<!--TABLE:main_mrpc-->

<!--AUTO:findings_mrpc-->

<!--FIG:cm_E1_mrpc_seed42.png|Figure 5. Confusion matrix of E1 (seed 42) on MRPC validation.-->

## 8.3 Training curves

<!--FIG:curves_E1_sst2_seed42.png|Figure 6. E1 on SST-2: classification loss, accuracy and auxiliary LM loss.-->

<!--FIG:curves_E2_sst2_seed42.png|Figure 7. E2 on SST-2 (lambda = 0; no LM loss panel).-->

<!--FIG:curves_E3_sst2_seed42.png|Figure 8. E3 on SST-2 (random initialisation).-->

<!--FIG:curves_E1_mrpc_seed42.png|Figure 9. E1 on MRPC (seed 42).-->

<!--AUTO:curves-->

## 8.4 Layer-transfer analysis (E6)

**Table 8.3.** Effect of the number of transferred pre-trained blocks on MRPC.

<!--TABLE:layer_transfer_mrpc-->

<!--FIG:layer_transfer_mrpc.png|Figure 10. MRPC validation score as a function of the number of pre-trained blocks transferred.-->

<!--AUTO:layers-->

## 8.5 Zero-shot sentiment (E5)

**Table 8.4.** Zero-shot SST-2 with the paper's heuristic (§5 of the paper).

<!--TABLE:zero_shot_sst2-->

<!--FIG:zero_shot_logodds.png|Figure 11. Zero-shot log-odds of "positive" versus "negative" after appending "very", split by true label.-->

<!--AUTO:zeroshot-->

## 8.6 Error analysis

**Table 8.5.** The most confident mistakes of E1 on SST-2. The last column is an automatic heuristic (negation
words, contrast connectives, length), not a human judgement.

<!--TABLE:misclassified_sst2-->

<!--AUTO:misclassified-->

The last column of Table 8.5 gives an automatic first diagnosis of each error. The categories are the usual
failure modes of sentence-level sentiment models:

1. **Contrast and concession.** Sentences of the form "X, but Y" require weighing two clauses of opposite
   polarity. The model has to decide which clause carries the reviewer's overall opinion, and SST-2 annotators
   usually follow the clause after "but".
2. **Negation and hedging.** "not bad", "hardly a masterpiece" and "never boring" flip or soften the polarity of
   the word they modify. Errors appear when the sentiment-bearing word is strong and the negation is several tokens
   away.
3. **Irony, faint praise and world knowledge.** Phrases that are positive on the surface but negative in intent
   (or the reverse) need knowledge of film-review conventions that neither the phrase-level training data nor the
   BooksCorpus pre-training covers reliably.
4. **Label noise and genuinely mixed sentences.** Some "errors" are sentences whose gold label is itself debatable.
   SST-2 labels come from averaging several annotators' sentiment scores and then binarising.

## 8.7 Attention visualisation

<!--FIG:attention_sst2.png|Figure 12. Attention in the fine-tuned SST-2 model (E1). Left: last layer, average over heads. Right: attention of the extract token ⟨e⟩ in every layer.-->

<!--AUTO:attention-->

# 9. Comparison and discussion

**Table 9.1.** Our validation results next to the paper's test results.

<!--TABLE:paper_comparison-->

**How close are we to the paper?** Several differences make an exact comparison impossible, so the table should be
read as "same ballpark or not":

- *Split.* The paper reports the hidden GLUE **test** sets. We report the public **validation** sets, which have a
  different size and composition (SST-2 validation has 872 sentences, so one example is 0.11 points of accuracy;
  MRPC validation has 408 pairs).
- *Variance.* Fine-tuning runs of this size vary by around a point between seeds, more on MRPC. The paper reports
  single numbers; we report means over the seeds we ran.
- *Pre-processing and tokenisation.* We use the Hugging Face port of the original ftfy + spaCy + BPE pipeline.
  Small differences in text cleaning are possible.
- *Implementation details not in the paper.* Gradient clipping, fp16 training, exact warm-up rounding, the
  treatment of special tokens in the LM loss, and the choice of the final-epoch model are our choices.
- *Per-task tuning.* The paper says it used $6.25 \times 10^{-5}$ and batch 32 "for most tasks", so a few tasks may
  have used other settings. We used the stated defaults everywhere.

**Pre-training.** In the paper (Table 5), removing pre-training causes the largest drop of any ablation
(−14.8 on average). The mechanism is straightforward. A 117 M-parameter network fine-tuned from scratch on a few
thousand or tens of thousands of labelled examples has far more parameters than the labels can constrain, and it
cannot learn syntax, negation or world knowledge from the labels alone, whereas the pre-trained network brings them
from BooksCorpus. <!--AUTO:pretrain_discussion-->

**The auxiliary LM objective is not a free win.** The paper introduces $L_3$ because it "improved generalization"
and "accelerated convergence", but its own ablation (Table 5) shows the auxiliary loss helps the large NLI datasets
and QQP and *hurts* slightly on SST-2 (92.0 vs 91.3) and MRPC (84.9 vs 82.3). The authors conclude that "larger
datasets benefit from the auxiliary objective but smaller datasets do not". Our E1-vs-E2 comparison tests exactly
this on the two datasets we used: <!--AUTO:aux_discussion--> There is also a plausible mechanism for why the auxiliary loss matters less here. With only three
epochs and a small learning rate, fine-tuning moves the weights little, so the knowledge that the LM loss is
meant to protect is not forgotten much anyway. Its benefit as a regulariser is therefore small, while it spends
part of the gradient on modelling the (short, domain-specific) task text.

**Transformers against a classical baseline.** TF-IDF + logistic regression is fast, CPU-only and
interpretable. It sees only which n-grams occur, not how they combine, so beyond bigrams it cannot model negation
scope or contrast ("not really good", "good, but"), which is where a contextual model is expected to help.
<!--AUTO:baseline_discussion--> On MRPC the classical model has to rely on lexical overlap features; paraphrases with little word
overlap and non-paraphrases with high overlap are its typical errors.

**Zero-shot behaviour.** The paper claims that the generative model learns task-relevant knowledge, such as
sentiment, from books alone (§5, Fig. 2 right). <!--AUTO:zeroshot_discussion--> GPT-2 and GPT-3 later showed that
the zero-shot gap to fine-tuned models shrinks as the pre-trained model grows.

**Practical cost.** Fine-tuning a 117 M-parameter model on a 4 GB laptop GPU is feasible with gradient
accumulation, mixed precision, dynamic padding and a memory-aware LM loss. The effective batch size and all
optimisation hyper-parameters remain those of the paper, so these engineering changes affect speed and memory, not
the method.

# 10. Limitations and observations

- **Validation instead of test.** GLUE test labels are hidden; submitting to the GLUE server was out of scope.
  Our validation numbers are not a like-for-like comparison with the paper's test numbers.
- **Seeds.** SST-2 runs use a single seed because of their length on a laptop GPU (see the compute table in Section 6.4); MRPC main runs
  use {{mrpc_seeds}}. Differences of about one point should not be over-interpreted.
- **No pre-training of our own.** Pre-training on BooksCorpus (100 epochs on 64 × 512-token batches) is
  thousands of GPU-days beyond this hardware. We verified that our architecture reproduces the released model
  exactly and started from its weights.
- **LSTM ablation not reproduced.** The paper's fourth ablation (a 2048-unit LSTM in the same framework) would need
  an LSTM pre-trained on BooksCorpus, which is not publicly available.
- **Only two of the four input transformations exercised.** Entailment and multiple-choice formatting are described
  and are one-line variants of the implemented ones, but were not run.
- **Hyper-parameters not re-tuned.** We used the paper's defaults everywhere, including for E3. A randomly
  initialised Transformer would likely prefer a larger learning rate and more epochs, so E3 measures "the paper's
  recipe without pre-training", not the best achievable model trained from scratch.
- **Domain.** SST-2 consists of movie reviews and MRPC of news. For real customer-feedback mining, the model should
  be validated (and ideally fine-tuned further) on in-domain labelled data such as product or service reviews.
- **Observation: GPU thermals.** The laptop GPU ran at 80–86 °C even when idle, so long runs may be slowed by
  thermal throttling. This affects the timing figures, not the results.

# 11. Conclusion

We studied GPT-1 from the original paper and re-implemented it from scratch in PyTorch: a 12-layer, 768-wide,
12-head post-LN Transformer decoder with learned positions, GELU and a tied output layer. We verified that it
reproduces the released weights' computation to within {{wm_maxdiff}}. We fine-tuned it with the paper's objective
$L_3 = L_2 + \lambda L_1$ and input transformations on a 4 GB GPU without changing the effective batch or
hyper-parameters. Checked against the paper's four qualitative conclusions, our sentiment and paraphrase
experiments give the following picture:

<!--AUTO:conclusion-->

The recipe of pre-training a large
Transformer language model on unlabelled text and adapting it with minimal task-specific machinery has since become
the foundation of modern NLP, and this assignment shows it working end to end on modest hardware.

# References {-}

1. A. Radford, K. Narasimhan, T. Salimans, I. Sutskever. *Improving Language Understanding by Generative
   Pre-Training.* OpenAI technical report, 2018.
   https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
2. A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, I. Polosukhin. Attention is all
   you need. *NeurIPS*, 2017.
3. P. J. Liu, M. Saleh, E. Pot, B. Goodrich, R. Sepassi, Ł. Kaiser, N. Shazeer. Generating Wikipedia by summarizing
   long sequences. *ICLR*, 2018.
4. T. Mikolov, I. Sutskever, K. Chen, G. Corrado, J. Dean. Distributed representations of words and phrases and
   their compositionality. *NeurIPS*, 2013.
5. J. Pennington, R. Socher, C. D. Manning. GloVe: Global vectors for word representation. *EMNLP*, 2014.
6. A. M. Dai, Q. V. Le. Semi-supervised sequence learning. *NeurIPS*, 2015.
7. M. E. Peters et al. Deep contextualized word representations (ELMo). *NAACL*, 2018.
8. J. Howard, S. Ruder. Universal language model fine-tuning for text classification (ULMFiT). *ACL*, 2018.
9. J. Devlin, M.-W. Chang, K. Lee, K. Toutanova. BERT: Pre-training of deep bidirectional Transformers for language
   understanding. *NAACL*, 2019.
10. A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever. Language models are unsupervised multitask
    learners (GPT-2). OpenAI, 2019.
11. T. B. Brown et al. Language models are few-shot learners (GPT-3). *NeurIPS*, 2020.
12. R. Sennrich, B. Haddow, A. Birch. Neural machine translation of rare words with subword units. *ACL*, 2016.
13. D. Hendrycks, K. Gimpel. Gaussian error linear units (GELUs). arXiv:1606.08415, 2016.
14. J. L. Ba, J. R. Kiros, G. E. Hinton. Layer normalization. arXiv:1607.06450, 2016.
15. D. P. Kingma, J. Ba. Adam: A method for stochastic optimization. *ICLR*, 2015.
16. I. Loshchilov, F. Hutter. Fixing weight decay regularization in Adam (Decoupled weight decay regularization).
    arXiv:1711.05101, 2017.
17. Y. Zhu et al. Aligning books and movies: Towards story-like visual explanations by watching movies and reading
    books (BooksCorpus). *ICCV*, 2015.
18. A. Wang, A. Singh, J. Michael, F. Hill, O. Levy, S. R. Bowman. GLUE: A multi-task benchmark and analysis platform
    for natural language understanding. arXiv:1804.07461, 2018.
19. R. Socher et al. Recursive deep models for semantic compositionality over a sentiment treebank (SST). *EMNLP*,
    2013.
20. W. B. Dolan, C. Brockett. Automatically constructing a corpus of sentential paraphrases (MRPC). *IWP*, 2005.
21. Software: PyTorch (Paszke et al., 2019); Hugging Face Transformers (Wolf et al., 2020) and Datasets (Lhoest et al.,
    2021), model card `openai-community/openai-gpt`; scikit-learn (Pedregosa et al., 2011); spaCy; ftfy; matplotlib.

# Appendix A. How to reproduce {-}

The full instructions, with expected runtimes, are in `README.md`. In short:

```powershell
# 1. environment (Windows PowerShell, Python 3.12, NVIDIA GPU)
py -3.12 -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt; python -m spacy download en_core_web_sm
powershell -ExecutionPolicy Bypass -File scripts\download_weights.ps1   # optional: resumable weight download

# 2. smoke tests of every script (100 examples, 1 epoch)
python src\run_all.py --smoke

# 3. all experiments E1-E6, then tables, figures and this report (resumable)
python src\run_all.py

# 4. rebuild only the report from results\
python src\make_report_assets.py --attention; python src\build_report.py
```

Every table and figure in Sections 7–9 is generated from `results/*.json` by `src/make_report_assets.py`, and every
number in the text by `src/build_report.py`. A run that has not been done appears as "*not run*" rather than as a
value.

# Appendix B. Fine-tuning curves and confusion matrices for other runs {-}

<!--FIG:cm_E4_tfidf_lr_sst2.png|Figure B1. Confusion matrix of the TF-IDF + LR baseline on SST-2.-->

<!--FIG:cm_E5_zeroshot_sst2.png|Figure B2. Confusion matrix of the zero-shot heuristic on SST-2.-->

<!--FIG:curves_E2_mrpc_seed42.png|Figure B3. E2 on MRPC (seed 42).-->

<!--FIG:curves_E3_mrpc_seed42.png|Figure B4. E3 on MRPC (random initialisation).-->
