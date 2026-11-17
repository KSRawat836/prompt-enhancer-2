# Prompt Enhancer (Deep Learning mini project)

Takes a rough prompt, detects **what it is missing**, and rewrites it into a clearer one.
Built with an **RNN (BiLSTM + attention)** in PyTorch. **No LLM and no API calls.**

```
raw prompt
   |
   v
[BiLSTM + attention]  -> task type (6 classes)  +  missing parts (5 yes/no labels)
   |
   v
[rule safety net]     -> corrects "missing" calls when a limit / format is clearly present
   |
   v
[templates]           -> rewritten prompt + list of placeholders for the user to fill
```

## Quick start

```bash
pip install -r requirements.txt

python -m src.generate_dataset   # builds data/train|val|test.csv (synthetic, auto-labelled)
python -m src.train              # trains BiLSTM, saves artifacts/model.pt
python -m src.evaluate           # baseline vs BiLSTM vs hybrid, plots, results.md
python -m src.enhancer "write a python function to check if a number is prime"
streamlit run app.py             # demo UI with attention heatmap
pytest -q                        # sanity tests
```

`artifacts/` already contains a trained model, so `evaluate`, `enhancer`, `app.py` and `pytest` work immediately
(skip generate/train unless you want to retrain).

## Folder map

| Path | What it does |
|---|---|
| `src/generate_dataset.py` | Builds prompts from a core task + optional components. Labels come free: a component we did not add = "missing". Adds typos, casing and filler noise. |
| `src/data.py` | Tokenizer, vocabulary, PyTorch `Dataset`, padding/batching, word dropout |
| `src/model.py` | Embedding -> BiLSTM -> attention pooling -> two heads (task softmax, gap sigmoids) |
| `src/train.py` | Multi-task loss (cross-entropy + BCE), Adam, gradient clipping, early stopping |
| `src/baseline.py` | TF-IDF + logistic regression, the model the BiLSTM must justify itself against |
| `src/evaluate.py` | Metrics on synthetic test and hand-written test, confusion matrix, attention plots |
| `src/slots.py` | Regex rules: detect language, word limits, format words |
| `src/templates.py` | Per-task rewrite text. Edit this to improve output quality |
| `src/enhancer.py` | Glue: model + rules + templates, and a CLI |
| `app.py` | Streamlit demo |
| `data/manual_test.csv` | 40 hand-written, hand-labelled prompts (the honest test set) |

## Labels

Task: `coding, writing, summarization, explanation, analysis, brainstorming`

Gaps (1 = missing): `context, audience, constraints, output_format, purpose`

## Design choices worth knowing

- **Multi-task learning**: one shared encoder, two heads. The task and the gaps help each other.
- **Multi-label, not multi-class**: a prompt can lack several parts at once, so each gap gets its
  own sigmoid and BCE loss.
- **Attention** is used for pooling and for explanation (which words drove the decision).
- **The original prompt is never edited.** It is kept verbatim as the `Task:` line, and only
  missing parts are appended, so the user's intent cannot drift.
- **Context and purpose become placeholders** because only the user knows them. Audience,
  constraints and format get sensible defaults, shown as assumptions.

## Results (from `python -m src.evaluate`, see `artifacts/results.md`)

| Test set | Model | Task acc | Gap macro-F1 |
|---|---|---|---|
| synthetic `test.csv` | TF-IDF + LogReg | 1.000 | 0.999 |
| synthetic `test.csv` | BiLSTM + attention | 1.000 | 1.000 |
| hand-written `manual_test.csv` (40) | TF-IDF + LogReg | 0.975 | 0.943 |
| hand-written `manual_test.csv` (40) | BiLSTM + attention | 0.950 | 0.939 |

**Read this honestly:** on this data the simple baseline is about as good as the BiLSTM.
Synthetic data is too easy (near-perfect scores), and 40 manual prompts is too few to separate
the models. Do not claim the BiLSTM "wins"; claim what you can show: it matches the baseline,
gives attention-based explanations, and the project's next step is harder, more realistic
data where the gap should open up. Note also that the generator was improved (inline phrasing,
e.g. "in under 200 words") after the first model failed on one manual prompt, so the manual
numbers are slightly optimistic. Add fresh, unseen prompts to keep the test honest.

## Known limitations (be upfront about these in your report)

- Training data is synthetic and template-based, so scores on `test.csv` are optimistic.
  Use `manual_test.csv` as the realistic number, and grow it.
- The baseline is strong on this kind of data. The BiLSTM's advantages are word-order
  handling, shared learning across tasks and explainability, and they show up more as
  the data gets more varied.
- Rewriting is template-based, not generated, so output style is consistent but not creative.

## How to improve it

1. **More realistic data**: collect 200+ real prompts (friends, classmates, public prompt
   datasets), label them, and add them to training. This is the biggest win.
2. **Harder synthetic data**: add more paraphrases per component and prompts where a
   component is implied rather than stated.
3. **Pretrained embeddings**: initialise `nn.Embedding` with GloVe.
4. **Threshold tuning**: pick the per-label threshold that maximises F1 on validation data.
5. **Learned rewriter** (optional): a seq2seq LSTM encoder-decoder with attention.
6. **Ablations for the report**: with/without attention, LSTM vs GRU vs vanilla RNN,
   uni vs bidirectional, with/without word dropout.
# prompt-enhancer-2
