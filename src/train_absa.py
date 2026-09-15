"""
Aspect-Based Sentiment Analysis on SemEval-2014 Laptop Reviews

Converted from the original exploratory notebook into a reproducible script.
"""

# # SemEval-2014 Laptop Aspect-Term Sentiment Analysis
# This notebook implements an end-to-end Aspect-Based Sentiment Analysis (ABSA) pipeline for **SemEval-2014 Task 4 (Laptop domain)**:
# 
# - XML parsing + sentence => aspect sample explosion (with offsets)
# - Leakage-reduced group split by `sentence_id`
# - Phase 1 plots (class dist, length hist) + stronger wordclouds + top-words tables
# - Transformer fine-tuning (DeBERTa-v3-base by default) with target marking
# - Mini hyperparameter search + best-of-seeds retrain
# - Train+Val loss and accuracy curves
# - Phase 3 test evaluation: classification report + confusion matrix
# - Error analysis: 5 diverse misclassifications + reason hints
# - Phase 4: 20 challenging sentences + outputs
# - Phase 4: meaning-preserving adversarial flip auto-search

!pip -q install --no-cache-dir \
  "pandas==2.2.2" \
  "torch==2.9.0" "torchvision==0.24.0" "torchaudio==2.9.0" \
  "transformers==4.39.3" "accelerate==0.28.0" "datasets==2.18.0" "evaluate==0.4.2" \
  "scikit-learn==1.5.2" "lxml==5.2.2" "matplotlib==3.8.4" "wordcloud==1.9.3"

!wget -O Laptop_Train_v2.xml \
"https://huggingface.co/datasets/alexcadillon/SemEval2014Task4/raw/main/SemEval%2714-ABSA-TrainData_v2%20%26%20AnnotationGuidelines/Laptop_Train_v2.xml"

!wget -O Laptops_Test_Gold.xml \
"https://huggingface.co/datasets/alexcadillon/SemEval2014Task4/raw/main/ABSA_Gold_TestData/Laptops_Test_Gold.xml"

!ls -lh

!pip -q uninstall -y peft accelerate transformers
!pip -q install --no-cache-dir \
  "transformers==4.39.3" \
  "accelerate==0.29.3" \
  "peft==0.10.0"

import os, json, string, math
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from lxml import etree
from wordcloud import WordCloud, STOPWORDS

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback,
    set_seed,
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GroupShuffleSplit

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)
if DEVICE == "cuda":
    props = torch.cuda.get_device_properties(0)
    print("GPU:", props.name, "| VRAM(GB):", round(props.total_memory / (1024**3), 2))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
print("Torch:", torch.__version__)

# ## Configs

TRAIN_XML = "Laptop_Train_v2.xml"
TEST_XML  = "Laptops_Test_Gold.xml"

MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LEN = 128
VAL_FRAC = 0.15
USE_TARGET_MARKING_DEFAULT = True
LOSS_DEFAULT = "focal"
SEARCH = True
SEARCH_LRS = [1e-5, 2e-5, 3e-5]
SEARCH_EPOCHS = [3, 4]
SEEDS = [13, 42, 77]
EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 2
BATCH_TRAIN = 48
BATCH_EVAL  = 96
GRAD_ACCUM = 1
USE_GRAD_CHECKPOINTING = False
OUTDIR = "absa_out"
FIGDIR = "report_figs"
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

for p in [TRAIN_XML, TEST_XML]:
    print(p, "->", "OK" if os.path.exists(p) else "MISSING (upload it or fix the path)")


# ## Utilities (whitespace, tokenization, robust span finding)

def _clean_ws(s: str) -> str:
    if s is None:
        return ""
    return " ".join(str(s).split())

def _to_int(x):
    try:
        return int(x)
    except Exception:
        return None

def _is_missing(x) -> bool:
    try:
        return x is None or bool(pd.isna(x))
    except Exception:
        return x is None

def _tokenize_simple(text: str):
    out = []
    cur = []
    for ch in (text or ""):
        if ch.isalnum():
            cur.append(ch.lower())
        else:
            if cur:
                out.append("".join(cur))
                cur = []
    if cur:
        out.append("".join(cur))
    return out

def _aspect_tokens(aspect: str):
    return set(_tokenize_simple(aspect or ""))

def _build_wordcloud_text(sentences, extra_stop=set(), remove_tokens=set()):
    toks = []
    for s in sentences:
        for t in _tokenize_simple(s):
            if t in extra_stop:
                continue
            if t in remove_tokens:
                continue
            toks.append(t)
    return " ".join(toks)

def _top_words(sentences, extra_stop=set(), remove_tokens=set(), topk=20):
    freq = {}
    for s in sentences:
        for t in _tokenize_simple(s):
            if t in extra_stop or t in remove_tokens:
                continue
            freq[t] = freq.get(t, 0) + 1
    items = sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:topk]
    return items

def _norm_cmp(s: str) -> str:
    return _clean_ws(s).lower()

def _normalize_with_mapping(s: str):
    s = s or ""
    norm_chars = []
    mapping = []
    prev_space = False
    for i, ch in enumerate(s):
        if ch.isspace():
            if prev_space:
                continue
            norm_chars.append(" ")
            mapping.append(i)
            prev_space = True
        else:
            norm_chars.append(ch)
            mapping.append(i)
            prev_space = False
    return "".join(norm_chars), mapping

def _find_aspect_span_robust(sentence_raw: str, aspect: str):
    s = sentence_raw or ""
    a = (aspect or "").strip()
    if not s or not a:
        return None, None

    s_low = s.lower()
    a_low = a.lower()
    idx = s_low.find(a_low)
    if idx >= 0:
        return idx, idx + len(a)

    s_norm, s_map = _normalize_with_mapping(s)
    a_norm, _ = _normalize_with_mapping(a)
    s_norm_low = s_norm.lower()
    a_norm_low = a_norm.lower()

    idx2 = s_norm_low.find(a_norm_low)
    if idx2 < 0:
        return None, None

    start_orig = s_map[idx2]
    end_norm = idx2 + len(a_norm)
    end_orig = s_map[end_norm - 1] + 1 if end_norm - 1 < len(s_map) else len(s)
    if not (0 <= start_orig <= end_orig <= len(s)):
        return None, None
    return start_orig, end_orig

print(_tokenize_simple("I don't like it."))
print(_find_aspect_span_robust("Great battery life!", "battery life"))

# ## Data: parse SemEval XML → explode to samples (with offsets)

def parse_semeval_laptop_xml(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path} (upload it or update TRAIN_XML/TEST_XML).")

    tree = etree.parse(path)
    root = tree.getroot()

    rows = []
    sent_counter = 0

    for sent in root.findall(".//sentence"):
        sid = sent.get("id", "")
        if not sid:
            sid = f"sent_{sent_counter}"
        sent_counter += 1

        text_node = sent.find("text")
        sentence_raw = text_node.text if (text_node is not None and text_node.text is not None) else ""

        aspect_terms = sent.find("aspectTerms")
        if aspect_terms is None:
            continue

        for at in aspect_terms.findall("aspectTerm"):
            aspect = at.get("term", "")
            pol = (at.get("polarity", "") or "").strip().lower()

            if pol not in LABEL2ID:
                continue
            if not (aspect or "").strip():
                continue

            from_i = _to_int(at.get("from"))
            to_i   = _to_int(at.get("to"))

            if from_i is not None and to_i is not None:
                if not (0 <= from_i <= to_i <= len(sentence_raw)):
                    from_i, to_i = None, None

            rows.append({
                "sentence_id": sid,
                "sentence_raw": sentence_raw,
                "aspect": aspect,
                "polarity": pol,
                "from_i": from_i,
                "to_i": to_i,
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["from_i"] = pd.to_numeric(df["from_i"], errors="coerce").astype("Int64")
        df["to_i"]   = pd.to_numeric(df["to_i"],   errors="coerce").astype("Int64")
    return df

train_df_raw = parse_semeval_laptop_xml(TRAIN_XML)
test_df = parse_semeval_laptop_xml(TEST_XML)

print("Exploded samples -> train_raw:", len(train_df_raw), "test:", len(test_df))
print("Train_raw dist:", train_df_raw["polarity"].value_counts().to_dict())
train_df_raw.head(3)

# ## Split: leakage-reduced GroupSplit by sentence_id

def group_split_balanced(df: pd.DataFrame, val_frac=0.15, seed=42, tries=80):
    rng = np.random.default_rng(seed)

    y = df["polarity"].values
    groups = df["sentence_id"].values

    overall = df["polarity"].value_counts(normalize=True).reindex(["negative","neutral","positive"]).fillna(0).values

    best = None
    best_score = float("inf")

    unique_groups = np.unique(groups)
    if len(unique_groups) < 5:
        idx = np.arange(len(df))
        rng.shuffle(idx)
        n_val = max(1, int(round(val_frac * len(df))))
        val_idx = idx[:n_val]
        tr_idx = idx[n_val:]
        tr = df.iloc[tr_idx].reset_index(drop=True)
        va = df.iloc[val_idx].reset_index(drop=True)
        return tr, va

    for _ in range(tries):
        rs = int(rng.integers(0, 1_000_000))
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=rs)
        tr_idx, va_idx = next(splitter.split(df, y=y, groups=groups))

        tr = df.iloc[tr_idx]
        va = df.iloc[va_idx]

        if tr["polarity"].nunique() < 3 or va["polarity"].nunique() < 3:
            continue

        va_dist = va["polarity"].value_counts(normalize=True).reindex(["negative","neutral","positive"]).fillna(0).values
        score = float(np.abs(va_dist - overall).sum())

        if score < best_score:
            best_score = score
            best = (tr.reset_index(drop=True), va.reset_index(drop=True))

    if best is None:
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
        tr_idx, va_idx = next(splitter.split(df, y=y, groups=groups))
        best = (df.iloc[tr_idx].reset_index(drop=True), df.iloc[va_idx].reset_index(drop=True))

    return best

train_df, val_df = group_split_balanced(train_df_raw, val_frac=VAL_FRAC, seed=42, tries=80)

for _df in (train_df, val_df, test_df):
    if not _df.empty:
        _df["from_i"] = pd.to_numeric(_df["from_i"], errors="coerce").astype("Int64")
        _df["to_i"]   = pd.to_numeric(_df["to_i"],   errors="coerce").astype("Int64")

print("train:", len(train_df), "| dist:", train_df["polarity"].value_counts().to_dict())
print("val  :", len(val_df),   "| dist:", val_df["polarity"].value_counts().to_dict())
print("test :", len(test_df),  "| dist:", test_df["polarity"].value_counts().to_dict())
train_df.head(2)

# ## Phase 1: EDA plots + improved wordclouds + top-words tables

counts = train_df["polarity"].value_counts().reindex(["negative","neutral","positive"]).fillna(0)
plt.figure()
plt.bar(counts.index, counts.values)
plt.title("Class distribution (train)")
plt.xlabel("Polarity")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "class_distribution.png"))
plt.close()

lens = train_df["sentence_raw"].apply(lambda s: len(_clean_ws(s).split()))
plt.figure()
plt.hist(lens, bins=30)
plt.title("Sentence length histogram (train)")
plt.xlabel("Words per sentence")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "sentence_length_hist.png"))
plt.close()

extra_stop = set(STOPWORDS)
extra_stop.update({
    "laptop","computer","pc","notebook","one","also","get","got","still","really","very",
    "would","could","should","im","ive","dont","doesnt","didnt","cant","wont","isnt","arent",
    "don",
})
for ch in string.ascii_lowercase:
    extra_stop.add(ch)

top_aspects = train_df["aspect"].apply(lambda x: (x or "").strip()).value_counts().head(5).index.tolist()

aspect_topwords_rows = []
for asp in top_aspects:
    sub = train_df.loc[train_df["aspect"].astype(str).str.strip() == asp]
    remove_toks = _aspect_tokens(asp)
    wc_text = _build_wordcloud_text(sub["sentence_raw"].tolist(), extra_stop=extra_stop, remove_tokens=remove_toks)
    if not wc_text.strip():
        continue

    wc = WordCloud(
        width=900, height=400, background_color="white",
        stopwords=extra_stop, collocations=False, max_words=150
    ).generate(wc_text)
    plt.figure(figsize=(10, 4))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"WordCloud — aspect: {asp}")
    plt.tight_layout()
    safe = "".join([c if c.isalnum() else "_" for c in asp])[:40]
    plt.savefig(os.path.join(FIGDIR, f"wordcloud_aspect_{safe}.png"))
    plt.close()

    topw = _top_words(sub["sentence_raw"].tolist(), extra_stop=extra_stop, remove_tokens=remove_toks, topk=20)
    for w, c in topw:
        aspect_topwords_rows.append({"type": "aspect", "key": asp, "word": w, "count": c})

polarity_topwords_rows = []
for pol in ["negative", "neutral", "positive"]:
    sub = train_df.loc[train_df["polarity"] == pol]
    wc_text = _build_wordcloud_text(sub["sentence_raw"].tolist(), extra_stop=extra_stop, remove_tokens=set())
    if not wc_text.strip():
        continue

    wc = WordCloud(
        width=900, height=400, background_color="white",
        stopwords=extra_stop, collocations=False, max_words=150
    ).generate(wc_text)
    plt.figure(figsize=(10, 4))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"WordCloud — polarity: {pol}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, f"wordcloud_polarity_{pol}.png"))
    plt.close()

    topw = _top_words(sub["sentence_raw"].tolist(), extra_stop=extra_stop, remove_tokens=set(), topk=20)
    for w, c in topw:
        polarity_topwords_rows.append({"type": "polarity", "key": pol, "word": w, "count": c})

pd.DataFrame(aspect_topwords_rows + polarity_topwords_rows).to_csv(os.path.join(OUTDIR, "top_words_tables.csv"), index=False)

print(f"Saved Phase 1 figures to: {FIGDIR}/")
print(f"Saved top-words tables to: {OUTDIR}/top_words_tables.csv")

# ## Tokenizer + target-marking (offset-based + safe fallback)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
special_added = tokenizer.add_special_tokens({"additional_special_tokens": ["[TGT]", "[/TGT]"]})
print("Added special tokens:", special_added)

def mark_target_from_offsets(sentence_raw: str, aspect: str, from_i, to_i) -> str:
    s = sentence_raw if sentence_raw is not None else ""
    a = (aspect or "").strip()

    if _is_missing(from_i) or _is_missing(to_i):
        return _clean_ws(s)

    try:
        fi = int(from_i)
        ti = int(to_i)
    except Exception:
        return _clean_ws(s)

    if not (0 <= fi <= ti <= len(s)):
        return _clean_ws(s)

    if a:
        span = s[fi:ti]
        if _norm_cmp(span) != _norm_cmp(a):
            fi2, ti2 = _find_aspect_span_robust(s, a)
            if fi2 is None or ti2 is None:
                return _clean_ws(s)
            fi, ti = fi2, ti2

    marked = s[:fi] + "[TGT] " + s[fi:ti] + " [/TGT]" + s[ti:]
    return _clean_ws(marked)

def mark_target_by_substring(sentence: str, aspect: str) -> str:
    s = sentence or ""
    a = (aspect or "").strip()
    if not s or not a:
        return _clean_ws(s)
    s_low = s.lower()
    a_low = a.lower()
    idx = s_low.find(a_low)
    if idx < 0:
        return _clean_ws(s)
    j = idx + len(a)
    marked = s[:idx] + "[TGT] " + s[idx:j] + " [/TGT]" + s[j:]
    return _clean_ws(marked)

row = train_df.sample(1, random_state=1).iloc[0]
print("RAW:", row["sentence_raw"])
print("ASP:", row["aspect"], "| offsets:", row["from_i"], row["to_i"])
print("MARKED:", mark_target_from_offsets(row["sentence_raw"], row["aspect"], row["from_i"], row["to_i"]))

# ## Metrics + loss + custom Trainer + train-metrics callback

def compute_metrics_absa(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    p, r, f, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "precision_macro": p, "recall_macro": r, "f1_macro": f}

def focal_loss(logits, labels, alpha=None, gamma=2.0):
    ce = torch.nn.functional.cross_entropy(logits, labels, weight=alpha, reduction="none")
    pt = torch.exp(-ce)
    loss = ((1 - pt) ** gamma) * ce
    return loss.mean()

class CustomLossTrainer(Trainer):
    def __init__(self, *args, class_weights=None, loss_type="focal", **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.loss_type = loss_type

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits
        w = self.class_weights.to(logits.device) if self.class_weights is not None else None

        if self.loss_type == "weighted_ce":
            loss_fn = torch.nn.CrossEntropyLoss(weight=w)
            loss = loss_fn(logits.view(-1, 3), labels.view(-1))
        else:
            loss = focal_loss(logits.view(-1, 3), labels.view(-1), alpha=w, gamma=2.0)

        return (loss, outputs) if return_outputs else loss

class TrainMetricsCallback(TrainerCallback):
    def __init__(self, train_dataset):
        self.train_dataset = train_dataset
        self.trainer_ref = None

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.trainer_ref is None:
            return control

        handler = getattr(self.trainer_ref, "callback_handler", None)
        saved_callbacks = None
        if handler is not None and hasattr(handler, "callbacks"):
            saved_callbacks = list(handler.callbacks)
            handler.callbacks = [cb for cb in saved_callbacks if cb.__class__.__name__ != "EarlyStoppingCallback"]

        metrics = self.trainer_ref.evaluate(self.train_dataset, metric_key_prefix="train")

        if saved_callbacks is not None:
            handler.callbacks = saved_callbacks

        self.trainer_ref.log(metrics)
        return control

# ## HF Datasets + tokenization + class weights

train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
val_ds   = Dataset.from_pandas(val_df.reset_index(drop=True))
test_ds  = Dataset.from_pandas(test_df.reset_index(drop=True))

def make_tokenizer_fn(use_target_marking: bool):
    def tok(batch):
        sents = batch["sentence_raw"]
        aspects = batch["aspect"]
        pols = batch["polarity"]
        from_is = batch["from_i"]
        to_is = batch["to_i"]

        out_sents = []
        aspects_clean = []
        for s, a, f, t in zip(sents, aspects, from_is, to_is):
            a_clean = _clean_ws(a)
            aspects_clean.append(a_clean)
            if use_target_marking:
                out_sents.append(mark_target_from_offsets(s, a_clean, f, t))
            else:
                out_sents.append(_clean_ws(s))

        enc = tokenizer(
            out_sents,
            text_pair=aspects_clean,
            truncation=True,
            max_length=MAX_LEN,
        )
        enc["labels"] = [LABEL2ID[p] for p in pols]
        return enc
    return tok

collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_labels = [LABEL2ID[p] for p in train_df["polarity"].tolist()]
counts_arr = np.bincount(train_labels, minlength=3).astype(np.float64)
weights = counts_arr.sum() / (len(counts_arr) * np.maximum(counts_arr, 1.0))
class_weights = torch.tensor(weights, dtype=torch.float32)
print("Class counts :", counts_arr)
print("Class weights:", class_weights.detach().cpu().numpy())

tok_fn = make_tokenizer_fn(USE_TARGET_MARKING_DEFAULT)
tmp = train_ds.select(range(2)).map(tok_fn, batched=True)
print(tmp[0].keys())

# ## TrainingArguments + train_one_run()

def make_args(run_name, lr, epochs, seed):
    base = dict(
        output_dir=os.path.join(OUTDIR, run_name),
        report_to=[],
        logging_strategy="epoch",
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=1,

        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.06,
        lr_scheduler_type="linear",

        num_train_epochs=epochs,
        per_device_train_batch_size=BATCH_TRAIN if DEVICE == "cuda" else max(4, BATCH_TRAIN // 2),
        per_device_eval_batch_size=BATCH_EVAL if DEVICE == "cuda" else max(8, BATCH_EVAL // 2),

        gradient_accumulation_steps=max(1, int(GRAD_ACCUM)),

        fp16=(DEVICE == "cuda"),
        seed=seed,
    )
    try:
        return TrainingArguments(**base)
    except TypeError:
        base.pop("greater_is_better", None)
        return TrainingArguments(**base)

def train_one_run(lr, epochs, seed, use_target_marking, loss_type):
    set_seed(seed)

    tok_fn = make_tokenizer_fn(use_target_marking)
    train_tok = train_ds.map(tok_fn, batched=True, remove_columns=train_ds.column_names)
    val_tok   = val_ds.map(tok_fn,   batched=True, remove_columns=val_ds.column_names)
    test_tok  = test_ds.map(tok_fn,  batched=True, remove_columns=test_ds.column_names)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    if USE_GRAD_CHECKPOINTING:
        model.gradient_checkpointing_enable()
    if special_added > 0:
        model.resize_token_embeddings(len(tokenizer))
    model.to(DEVICE)

    run_name = f"{MODEL_NAME.split('/')[-1]}_lr{lr}_ep{epochs}_seed{seed}_tgt{int(use_target_marking)}_{loss_type}"
    args = make_args(run_name, lr, epochs, seed)

    callbacks = []
    train_cb = TrainMetricsCallback(train_tok)
    callbacks.append(train_cb)

    if EARLY_STOPPING:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE))

    trainer = CustomLossTrainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics_absa,
        class_weights=class_weights,
        loss_type=loss_type,
        callbacks=callbacks,
    )
    train_cb.trainer_ref = trainer

    trainer.train()
    val_metrics = trainer.evaluate()

    return {
        "run_name": run_name,
        "lr": lr,
        "epochs": epochs,
        "seed": seed,
        "target_marking": use_target_marking,
        "loss_type": loss_type,
        "val_f1_macro": float(val_metrics.get("eval_f1_macro", np.nan)),
        "val_accuracy": float(val_metrics.get("eval_accuracy", np.nan)),
        "trainer": trainer,
        "test_tok": test_tok,
    }

# ## Mini-search (LR × epochs)

results = []
best_cfg = None

if SEARCH:
    print("=== Mini-search (LR x epochs) ===")
    for lr in SEARCH_LRS:
        for ep in SEARCH_EPOCHS:
            out = train_one_run(
                lr=lr,
                epochs=ep,
                seed=42,
                use_target_marking=USE_TARGET_MARKING_DEFAULT,
                loss_type=LOSS_DEFAULT,
            )
            results.append({k: out[k] for k in out if k not in ("trainer", "test_tok")})
            print(f"Run: lr={lr} ep={ep} -> val_f1_macro={out['val_f1_macro']:.4f}")

    res_df = pd.DataFrame(results).sort_values("val_f1_macro", ascending=False).reset_index(drop=True)
    res_df.to_csv(os.path.join(OUTDIR, "search_results.csv"), index=False)
    print("\nTop search results:")
    try:
        display(res_df.head(10))
    except Exception:
        print(res_df.head(10))

    best_row = res_df.iloc[0].to_dict()
    best_cfg = {
        "lr": float(best_row["lr"]),
        "epochs": int(best_row["epochs"]),
        "target_marking": bool(best_row["target_marking"]),
        "loss_type": str(best_row["loss_type"]),
    }
else:
    best_cfg = {
        "lr": 2e-5,
        "epochs": 4,
        "target_marking": USE_TARGET_MARKING_DEFAULT,
        "loss_type": LOSS_DEFAULT,
    }

print("\nSelected best config:", best_cfg)

# ## Best-of-seeds retrain

print("=== Best-of-seeds retrain ===")
seed_results = []
best_run = None

for sd in SEEDS:
    out = train_one_run(
        lr=best_cfg["lr"],
        epochs=best_cfg["epochs"],
        seed=sd,
        use_target_marking=best_cfg["target_marking"],
        loss_type=best_cfg["loss_type"],
    )
    seed_results.append({k: out[k] for k in out if k not in ("trainer", "test_tok")})
    print(f"Seed {sd} -> val_f1_macro={out['val_f1_macro']:.4f}")

    if (best_run is None) or (out["val_f1_macro"] > best_run["val_f1_macro"]):
        best_run = out

seed_df = pd.DataFrame(seed_results).sort_values("val_f1_macro", ascending=False).reset_index(drop=True)
seed_df.to_csv(os.path.join(OUTDIR, "seed_results.csv"), index=False)

print("\nSeed results (sorted):")
try:
    display(seed_df)
except Exception:
    print(seed_df)

trainer = best_run["trainer"]
test_tok = best_run["test_tok"]
print("\nBest run:", best_run["run_name"], "| val_f1_macro=", best_run["val_f1_macro"])

# ## Curves: train+val loss and train+val accuracy

log_history = trainer.state.log_history

epoch_train_loss = {}
epoch_val_loss = {}
epoch_train_acc = {}
epoch_val_acc = {}

for item in log_history:
    if "epoch" not in item:
        continue
    ep = float(item["epoch"])

    if ("loss" in item) and ("eval_loss" not in item) and ("train_loss" not in item):
        epoch_train_loss[ep] = item["loss"]

    if "eval_loss" in item:
        epoch_val_loss[ep] = item["eval_loss"]
        if "eval_accuracy" in item:
            epoch_val_acc[ep] = item["eval_accuracy"]

    if "train_loss" in item:
        epoch_train_loss[ep] = item["train_loss"]
    if "train_accuracy" in item:
        epoch_train_acc[ep] = item["train_accuracy"]

all_epochs = sorted(set(list(epoch_train_loss.keys()) + list(epoch_val_loss.keys())))
train_loss = [epoch_train_loss.get(ep, np.nan) for ep in all_epochs]
val_loss   = [epoch_val_loss.get(ep, np.nan) for ep in all_epochs]
train_acc  = [epoch_train_acc.get(ep, np.nan) for ep in all_epochs]
val_acc    = [epoch_val_acc.get(ep, np.nan) for ep in all_epochs]

plt.figure()
plt.plot(all_epochs, train_loss, marker="o")
plt.plot(all_epochs, val_loss, marker="o")
plt.title("Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["train_loss", "val_loss"])
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "training_loss_curve.png"))
plt.close()

plt.figure()
plt.plot(all_epochs, train_acc, marker="o")
plt.plot(all_epochs, val_acc, marker="o")
plt.title("Accuracy vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["train_accuracy", "val_accuracy"])
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "accuracy_curve_train_val.png"))
plt.close()

print(f"Saved training curves to: {FIGDIR}/")

# ## Phase 3: Test evaluation + confusion matrix

pred_out = trainer.predict(test_tok)
test_logits = pred_out.predictions
test_preds = np.argmax(test_logits, axis=-1)
test_true  = np.array(pred_out.label_ids)

rep = classification_report(
    test_true, test_preds,
    target_names=[ID2LABEL[i] for i in range(3)],
    digits=4,
    zero_division=0
)
print("Classification report (TEST):\n")
print(rep)

with open(os.path.join(OUTDIR, "classification_report.txt"), "w") as f:
    f.write(rep)

cm = confusion_matrix(test_true, test_preds, labels=[0,1,2])
plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix (Test)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks([0,1,2], [ID2LABEL[i] for i in range(3)])
plt.yticks([0,1,2], [ID2LABEL[i] for i in range(3)])
for i in range(3):
    for j in range(3):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "confusion_matrix.png"))
plt.close()

print(f"Saved confusion matrix to: {FIGDIR}/confusion_matrix.png")

# ## Phase 3: Error analysis (5 diverse misclassifications + reason hints)

def _contains_any(sentence: str, phrases):
    s = (sentence or "").lower()
    return any(p in s for p in phrases)

def guess_error_reason(sentence: str, aspect: str, true_lbl: str, pred_lbl: str) -> str:
    s = (sentence or "").lower()
    a = (aspect or "").lower()

    if (" not " in f" {s} ") or ("n't" in s):
        return "Negation scope (not/n't)"
    if _contains_any(s, [" but ", " however ", " although ", " though ", " yet "]):
        return "Contrast / discourse shift (but/however/although)"
    if _contains_any(s, [" if ", " unless ", " provided "]):
        return "Conditional / hypothetical language"
    if ("?" in s) or _contains_any(s, ["yeah right", "sure", "as if"]):
        return "Sarcasm / rhetorical phrasing"
    if a and (s.count(a) >= 2):
        return "Multiple target mentions / ambiguity"
    if true_lbl == "neutral" and pred_lbl != "neutral":
        return "Neutral vs sentiment confusion (implicit sentiment)"
    return "Implicit sentiment / rare phrasing / context missing"

mis = np.where(test_true != test_preds)[0]
print("Total misclassified:", len(mis))

selected = []
seen_reasons = set()
for i in mis:
    row = test_df.iloc[int(i)]
    true_lbl = ID2LABEL[int(test_true[int(i)])]
    pred_lbl = ID2LABEL[int(test_preds[int(i)])]
    reason = guess_error_reason(row["sentence_raw"], row["aspect"], true_lbl, pred_lbl)
    if reason not in seen_reasons:
        selected.append((int(i), reason))
        seen_reasons.add(reason)
    if len(selected) == 5:
        break

if len(selected) < 5:
    for i in mis:
        if any(i == j for j, _ in selected):
            continue
        row = test_df.iloc[int(i)]
        true_lbl = ID2LABEL[int(test_true[int(i)])]
        pred_lbl = ID2LABEL[int(test_preds[int(i)])]
        reason = guess_error_reason(row["sentence_raw"], row["aspect"], true_lbl, pred_lbl)
        selected.append((int(i), reason))
        if len(selected) == 5:
            break

err_out = []
for idx, reason in selected[:5]:
    row = test_df.iloc[idx]
    true_lbl = ID2LABEL[int(test_true[idx])]
    pred_lbl = ID2LABEL[int(test_preds[idx])]
    err_out.append({
        "sentence": _clean_ws(row["sentence_raw"]),
        "aspect": _clean_ws(row["aspect"]),
        "true": true_lbl,
        "pred": pred_lbl,
        "reason_hint": reason
    })

with open(os.path.join(OUTDIR, "error_analysis_5.json"), "w") as f:
    json.dump(err_out, f, indent=2)

print("Saved:", os.path.join(OUTDIR, "error_analysis_5.json"))
pd.DataFrame(err_out)

# ## Phase 4: 20 challenging sentences + outputs

def predict_one(sentence: str, aspect: str, use_target_marking=True):
    trainer.model.eval()
    s = mark_target_by_substring(sentence, aspect) if use_target_marking else _clean_ws(sentence)
    enc = tokenizer(_clean_ws(s), text_pair=_clean_ws(aspect), truncation=True, max_length=MAX_LEN, return_tensors="pt")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        logits = trainer.model(**enc).logits[0]
    probs = torch.softmax(logits, dim=0).detach().cpu().numpy()
    pred = ID2LABEL[int(np.argmax(probs))]
    return pred, {ID2LABEL[i]: float(probs[i]) for i in range(3)}

challenging_by_type = {
    "Negation scope": [
        ("The battery is not bad at all.", "battery"),
        ("I don't think the keyboard is good.", "keyboard"),
        ("The touchpad isn't terrible, just not great.", "touchpad"),
        ("Not only is the screen bright, it’s also not too reflective.", "screen"),
    ],
    "Contrast (but/however)": [
        ("The screen is sharp, but the speakers are disappointing.", "speakers"),
        ("The keyboard feels solid; however, the trackpad is frustrating.", "trackpad"),
        ("Performance is fast, but the battery life is short.", "battery life"),
        ("The laptop is lightweight, yet the build quality feels cheap.", "build quality"),
    ],
    "Conditionals / hypotheticals": [
        ("If the battery lasted longer, this would be perfect.", "battery"),
        ("It would be great if the fan noise were lower.", "fan"),
        ("Unless you use headphones, the speakers won't impress you.", "speakers"),
        ("Provided you keep it plugged in, performance is excellent.", "performance"),
    ],
    "Sarcasm / rhetorical": [
        ("Oh great, the keyboard stops working again.", "keyboard"),
        ("Sure, the battery is 'amazing'—it dies in two hours.", "battery"),
        ("Yeah right, the touchpad is 'precise'.", "touchpad"),
        ("Who thought this screen glare was a good idea?", "screen"),
    ],
    "Aspect collision / mixed sentiment": [
        ("The display is gorgeous, and the keyboard is great, but the battery is awful.", "battery"),
        ("I love the speed, but I hate the heat.", "heat"),
        ("The speakers are loud, though the audio quality is muddy.", "audio quality"),
        ("The build is sturdy, but the hinge feels weak.", "hinge"),
    ],
}

challenging_cases = []
for k in challenging_by_type:
    for case in challenging_by_type[k]:
        challenging_cases.append((k, case[0], case[1]))
challenging_cases = challenging_cases[:20]

challenging_outputs = []
for cat, sent, asp in challenging_cases:
    pred, probs = predict_one(sent, asp, use_target_marking=best_cfg["target_marking"])
    challenging_outputs.append({
        "category": cat,
        "sentence": sent,
        "aspect": asp,
        "pred": pred,
        "probs": probs
    })

with open(os.path.join(OUTDIR, "challenging_outputs_20.json"), "w") as f:
    json.dump(challenging_outputs, f, indent=2)

print("Saved:", os.path.join(OUTDIR, "challenging_outputs_20.json"))
pd.DataFrame(challenging_outputs)[["category","aspect","pred"]].head(10)

# ## Phase 4: Adversarial flip auto-search (meaning-preserving edits)

def apply_meaning_preserving_edits(sentence: str):
    s = _clean_ws(sentence)
    if not s:
        return []

    variants = []
    variants.append(s)
    variants.append(s + " Overall.")
    if s.endswith("."):
        variants.append(s[:-1] + "!")
    else:
        variants.append(s + "!")
    variants.append("In general, " + (s[0].lower() + s[1:] if len(s) > 1 else s))
    variants.append("Honestly, " + (s[0].lower() + s[1:] if len(s) > 1 else s))

    words = s.split()
    for i, w in enumerate(words):
        lw = w.lower().strip(string.punctuation)
        if lw == "very":
            w2 = w.replace("very", "really") if "very" in w else (w.replace("Very", "Really") if "Very" in w else w)
            variants.append(" ".join(words[:i] + [w2] + words[i+1:]))
        if lw == "really":
            w2 = w.replace("really", "very") if "really" in w else (w.replace("Really", "Very") if "Really" in w else w)
            variants.append(" ".join(words[:i] + [w2] + words[i+1:]))

    tokens = s.split()
    for i, w in enumerate(tokens):
        if w.lower() == "but" and i > 0:
            prev = tokens[i-1]
            if not prev.endswith(","):
                variants.append(" ".join(tokens[:i-1] + [prev + ","] + tokens[i:]))
            break

    out, seen = [], set()
    for v in variants:
        v = _clean_ws(v)
        if v and v not in seen:
            out.append(v)
            seen.add(v)
    return out

def find_adversarial_flip(max_samples_to_try=400):
    rng = np.random.default_rng(42)
    idxs = np.arange(len(test_df))
    rng.shuffle(idxs)

    tried = 0
    for idx in idxs[:max_samples_to_try]:
        tried += 1
        row = test_df.iloc[int(idx)]
        base_sent = _clean_ws(row["sentence_raw"])
        asp = _clean_ws(row["aspect"])
        true_lbl = _clean_ws(row["polarity"]).lower()
        if not base_sent or not asp or true_lbl not in LABEL2ID:
            continue

        base_pred, _ = predict_one(base_sent, asp, use_target_marking=best_cfg["target_marking"])
        if base_pred != true_lbl:
            continue

        candidates = apply_meaning_preserving_edits(base_sent)
        for v in candidates:
            if asp and (v.lower().find(asp.lower()) < 0):
                continue
            new_pred, _ = predict_one(v, asp, use_target_marking=best_cfg["target_marking"])
            if new_pred != base_pred:
                return {
                    "found": True,
                    "aspect": asp,
                    "true_label": true_lbl,
                    "original_sentence": base_sent,
                    "original_pred": base_pred,
                    "modified_sentence": v,
                    "modified_pred": new_pred,
                    "samples_tried": int(tried),
                }

    return {"found": False, "samples_tried": int(tried)}

adv = find_adversarial_flip()
with open(os.path.join(OUTDIR, "adversarial_example.json"), "w") as f:
    json.dump(adv, f, indent=2)

print("Saved:", os.path.join(OUTDIR, "adversarial_example.json"))
adv

# ## Key outputs:
# - `absa_out/search_results.csv`
# - `absa_out/seed_results.csv`
# - `absa_out/classification_report.txt`
# - `absa_out/top_words_tables.csv`
# - `absa_out/error_analysis_5.json`
# - `absa_out/challenging_outputs_20.json`
# - `absa_out/adversarial_example.json`
# - `report_figs/*` (plots)
