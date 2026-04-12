# Phase 1 — IUPACGPT Deep-Dive Walkthrough

## Overview: What Is Phase 1 and Why Does It Exist?

Phase 1 is the **text-based toxicity prediction engine** of ToxGuard. Its core insight is radical and elegant: throw away SMILES strings, 2D fingerprints, and graph structures entirely — and ask a **language model trained on IUPAC chemical names** to classify molecules as toxic or non-toxic based on *the name alone*.

Why is this powerful?

- IUPAC names are **systematic strings that encode molecular structure**. The name `2-chloro-1,3-butadiene` literally tells you a chlorine atom is on carbon 2 of a butadiene skeleton. A language model that understands this language has implicitly learned chemistry.
- Toxicologists and chemists think in IUPAC names. A model operating in this space is interpretable to domain experts.
- It feeds a **256-dimensional molecular embedding** to Phase 2 (EGNN) — acting as a learned text-based node feature for the 3D graph model.
- Phase 2, 3, and 4 all **depend on Phase 1's trained weights** as their foundation.

---

## Directory Structure

```
Phase1-IUPACGPT/
├── steps/                        # The 7-step sequential pipeline (run these in order)
│   ├── step1_download_data.py    # Download & prepare all 6 raw datasets
│   ├── step2_preprocess.py       # Canonicalize SMILES, compute binary labels, cross-dedup
│   ├── step3_smiles_to_iupac.py  # SMILES → IUPAC name via 3-API cascade + collision fix
│   ├── step4_verify_lora.py      # Verify LoRA setup before training
│   ├── step5_train.py            # LoRA fine-tune IUPACGPT backbone on binary classification
│   ├── step5b_train_5fold_cv.py  # 5-fold cross-validation variant
│   ├── step6_evaluate.py         # Full evaluation: test split + external T3DB validation
│   └── step7_predict.py          # Inference: single molecule or batch with attention maps
│
├── iupacGPT_finetune/            # The core Python package (imported by all steps)
│   ├── __init__.py               # Package exports
│   ├── tokenizer.py              # SentencePiece-based IUPAC tokenizer
│   ├── model.py                  # ToxGuardModel + ToxGuardLitModel (PyTorch Lightning)
│   ├── lora.py                   # LoRA adapter: LoRAConfig, LoRALayer, apply_lora_to_model
│   ├── data_pipeline.py          # Dataset classes, scaffold splitting, DataLoaders
│   ├── inference.py              # ToxGuardPredictor + ToxGuardPrediction
│   ├── interpretability.py       # Attention attribution, heatmaps, toxicophore detection
│   └── calibration.py            # Temperature scaling (post-hoc probability calibration)
│
├── Ablations/
│   └── run_ablations.py          # Ablation experiments (pooling, rank, loss, threshold)
│
└── iupacGPT_outputs/             # Training artifacts (auto-created by step5)
    ├── last_run.txt              # Pointer to most recent run folder
    ├── lora_config.json          # LoRA config written after each training run
    └── run_YYYYMMDD_HHMMSS/      # Per-run folder
        ├── lora_weights.pt       # Saved LoRA adapter weights (~4MB)
        ├── config.json           # Full training config
        ├── results.json          # Test metrics
        ├── checkpoints/          # PyTorch Lightning checkpoints
        ├── tensorboard/          # TensorBoard training logs
        ├── evaluation_report.txt # Step6 evaluation report
        ├── eval_metrics.json     # Step6 metrics JSON
        ├── temperature.pt        # Calibration temperature
        └── threshold.json        # Optimal decision threshold from val sweep
```

---

## The 7-Step Pipeline — In Depth

### STEP 1 — `step1_download_data.py`: Download & Prepare Raw Datasets

**What it does:** Acquires all raw data needed for training. Manages 6 distinct chemical toxicity sources.

| Dataset | Source | Size | Label Type |
|---------|--------|------|-----------|
| ToxCast | DeepChem S3 | 8,597 compounds × 617 assays | Multi-assay panel |
| Tox21 | DeepChem S3 | 7,831 compounds × 12 assays | Multi-assay panel |
| T3DB | Local bulk CSV (www.t3db.ca) | ~3,500 compounds | Known toxins (LD50-based) |
| hERG | PyTDC (`hERG_Karim`) | 13,445 compounds | Cardiac ion channel blocking |
| DILI | PyTDC (`DILI`) | 475 compounds | Drug-induced liver injury |
| Common Molecules | Built separately | ~1,100 curated | Expert-labeled |

**Key design decisions:**
- ToxCast and Tox21 are downloaded as `.csv.gz` from DeepChem's S3 bucket
- hERG and DILI are fetched through the **PyTDC** library which wraps curated benchmarks
- T3DB must be downloaded manually (bulk CSV from www.t3db.ca) because it has no public API
- The `Common Molecules` dataset is built by a companion script `build_common_molecules.py` — it's a curated set of ~1,100 short IUPAC names (easy for the tokenizer) with authoritative toxic/non-toxic labels

**Why 6 datasets?** Toxicity is multi-dimensional — ToxCast covers general mechanism-level activity across 617 assays, Tox21 covers nuclear receptor and stress response pathways, T3DB covers natural toxins with LD50 data, hERG covers cardiac safety (cardiac safety is the #1 cause of drug withdrawal), and DILI covers liver injury (liver tox is the top reason for post-market withdrawal). Combining all gives breadth.

---

### STEP 2 — `step2_preprocess.py`: Canonicalize, Label, Deduplicate

**What it does:** Transforms raw multi-assay matrices into clean `(smiles, is_toxic)` binary pairs.

**Three critical operations happen here:**

#### 1. SMILES Canonicalization (RDKit)

```python
def canonicalize_smiles(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(smi.strip())
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if len(frags) > 1:
        mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())  # strip salts
    return Chem.MolToSmiles(mol, isomericSmiles=True)  # canonical + stereo
```

This strips salt counterions (e.g., `CC(=O)[O-].[Na+]` → `CC(=O)[O-]`) and converts to a single canonical SMILES form. Without this, the same molecule can appear with dozens of different SMILES strings.

#### 2. Binary Label Computation

- **ToxCast (617 assays):** `is_toxic = 1 if any_assay_positive else 0` (conservative: any signal = toxic)
- **Tox21 (12 assays):** Same logic — any positive assay = toxic
- **hERG:** Label comes directly (`is_herg_blocker == 1`)
- **DILI:** Label comes directly (`is_dili == 1`)
- **Conflict resolution:** If same canonical SMILES has conflicting labels within a dataset → keep the toxic label (safety-first)

#### 3. Cross-Dataset Deduplication (Priority Order)

```
T3DB > ToxCast > hERG > Tox21 > DILI
```

Steps:
1. Remove from ToxCast any SMILES already in T3DB
2. Remove from Tox21 any SMILES already in ToxCast or T3DB
3. Remove from hERG any SMILES already in T3DB or ToxCast
4. Remove from DILI any SMILES already in any higher-priority dataset

**Why dedup at all?** If "aspirin" appears in both ToxCast and Tox21, it could contaminate both splits of your train/test evaluation. By keeping each molecule in at most one dataset, the split is cleaner.

**Output:** `data/toxcast_final.csv`, `data/tox21_final.csv`, `data/herg_final.csv`, `data/dili_final.csv` — each with columns `[smiles, is_toxic]`. **No IUPAC names yet** at this stage.

---

### STEP 3 — `step3_smiles_to_iupac.py`: SMILES → IUPAC Name Resolution

This is the most complex step. The IUPACGPT model was trained on IUPAC names, so every molecule needs to be named. This script resolves IUPAC names via a **4-phase pipeline** with a 3-API cascade.

#### Phase 1 — SMILES Preprocessing (RDKit)

Beyond basic canonicalization, this step performs **tautomer canonicalization**:

```python
_TAUTOMER_ENUMERATOR = rdMolStandardize.TautomerEnumerator()
mol = _TAUTOMER_ENUMERATOR.Canonicalize(mol)
```

Why? A molecule like acetylacetone can exist in both keto and enol forms, each with a different SMILES but the same IUPAC name. Without tautomer canonicalization, they would both resolve to the same IUPAC name and be incorrectly flagged as "collision" — causing valid data to be dropped.

Two canonical SMILES are computed per molecule:
- `canonical_iso` — preserves stereo (for API lookup)
- `canonical_flat` — strips stereo (for collision grouping)

#### Phase 2 — Triple-API Resolution Cascade

For each unique canonical SMILES, the code tries APIs in order:

```
PubChem PUG REST → ChemSpider v2 → NCI CIR
```

**PubChem** (primary):
```python
POST https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/property/IUPACName/JSON
data = {"smiles": canonical_iso}
```
Rate limited to 5 req/s (~220ms delay between calls). Returns IUPAC names for ~80% of compounds.

**ChemSpider** (fallback):
3-step async API: submit filter → poll for compound ID → fetch IUPAC name. Requires an API key.

**NCI CIR** (last resort):
```
GET https://cactus.nci.nih.gov/chemical/structure/{url_encoded_smiles}/iupac_name
```
Simple REST, no auth needed. Slower and less reliable.

A persistent **cache** (`data/step3_cache.csv`) prevents re-querying molecules already resolved. Saved every 100 resolutions to survive interruptions.

#### Phase 3 — Collision Detection & Stereo Prefix Resolution

A **collision** = two different isomeric SMILES resolved to the **same** IUPAC name. This is common with stereo isomers: (R)-limonene and (S)-limonene might both return "limonene" from PubChem.

Resolution algorithm:
1. Group all canonical SMILES by the resolved IUPAC name
2. For each collision group, extract stereo prefixes from RDKit:
   ```python
   prefix = "(R)-"  # or "(2S,3R)-", "(E)-", etc.
   proposed_name = prefix + original_name
   ```
3. If all proposed names are now unique → collision resolved!
4. If still ambiguous → mark as `collision_unresolvable`, drop from datasets

**Why does this matter?** Without this step, two stereo isomers with different toxicities would be merged under the same name, poisoning labels.

#### Phase 4 — Apply to Datasets

Maps `raw_smiles → canonical_iso → final_iupac_name` and writes the result back into each `_final.csv`. Molecules with unresolved or blank IUPAC names are **dropped** — they cannot be processed by a tokenizer that only understands IUPAC syntax.

**Output:** All `_final.csv` files now have columns `[smiles, iupac_name, is_toxic]`.

> **Key insight:** IUPAC names are the *only* input to the model at inference. SMILES are only needed for scaffold-splitting and EGNN (Phase 2). Once training is complete, Phase 1 can classify any molecule given just its name.

---

### STEP 4 — `step4_verify_lora.py`: Pre-Training Verification

A sanity-check step that verifies:
- The IUPACGPT backbone loads correctly
- LoRA adapters can be applied without errors
- Parameter counts are as expected (~7M base, ~14% trainable with rank-32)
- A test forward pass produces valid outputs

Saves `iupacGPT_outputs/lora_config.json` which is later read by step5/step6/step7.

---

### STEP 5 — `step5_train.py`: LoRA Fine-Tuning

This is the heart of Phase 1 — where the model actually learns to predict toxicity.

#### Model Architecture (in full detail)

```
IUPAC Name (string)
        ↓
ToxGuardTokenizer (SentencePiece, ~32K vocab)
        ↓
[BOS] + token_ids + [EOS]    ← sequence of integers
        ↓
GPT2Model (8 layers, 8 heads, 256 hidden dim) ← loaded from IUPACGPT checkpoint
  + LoRA adapters on c_attn, c_proj, c_fc    ← only these are trained
        ↓
last_hidden_state: (B, L, 256)
        ↓
Pooling strategy (last_token / mean / cls)
        ↓
pooled: (B, 256)          ← single molecular representation
        ↓
    ┌──────────────────────────┐
    │                          │
ToxicityHead                egnn_projection
Linear(256,128)→GELU→       Linear(256,256)
Dropout→Linear(128,1)       →LayerNorm
    │                          │
binary_logit (B,)         egnn_vector (B,256)
    │
sigmoid(binary_logit)
    │
P(toxic) ∈ [0,1]
    │
severity label (5 bands)
```

**IUPACGPT backbone:** A compact GPT-2 style transformer (8 layers, 8 attention heads, 256 dimensions, ~7M parameters) pre-trained on the IUPAC Chemical Identifier (ICI) corpus — millions of IUPAC names scraped from PubChem. This backbone has learned the grammar and semantics of chemical nomenclature.

**Why GPT-2 style and not BERT/encoder?** IUPACGPT was designed as an autoregressive language model for IUPAC name generation. We repurpose it as a sequence classifier by extracting the last-token hidden state (which in autoregressive LMs captures information about the entire prefix). This is analogous to how GPT-2 was adapted for classification in GPT-2-based sentiment analysis.

#### LoRA (Low-Rank Adaptation) — In Depth

LoRA is applied to three GPT-2 weight matrices per layer:
- `c_attn` — the combined Q/K/V projection (8 modules, one per transformer layer)
- `c_proj` — the attention output + MLP output projection (16 modules)
- `c_fc` — the MLP feedforward first linear (FFN up-projection, 8 modules)

**The math:**
```
h = W₀x + (α/r) · B·A·x
```
where:
- `W₀` is the original frozen weight (not updated)
- `A ∈ R^(d_in × r)` — initialized with Kaiming uniform
- `B ∈ R^(r × d_out)` — initialized to **zero** (so LoRA starts as identity, no disruption)
- `α/r` is the scaling factor (with `α=64, r=32`, scale = 2.0)

**Why freeze the base and only train LoRA?** IUPACGPT's pre-trained weights encode ~5 years of chemical language knowledge. Fine-tuning everything would destroy this knowledge (catastrophic forgetting). LoRA lets us steer the representations toward toxicity classification while preserving the backbone.

**Parameter efficiency:**
```
Total parameters: ~7.1M
LoRA parameters: ~1.05M (on c_attn + c_proj + c_fc)
Toxicity head: ~100K
Trainable: ~14% of total
```

Only the trainable parameters are updated by the optimizer. The frozen backbone parameters require no gradient computation, saving significant memory and compute.

#### Loss Function — Focal Loss with Label Smoothing

```python
# Label smoothing: prevents overconfident predictions
smoothed_targets = labels * (1.0 - 0.1) + 0.1 / 2.0   # ε = 0.1

# Focal loss: down-weights easy examples, up-weights hard ones
bce = F.binary_cross_entropy_with_logits(logits, smoothed_targets, reduction="none")
p_t = p * y + (1 - p) * (1 - y)
focal_weight = (1 - p_t) ** 1.5          # gamma = 1.5
alpha_t = (0.45 * y + 0.55 * (1 - y))   # alpha = 0.45 for toxic class
loss = (alpha_t * focal_weight * bce).mean()
```

**Why focal loss?** The training set is ~54% toxic (after removing T3DB and ClinTox). Focal loss discounts easy examples (molecules the model has already learned) and focuses capacity on hard boundary cases — typically moderately toxic or unusual molecular structures.

**Why `focal_alpha = 0.45`?** With 54% toxic in training, `alpha=0.5` would be neutral. `0.45` slightly up-weights the non-toxic class, compensating for the small positive imbalance.

**Why exclude T3DB from training?** T3DB is 99.2% toxic. Including it would heavily bias the model toward over-predicting toxicity. It's kept as an **external validation set** (meaning it has never been seen during training) to test recall on a near-all-toxic corpus.

#### Training Infrastructure

```python
Trainer(
    max_epochs=40,
    accelerator="auto",          # GPU if available
    gradient_clip_val=1.0,       # Prevents gradient explosion
    accumulate_grad_batches=2,   # Effective batch size = 32
    precision="16-mixed",        # FP16 mixed precision
)
```

**Learning rate schedule:** Linear warmup (1200 steps, starting at 1% of peak LR) → cosine annealing decay to 1% of peak. This prevents the LoRA layers from overshooting early in training when they haven't yet adapted to the task.

**Early stopping:** Monitors `val_auroc` with patience=20 epochs and min_delta=1e-3.

**Checkpoint:** Saves top-3 checkpoints by `val_auroc`. Best checkpoint is automatically loaded for final test evaluation.

**Scaffold split (Bemis-Murcko):** The dataset is split using molecular scaffolds rather than random assignment. This prevents structural data leakage — molecules with the same scaffold (core ring system) all go to the same split. Without this, a model could trivially generalize by memorizing scaffolds seen in training, inflating test metrics.

---

### STEP 6 — `step6_evaluate.py`: Comprehensive Evaluation

This step loads the trained LoRA weights and runs three evaluation passes:

#### 1. Validation-Set Threshold Tuning

Sweeps thresholds 0.10–0.90 in 0.01 increments, computing F1, MCC, and accuracy at each. Selects the threshold that maximizes **MCC** (Matthews Correlation Coefficient — most robust metric for binary classification with class imbalance). This threshold is then used for all subsequent predictions and saved to `threshold.json`.

**Why MCC?** Unlike F1 which only considers positive class precision/recall, MCC accounts for all four quadrants of the confusion matrix. A high MCC means the model is genuinely discriminating, not just guessing the majority class.

#### 2. Temperature Scaling (Probability Calibration)

```python
P_calibrated(toxic) = sigmoid(logit / T)
```

Learns a single scalar `T` on the validation set by minimizing NLL using L-BFGS. If `T > 1`, the model's raw probabilities were overconfident and are softened. If `T < 1`, probabilities were underconfident and are sharpened.

**Why calibration matters?** A model with AUC-ROC = 0.92 but poor calibration gives `P(toxic)=0.95` for a molecule that's actually, say, 60% likely to be toxic. Downstream phases (Phase 2 EGNN, Phase 3 RAG, Phase 4 RL) use the probability scores, so miscalibration cascades.

#### 3. External Validation on T3DB

T3DB was **never seen during training** (99.2% toxic corpus). The headline metric is **Recall** (sensitivity): what fraction of known toxins did the model correctly flag? High recall on T3DB validates that the model hasn't learned to rely on statistical shortcuts from training data distribution.

**Metrics reported:**
- AUC-ROC, AUC-PRC (threshold-independent)
- Accuracy, Precision, Recall, F1, MCC at both 0.50 (default) and tuned threshold
- Specificity (TN / (TN+FP)) — important for T3DB where all positives matter
- Full confusion matrix

**Output artifacts:**
- `evaluation_report.txt` — human-readable report
- `eval_metrics.json` — machine-readable metrics
- `temperature.pt` — calibration temperature
- `threshold.json` — optimal threshold + full sweep

---

### STEP 7 — `step7_predict.py`: Inference API

The user-facing inference script. Takes any IUPAC name and returns a complete toxicity assessment.

**Usage modes:**
```bash
# Single molecule
python steps/step7_predict.py --molecule "1,3,7-trimethyl-3,7-dihydro-1H-purine-2,6-dione"

# Batch from file
python steps/step7_predict.py --input_file molecules.txt --output_file results.json

# With 256-dim EGNN vector (for Phase 2)
python steps/step7_predict.py --molecule "nitrobenzene" --egnn_vector

# With attention heatmap (interpretability)
python steps/step7_predict.py --molecule "tetrachloromethane" --attention_map
```

**Output for each molecule:**
```
Prediction    : TOXIC
P(toxic)      : 0.9234
Severity      : Highly toxic
Confidence    : 0.9234
```

**Severity bands** (derived purely from P(toxic), no separate training):
| P(toxic) range | Label |
|---|---|
| < 0.20 | Non-toxic |
| 0.20–0.50 | Unlikely toxic |
| 0.50–0.65 | Likely toxic |
| 0.65–0.80 | Moderately toxic |
| ≥ 0.80 | Highly toxic |

The 0.50 boundary aligns exactly with the binary decision threshold, so the severity label is always consistent with the binary prediction.

---

## The `iupacGPT_finetune` Package — Module by Module

### `tokenizer.py` — ToxGuardTokenizer

**The problem it solves:** The original IUPACGPT code used HuggingFace's `T5Tokenizer` as a wrapper around the SentencePiece vocabulary. In `transformers >= 5.x`, `T5Tokenizer` no longer correctly loads the full SPM vocabulary — it collapses to just 104 tokens (4 special + 100 extra_ids), mapping nearly all IUPAC subwords to `<unk>`. This completely broke tokenization.

**The fix:** Use `sentencepiece` Python library directly, bypassing `T5Tokenizer` entirely, while exposing a HuggingFace-compatible interface (`__call__`, `pad_token_id`, `eos_token_id`, etc.).

**IUPAC-specific handling:**
- Spaces in IUPAC names (e.g., "sodium chloride") are replaced with underscores before encoding — SentencePiece treats spaces as word boundaries but IUPAC spaces are part of the name
- Strips the leading `▁` sentinel token that SentencePiece prepends to the first word
- BOS token = UNK token (IUPACGPT convention: `<unk>` used as BOS)

**Special token IDs:**
- `0` = `<pad>`
- `1` = `</s>` (EOS)
- `2` = `<unk>` (also BOS)

---

### `model.py` — ToxGuardModel + ToxGuardLitModel

**ToxGuardModel:**
- Wraps `GPT2Model` (from HuggingFace transformers) loaded from the IUPACGPT checkpoint
- Has three pooling strategies: `last_token`, `mean`, `cls`
  - `last_token`: takes hidden state at the last non-padding position — fastest, original default
  - `mean`: attention-masked mean over all token positions — most robust, recommended for publication (fixes "oxidane bug" — 3-token molecules get noisy last-token representations)
  - `cls`: first token (BOS position) — useful for comparison experiments
- Outputs a `ToxGuardOutput` dataclass containing:
  - `binary_logits` — raw logit (B,)
  - `toxicity_score` — sigmoid(binary_logit), P(toxic)
  - `hidden_state` — the 256-dim egnn_vector from `egnn_projection`, for Phase 2
  - `attentions` — per-layer attention weights if requested (for interpretability)

**ToxGuardLitModel:**
- PyTorch Lightning wrapper around `ToxGuardModel`
- Handles training/validation/test loops
- Tracks `AUROC` and `AveragePrecision` metrics (from `torchmetrics`)
- Configures the optimizer: AdamW with `betas=(0.9, 0.999)` only on `requires_grad=True` parameters (only LoRA + heads)
- LR schedule: `LinearLR` warmup → `CosineAnnealingLR` decay, combined with `SequentialLR`

---

### `lora.py` — LoRA Implementation

Custom LoRA implementation (not using PEFT library) for precise control.

**`LoRALayer`** wraps any `nn.Module` (specifically GPT-2's `Conv1D` layers):
```python
result = original_layer(x)                               # frozen path
lora_out = dropout(x) @ lora_A @ lora_B * (alpha / r)  # trainable path
output = result + lora_out
```
- `lora_A` initialized with Kaiming uniform (gives good gradient flow)
- `lora_B` initialized to **zero** — so at initialization, LoRA contributes nothing and training starts from the intact pre-trained model

**`fan_in_fan_out=True`:** GPT-2 uses `Conv1D` (not `nn.Linear`). The weight matrix is stored transposed: shape is `(d_in, d_out)` instead of `(d_out, d_in)`. This flag tells LoRALayer to read dimensions correctly.

**`save_lora_weights`:** Saves only `requires_grad=True` parameters — approximately 4MB vs ~30MB for the full model checkpoint. This compact format is what gets stored in `lora_weights.pt`.

**`merge` / `unmerge`:** Can fold LoRA weights directly into the base model weights for faster inference (eliminates the extra matrix multiplications). Not used by default in ToxGuard but available.

---

### `data_pipeline.py` — Dataset Classes and Splitting

**`MoleculeDataset`:** Generic PyTorch `Dataset` for any `_final.csv` file. At `__getitem__`:
1. Gets the IUPAC name
2. Tokenizes it: `input_ids = tokenizer(name)["input_ids"]`
3. Prepends BOS token
4. Returns `{input_ids, attention_mask, binary_labels}` as tensors

**`T3DBDataset`:** Subclass with `all_toxic=True` — forces all labels to 1 regardless of the CSV's `is_toxic` column (since T3DB is an all-toxin database).

**`ToxicityDataset`:** Concatenation of multiple datasets. Implements cumulative indexing so `dataset[i]` correctly dispatches to the right sub-dataset.

**`ToxicityCollator`:** Custom collate function for variable-length sequences. Uses `pad_sequence` to pad all sequences in a batch to the same length (padding with `pad_token_id`). This is necessary because different IUPAC names tokenize to different lengths.

**Scaffold splitting (Bemis-Murcko):**
1. Computes Murcko scaffold for each molecule's SMILES
2. Groups molecules by scaffold
3. Assigns entire scaffold groups to train/val/test simultaneously (no molecule from the same scaffold is ever in two splits)
4. Uses a balance score to maintain the global toxic ratio across splits during assignment
5. Falls back to stratified random split if RDKit is unavailable

**Why is this better than random splitting?** Imagine your test set contains the same drug scaffold (ring system) as training set molecules. The model can generalize trivially by scaffold memory. Scaffold split ensures test compounds have structurally novel scaffolds — a much harder and more realistic evaluation.

---

### `inference.py` — ToxGuardPredictor

The unified inference API. Two main methods:

**`predict(iupac_name)`** — Single molecule:
1. Tokenizes the name, prepends BOS
2. Runs `model.forward()` with `torch.no_grad()`
3. Extracts `binary_prob = sigmoid(binary_logit)`
4. If `return_egnn_vector=True`, extracts the 256-dim `egnn_vector` from `hidden_state`
5. If `return_attention=True`, computes attention attribution via `interpretability.py`
6. Returns `ToxGuardPrediction` dataclass

**`predict_batch(iupac_names)`** — Multiple molecules:
- Tokenizes all inputs, pads into batches
- Runs single forward pass per batch (10-100× faster than sequential)
- Returns list of `ToxGuardPrediction` objects

**`get_egnn_vectors(iupac_names)`** — Phase 2 interface:
- Returns `(N, 256)` tensor of molecular representations
- This is the primary output consumed by Phase 2 EGNN as node features

The `ToxGuardPrediction` dataclass captures:
- `is_toxic` (bool) — binary classification result
- `toxicity_score` (float) — raw P(toxic) from sigmoid
- `severity_label` (str) — one of 5 bands
- `confidence` (float) — same as `toxicity_score` (P itself is the confidence)
- `egnn_vector` (List[float]) — 256-dim Phase 2 input
- `token_attributions`, `top_tokens`, `toxicophore_hits` — interpretability outputs

---

### `interpretability.py` — Attention Attribution

Provides post-hoc interpretability by extracting attention weights from all 8 transformer layers.

**`compute_attention_token_scores`:**
- Stacks all layer attention tensors: shape `(8_layers, B, 8_heads, L, L)`
- Averages across layers (or takes last layer only)
- Averages across attention heads
- For the chosen pooling position (last_token/mean/cls), reads the column of attention weights — "which tokens was this position attending to?"
- Normalizes to sum to 1 → per-token attribution score

**`build_token_attribution`:**
- Converts token IDs back to readable subword strings
- Removes special tokens (BOS, EOS, PAD) for clean display
- Returns top-K attended tokens

**`detect_toxicophore_attention`:**
- Scans sliding windows of 1-3 tokens for known toxicophore patterns: `"nitro"`, `"chloro"`, `"epoxy"`
- For each pattern hit, sums the attention scores of the matched span
- Returns hits sorted by attention score — the model's implicit focus on toxic fragments

**`save_attention_heatmap`:**
- Generates a 1xL color heatmap (magma colormap) saved as PNG
- X-axis = IUPAC subword tokens, color intensity = attention score
- Allows visual inspection of which parts of the IUPAC name drove the toxicity prediction

---

### `calibration.py` — Temperature Scaling

Post-hoc calibration following Guo et al. (ICML 2017):

```
P_calibrated = sigmoid(logit / T)
```

`T` is learned by minimizing NLL on the validation set using L-BFGS (a second-order optimizer — appropriate since only one scalar parameter is being optimized).

- `T > 1` → softens predictions (model was overconfident)
- `T < 1` → sharpens predictions (model was underconfident)
- `T = 1` → no change

This makes the numeric output `P(toxic)` match actual empirical frequencies — essential for the downstream RL reward function in Phase 4 which uses these probability scores directly.

---

## How Phase 1 Interacts with Phases 2, 3, and 4

### → Phase 2 (EGNN — 3D Graph Toxicity Model)

**The bridge:** The `egnn_vector` (256-dim) from `inference.py`'s `get_egnn_vectors()`.

Phase 2 builds a 3D molecular graph using EGNN (Equivariant Graph Neural Network) which processes atomic coordinates. The **node features** for this graph are enriched by Phase 1's text-based molecular representation:

```python
# In Phase 2, for each molecule:
egnn_input = predictor.get_egnn_vectors([iupac_name])  # shape (1, 256)
# This 256-dim vector becomes the initial node feature for all atoms in the EGNN graph
node_features = torch.cat([atom_features_3d, egnn_input.expand(n_atoms, -1)], dim=-1)
```

This is a **cross-modal fusion**: text understanding from Phase 1 + 3D geometry from Phase 2. The `egnn_projection` layer in `ToxGuardMultiTaskHead` was specifically designed for this role — it projects the pooled hidden state through a `Linear → LayerNorm` to prepare it as a compatible feature vector for the EGNN. The model knows, while training, that this vector will be a downstream input — hence the dedicated projection layer rather than just reusing the raw pooled state.

### → Phase 3 (RAG — Retrieval-Augmented Generation)

**The bridge:** Phase 1's predictions + `egnn_vector` as keys for embedding-based retrieval.

Phase 3 implements a Retrieve-Then-Explain pipeline:
1. Given a query molecule's IUPAC name, Phase 1 produces:
   - `P(toxic)`, `severity_label`, `egnn_vector`
2. The `egnn_vector` is used as an embedding key to find similar molecules from a precomputed knowledge base
3. The retrieved similar molecules + their experimentally validated toxicity information are injected into a RAG-style prompt
4. An LLM uses this context to generate a **chain-of-thought explanation** of *why* the molecule is predicted toxic

Phase 1's representation quality directly determines the quality of retrieval — if similar molecules cluster tightly in embedding space, retrieved neighbors will be chemically meaningful.

### → Phase 4 (RL — PPO-Based Molecule Detoxification)

**The bridge:** Phase 1 as the **reward model** for the RL detoxification agent.

Phase 4 uses PPO (Proximal Policy Optimization) to modify molecules to reduce toxicity. The RL loop:

1. **Agent** proposes a structural modification: e.g., replace `-NO₂` (nitro) with `-NH₂` (amino)
2. The modified molecule's IUPAC name is computed
3. Phase 1's `predictor.predict(modified_iupac)` is called → returns `P(toxic)`
4. **Reward** = reduction in `P(toxic)` relative to the original: `reward = P_original - P_modified`
5. Agent is updated via PPO to maximize reward

Phase 1 must be:
- **Fast** (inference is called thousands of times per RL episode) — hence the batch prediction API
- **Well-calibrated** (temperature scaling ensures rewards accurately reflect true toxicity reduction, not model overconfidence artifacts)
- **Chemically meaningful** (the embeddings must capture structure-activity relationships so the RL agent can generalize learned detoxification strategies)

> **Critical dependency:** Phase 4 specifically uses lora_weights.pt from Phase 1's trained run. If Phase 1's AUC-ROC is poor, Phase 4 will receive a noisy reward signal and fail to learn effective detoxification.

---

## Design Decisions Worth Understanding

### Why IUPAC Names Instead of SMILES?

SMILES is machine-native but syntactically arbitrary — the same molecule has many valid SMILES strings, and the character-level representation has no semantic meaning. IUPAC names, by contrast, are:
- **Systematic**: `2-chloro-1,3-butadiene` = chlorine at carbon 2 of a 1,3-butadiene
- **Readable by domain experts**: alignment between model input and human reasoning
- **Uniquely tokenizable**: IUPACGPT's SentencePiece tokenizer was trained specifically on IUPAC names, giving meaningful subword units like `"chloro"`, `"nitro"`, `"butyl"` — chemical fragments that carry real toxicological meaning

This is why the attention maps over IUPAC subwords are interpretable — when the model attends to `"nitro"` in the IUPAC name, it's genuinely attending to the nitro functional group.

### Why Not Use the T3DB in Training?

T3DB contains 99.2% toxic compounds. Including it would:
1. Shift the training distribution so far toward toxic that the model over-predicts toxicity
2. Contaminate the external validation signal — you can't use T3DB as a held-out recall test if you trained on it

It is specifically reserving T3DB as a **recall audit**: "out of all known toxins, how many does the model catch?" This is the most safety-critical metric — missing toxins is more dangerous than false positives.

### Why Custom LoRA Instead of PEFT?

The `lora.py` implementation is hand-written rather than using HuggingFace PEFT. Reasons:
1. **GPT-2's `Conv1D` layers**: PEFT's LoRA assumes `nn.Linear`. GPT-2 uses `Conv1D` which stores weights transposed — the `fan_in_fan_out=True` flag handles this correctly in the custom implementation
2. **Full control**: Custom implementation makes it easy to inspect parameter counts, merge/unmerge weights, and save only LoRA parameters (not the full model)
3. **Portability**: The `lora_weights.pt` file is completely self-contained and loadable without PEFT dependency

---

## Step-by-Step Execution Order

```bash
# From the ToxGuard project root
python Phase1-IUPACGPT/steps/step1_download_data.py     # ~5-20 min (network-dependent)
python Phase1-IUPACGPT/steps/step2_preprocess.py        # ~2-5 min
python Phase1-IUPACGPT/steps/step3_smiles_to_iupac.py   # ~2-8 hours (API rate limiting!)
python Phase1-IUPACGPT/steps/step4_verify_lora.py       # <1 min
python Phase1-IUPACGPT/steps/step5_train.py             # ~30-120 min (GPU-dependent)
python Phase1-IUPACGPT/steps/step6_evaluate.py          # ~5-15 min
python Phase1-IUPACGPT/steps/step7_predict.py           # instant
```

Step 3 is the bottleneck — PubChem rate limits to 5 req/s, so resolving 20K+ unique SMILES takes hours. Use `--limit 1000` to test the pipeline first, and the persistent cache means you can re-run without re-querying already-resolved molecules.

---

## Summary

Phase 1 is a complete, production-grade pipeline that:

1. **Acquires** 6 complementary toxicity datasets (ToxCast, Tox21, T3DB, hERG, DILI, Common Molecules)
2. **Cleans** them with RDKit canonicalization, cross-dataset deduplication, and binary label fusion
3. **Names** every molecule systematically via a 3-API cascade with stereo-aware collision resolution
4. **Fine-tunes** a chemical-language GPT-2 model using LoRA for parameter-efficient binary toxicity classification
5. **Evaluates** comprehensively with calibrated probabilities, tuned thresholds, external validation, and interpretable attention maps
6. **Exports** a 256-dimensional molecular embedding that feeds into Phase 2 (EGNN), provides retrieval keys for Phase 3 (RAG), and serves as the reward model for Phase 4 (RL)

The defining philosophy: **language models understand chemistry through IUPAC names** — and that understanding can be efficiently redirected toward safety-critical toxicity prediction with only ~14% of parameters actually updated.
