# ToxGuard

An end-to-end multi-agent pipeline for molecular toxicity prediction, mechanistic reasoning, and automated detoxification.

ToxGuard takes IUPAC chemical names as input and runs a comprehensive 4-phase analysis to predict toxicity, explain the mechanism of action, compile a detailed Safety Data Sheet (SDS) using Retrieval-Augmented Generation, and optionally propose safer molecular analogs via Reinforcement Learning.

---

## Quick Start: Running the Final Project

To run the complete end-to-end 4-phase pipeline (Phase 1 Prediction → Phase 2 CoT → Phase 3 RAG Synthesis → Phase 4 RL) and generate a comprehensive safety data sheet, use `main.py` from the project root directory:

```bash
# Run the full pipeline and print the formatted Safety Data Sheet
python main.py "nitrobenzene" -v

# Run for multiple molecules
python main.py "nitrobenzene" "ethanol" "sodium azide" -v

# Run with Phase 4 (RL Detoxification) enabled
python main.py "nitrobenzene" --detox -v
```

**Running without Phase 1 (CPU-only / Standalone Logic):**
If you do not have PyTorch/CUDA or the Phase 1 model weights installed, you can skip the predictive model. The system will assign a default dummy score (0.5) and immediately trigger the Phase 2 LLM reasoning and Phase 3 RAG synthesis.

```bash
python main.py "nitrobenzene" -v --no-phase1
```

---

## Setup & Requirements

### 1. Clone the repository

```bash
git clone https://github.com/P-Vishnupranav-Reddy/Toxgaurd.git
cd Toxgaurd
```

### 2. Create and activate a virtual environment

```bash
python -m venv toxguard_env

# Windows
toxguard_env\Scripts\activate

# Linux / macOS
source toxguard_env/bin/activate
```

### 3. Install PyTorch with CUDA (Recommended)

Visit [https://pytorch.org/get-started/locally](https://pytorch.org/get-started/locally) and select your OS, CUDA version, and package manager. Example for CUDA 11.8:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

> **Important:** Do not run `pip install torch` from `requirements.txt` blindly if you intend to use GPU — it often defaults to the CPU-only version.

### 4. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 5. API Keys Configuration

For Phase 2 (CoT) and Phase 3 (RAG) to function correctly, you need to provide a Groq API key (or specify another compatible LLM provider in the source configs). Add an `.env` file at the root:

```ini
GROQ_API_KEY=your_api_key_here
```
Or continuously pass it via the command line:
```bash
python main.py "nitrobenzene" --api-key your_api_key_here
```

---

## Project Structure & Architecture

```
ToxGuard/
├── main.py                       # Main Entry Point (End-to-End Pipeline)
├── Phase1-IUPACGPT/              # Predictive Modeling with LoRA
│   ├── steps/                    # Data processing & model training scripts
│   ├── iupacGPT_finetune/        # Model and LoRA implementations
│   └── iupacGPT_outputs/         # Trained checkpoints & outputs
├── Phase2-CoT/                   # Chain-of-Thought Reasoning Analysis
│   ├── llm_client.py             # LLM API handlers
│   ├── cot_analyzer.py           # Multi-step mechanistic reasoning
│   └── prompts.py                # Reasoning prompt templates
├── Phase3-RAG/                   # Retrieval-Augmented Generation
│   ├── rag_pipeline.py           # Synthesis of CoT and vector records
│   ├── retriever.py              # ChromaDB interactions
│   ├── safety_profile.py         # Structuring the Safety Data Sheet
│   └── eval_results/             # Phase 3 RAG benchmark metrics
├── Phase4-RL/                    # Reinforcement Learning Detoxification
│   ├── run_detox.py              # RL Detox Pipeline runner
│   ├── rl_trainer.py             # PPO algorithm
│   ├── detox_agent.py            # Transformation & validation logic
│   └── outputs/                  # Phase 4 generated RL results
├── data/                         # Raw and preprocessed dataset CSVs
├── Research_Papers/              # Related literature and research
└── iupacGPT/                     # Pretrained IUPACGPT base
```

### The 4 Analytical Phases:
1. **Phase 1: Prediction** (IUPAC name → GPT-2 + LoRA → P(toxic) & severity label)
2. **Phase 2: Mechanistic Reasoning** (LLM unpacks structure into toxicophores & pathways)
3. **Phase 3: RAG Synthesis** (Retrieves analogous compounds to compile a multi-section SDS)
4. **Phase 4: Detoxification** (RL Agent proposes and validates structural modifications for safer variants)

---

## Manual Pipeline Execution (Phase 1 Training)

If you wish to train the predictive model yourself, navigate to `Phase1-IUPACGPT` and run the numbered steps in order from the root directory:

1. **Download raw data**: `python Phase1-IUPACGPT/steps/step1_download_data.py`
2. **Preprocess datasets**: `python Phase1-IUPACGPT/steps/step2_preprocess.py`
3. **Resolve SMILES to IUPAC**: `python Phase1-IUPACGPT/steps/step3_smiles_to_iupac.py`
4. **Inject LoRA**: `python Phase1-IUPACGPT/steps/step4_verify_lora.py`
5. **Train**: `python Phase1-IUPACGPT/steps/step5_train.py`
6. **Evaluate on test set**: `python Phase1-IUPACGPT/steps/step6_evaluate.py`
7. **Stand-alone predictions**: `python Phase1-IUPACGPT/steps/step7_predict.py`

### Running Phases Independently

You can test Phase 2 (CoT), Phase 3 (RAG) or Phase 4 (RL) manually by providing a synthetic pseudo-toxicity score:

**Standalone Phase 2 (CoT Mechanistic Reasoning):**
```bash
python Phase2-CoT/run_cot.py "nitrobenzene" --score 0.91 -v
```

**Standalone Phase 3 (RAG Safety Synthesis):**
```bash
python Phase3-RAG/run_rag.py "nitrobenzene" --score 0.91 -v
```

**Standalone Phase 4 (RL Detoxification):**
```bash
python Phase4-RL/run_detox.py detox "nitrobenzene" --score 0.91 -v
```

---

## Datasets

The core Phase 1 model is trained and evaluated on an aggregated, deduplicated set of labeled toxicological data spanning thousands of compounds:

| Dataset | Description / Task |
|---------|--------------------|
| ToxCast | Multi-assay binary label predictions |
| hERG | Cardiotoxicity and potassium channel blockage |
| T3DB | Broad database of known toxins |
| Tox21 | Multi-assay panel testing |
| DILI | Drug-induced liver injury associations |
| Common molecules | Curated safe/unsafe baseline set |

---

## License

This project incorporates the [IUPACGPT](https://github.com/iupacgpt/iupac-gpt) pretrained language model as its foundational Phase 1 engine. Please refer to its original license before considering commercial use or distribution.
