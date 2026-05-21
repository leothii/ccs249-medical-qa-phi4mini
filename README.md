# MedAI — Medical QA Chatbot

**CCS 249 Final Project · West Visayas State University, CICT**

A medical question-answering chatbot powered by **Microsoft Phi-4 Mini** (3.8B) fine-tuned on the **NIH MedQuAD** dataset using **LoRA** and the **Unsloth** framework. The project includes a full-stack web application with a liquid-glass UI, streaming responses, and a medical text summarizer.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Running the App](#running-the-app)
- [API Reference](#api-reference)
- [Tech Stack](#tech-stack)
- [Disclaimer](#disclaimer)

---

## Overview

MedAI fine-tunes **Phi-4 Mini Instruct** on medical Q&A pairs from the National Institutes of Health to produce accurate, source-grounded answers to clinical and patient-facing questions. The system runs as a local Flask server with an interactive web front-end and degrades gracefully to **Demo Mode** when no GPU is available.

---

## Features

| Feature | Description |
|---|---|
| **Medical Q&A Chat** | Ask any medical question and receive a detailed, NIH-grounded answer |
| **Streaming Responses** | Server-Sent Events (SSE) deliver tokens in real time |
| **Document Summarizer** | Paste clinical text (up to 5,000 chars) for a concise 3–5 sentence summary |
| **Urgent Symptom Detection** | Automatically flags emergency keywords and surfaces a safety warning |
| **Suggested Questions** | Sidebar with curated starter prompts |
| **Related Questions** | Each answer generates 3 follow-up question suggestions |
| **Chat History & Bookmarks** | Session-scoped history and saved-answer panels |
| **Export Chat** | Download the current conversation as a file |
| **Customizable UI** | Light / Dark theme, adjustable glass blur intensity |
| **Demo Mode** | Placeholder responses when the model cannot be loaded |
| **Health Endpoint** | `/health` reports backend, GPU, device map, and LoRA status |

---

## Model Architecture

| Property | Value |
|---|---|
| **Base model** | `microsoft/Phi-4-mini-instruct` |
| **Architecture** | Phi-3 / Phi3ForCausalLM |
| **Parameter count** | 3.8 B |
| **Hidden size** | 3,072 |
| **Hidden layers** | 32 |
| **Attention heads** | 24 (GQA — 8 KV heads) |
| **Vocabulary** | 200,064 tokens |
| **Max context** | 131,072 tokens (LongRoPE) |
| **Quantization** | 4-bit NF4 (BitsAndBytes) |
| **Fine-tuning method** | LoRA (PEFT) via Unsloth |
| **LoRA rank** | 16 |
| **LoRA alpha** | 32 |
| **LoRA dropout** | 0.05 |
| **LoRA target modules** | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |

---

## Dataset

**MedQuAD — NIH Medical Question Answer Dataset**

A curated collection of medical Q&A pairs sourced from the National Institutes of Health (NIH), covering conditions, symptoms, treatments, genetics, and clinical trials. The dataset spans a wide range of rare and common diseases drawn from authoritative NIH pages (MedlinePlus, GARD, NINDS, NCI, etc.).

---

## Training

Training was performed on **Kaggle** using the Unsloth-optimized Phi-4 Mini 4-bit checkpoint.

| Hyperparameter | Value |
|---|---|
| **Epochs** | 3 |
| **Total steps** | 3,567 |
| **Batch size** | 2 |
| **Max sequence length** | 512 |
| **Optimizer** | AdamW with cosine LR schedule |
| **Initial learning rate** | 2 × 10⁻⁴ |
| **Evaluation interval** | Every 500 steps |
| **Checkpoint saves** | Every 500 steps |
| **Framework** | Unsloth + Hugging Face TRL (SFT) |

### Training Loss Curve

| Step | Train Loss | Eval Loss |
|---|---|---|
| 50 | 1.708 | — |
| 500 | 0.926 | 1.840 |
| 1000 | 0.868 | 1.770 |
| 1500 | 0.842 | 1.731 |
| 2000 | 0.794 | 1.705 |
| 2500 | 0.778 | 1.701 |
| 3000 | 0.755 | 1.698 |
| 3500 | 0.733 | 1.694 |
| **3567 (final)** | **0.703** | **1.694** |

The best checkpoint was saved at **step 3567** (epoch 3.0) with an evaluation loss of **1.694**.

---

## Evaluation

Model outputs were evaluated against MedQuAD reference answers using **BLEU** and **ROUGE** metrics.

| Metric | Description |
|---|---|
| **Perplexity** | `exp(eval_loss)` — measures how confidently the model predicts the next token; lower is better |
| **BLEU** | Measures n-gram precision between generated and reference answers |
| **ROUGE-1** | Unigram overlap (recall-oriented) |
| **ROUGE-2** | Bigram overlap |
| **ROUGE-L** | Longest common subsequence similarity |

### Perplexity over Training

Perplexity is computed directly from the evaluation cross-entropy loss at each checkpoint (`PPL = e^eval_loss`).

| Checkpoint (step) | Eval Loss | Perplexity |
|---|---|---|
| 500 | 1.840 | 6.30 |
| 1000 | 1.770 | 5.87 |
| 1500 | 1.731 | 5.65 |
| 2000 | 1.705 | 5.50 |
| 2500 | 1.701 | 5.48 |
| 3000 | 1.698 | 5.46 |
| 3500 | 1.694 | 5.44 |
| **3567 (best)** | **1.694** | **5.44** |

Perplexity dropped from **6.30** at the first evaluation to **5.44** at convergence, indicating the model became noticeably more confident and consistent in generating medical text over the 3 training epochs.

Full per-question BLEU / ROUGE scores are available in [`data/evaluation_results.csv`](data/evaluation_results.csv). Sample results show the model performs best on structured reference answers (e.g., symptom lists and inheritance patterns) and more variably on open-ended explanatory questions.

---

## Project Structure

```
ccs249-medical-qa-phi4mini/
│
├── app.py                          # Flask backend — model loading, inference, API routes
│
├── frontend/
│   ├── index.html                  # Main chat application (liquid-glass UI)
│   ├── splashscreen.html           # Welcome screen with disclaimer flow
│   ├── styles.css                  # Global stylesheet
│   └── assets/
│       └── logo-avatar.png         # MedAI brand logo
│
├── models/
│   ├── phi4-medical-lora/          # Final LoRA adapter weights (loaded at runtime)
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors
│   │   └── tokenizer.json
│   ├── phi4-medical-final/         # Merged full model (optional; large)
│   └── phi4-medical-qa/            # Training checkpoints
│       ├── checkpoint-3500/
│       └── checkpoint-3567/        # Best checkpoint
│
├── data/
│   └── evaluation_results.csv      # BLEU & ROUGE scores per question
│
├── reports/
│   ├── baseline_comparison.png     # Fine-tuned vs baseline metric comparison
│   └── dataset_distribution.png   # MedQuAD dataset category distribution
│
└── runtime/
    └── unsloth_compiled_cache/     # Unsloth JIT-compiled kernel cache
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU recommended (runs in CPU / Demo Mode otherwise)
- 6 GB+ VRAM for 4-bit inference

### 1. Clone the repository

```bash
git clone <repo-url>
cd ccs249-medical-qa-phi4mini
```

### 2. Create a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install flask flask-cors torch transformers peft unsloth waitress
```

> **Note:** Install the correct PyTorch version for your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/).
> For Unsloth, follow the [official install guide](https://github.com/unslothai/unsloth).

### 4. Place model weights

Ensure the LoRA adapter directory exists at:

```
models/phi4-medical-lora/
  adapter_config.json
  adapter_model.safetensors
  tokenizer.json
  tokenizer_config.json
```

The base model (`microsoft/Phi-4-mini-instruct`) is downloaded automatically from Hugging Face on first run.

---

## Running the App

```bash
python app.py
```

The server starts on **http://localhost:8000**.

| URL | Description |
|---|---|
| `http://localhost:8000` | MedAI splash screen |
| `http://localhost:8000/app` | Main chat interface |
| `http://localhost:8000/health` | Backend health check (JSON) |

> **Waitress** is used as the production WSGI server when installed, providing better handling of long-running generation requests. The app falls back to the Flask development server if Waitress is not available.

### Demo Mode

If the model fails to load (no GPU, missing weights, etc.), the app continues running in **Demo Mode** — the UI is fully functional but responses are placeholder text.

---

## API Reference

### `POST /chat`

Generate a medical answer (non-streaming).

**Request body:**
```json
{
  "question": "What is pneumonia and how is it treated?"
}
```

**Response:**
```json
{
  "answer": "Pneumonia is an infection that inflames...",
  "confidence": 72,
  "source": "MedQuAD (NIH) — Fine-tuned Phi-4 Mini",
  "related_questions": ["What causes pneumonia?", "..."],
  "urgent_warning": ""
}
```

---

### `POST /chat_stream`

Generate a medical answer with real-time token streaming (SSE).

**Request body:** Same as `/chat`.

**Response:** `text/event-stream`

```
data: {"delta": "Pneumonia"}
data: {"delta": " is an"}
...
event: done
data: {"answer": "...", "confidence": 72, "source": "...", "related": [...], "urgent_warning": ""}
```

---

### `POST /summarize`

Summarize a block of medical text.

**Request body:**
```json
{
  "text": "Patient presents with bilateral lower lobe consolidation..."
}
```

**Response:**
```json
{
  "summary": "The patient has bilateral lower lobe pneumonia..."
}
```

---

### `GET /health`

Check model status and hardware details.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "backend": "unsloth",
  "model_path": "microsoft/Phi-4-mini-instruct",
  "lora_path": "models/phi4-medical-lora",
  "gpu": "NVIDIA GeForce RTX ...",
  "cpu_offload": false,
  "device_map": {"cuda:0": 32}
}
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | Microsoft Phi-4 Mini Instruct (3.8B) |
| **Fine-tuning** | Unsloth + Hugging Face PEFT / TRL |
| **Quantization** | BitsAndBytes 4-bit NF4 |
| **Backend** | Python · Flask · Waitress |
| **Frontend** | Vanilla HTML · CSS (liquid-glass design) · JavaScript |
| **Streaming** | Server-Sent Events (SSE) |
| **Training platform** | Kaggle (GPU T4 / P100) |
| **Dataset** | MedQuAD — NIH Medical QA |

---

## Disclaimer

> MedAI is developed for **educational purposes** as an NLP course final project.
> It is **not** a substitute for professional medical advice, diagnosis, or treatment.
> Always consult a licensed healthcare professional for personal medical concerns.
> For urgent symptoms, contact your local emergency services immediately.
