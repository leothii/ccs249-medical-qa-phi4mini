"""
CCS 249 Final Project — Medical QA Chatbot Backend
West Visayas State University — CICT
Flask API server for the fine-tuned Phi-4 Mini MedQuAD model.

Usage:
    python app.py

Endpoints:
    POST /chat       — Generate a medical answer
    POST /summarize  — Summarize a block of medical text
    GET  /health     — Health check
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import os
import re

app = Flask(__name__)
CORS(app)  # Allow requests from the HTML frontend

# ─── Model Configuration ──────────────────────────────────────────────────────
MODEL_PATH    = "./outputs/phi4-medical-final"   # Merged fine-tuned model
LORA_PATH     = "./outputs/phi4-medical-lora"    # LoRA adapters (fallback)
BASE_MODEL    = "microsoft/Phi-4-mini-instruct"
MAX_SEQ_LEN   = 512
MAX_NEW_TOKENS = 250

SYSTEM_MSG = (
    "You are a helpful medical assistant trained on NIH (National Institutes of Health) data. "
    "Answer medical questions accurately and clearly. "
    "Always remind users to consult a licensed healthcare professional for personal medical advice."
)

# ─── Load Model ───────────────────────────────────────────────────────────────
print("=" * 60)
print(" CCS 249 Medical QA — Loading model...")
print("=" * 60)

MODEL_BACKEND = "demo"

try:
    from unsloth import FastLanguageModel

    # Try merged model first, fall back to base + LoRA
    load_path = MODEL_PATH if os.path.exists(MODEL_PATH) else BASE_MODEL

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = load_path,
        max_seq_length = MAX_SEQ_LEN,
        dtype          = None,
        load_in_4bit   = True,
    )

    # Load LoRA adapters if using base model
    if load_path == BASE_MODEL and os.path.exists(LORA_PATH):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, LORA_PATH)
        print("  LoRA adapters loaded from:", LORA_PATH)

    FastLanguageModel.for_inference(model)
    print(f"  Model loaded from : {load_path}")
    print(f"  Device            : {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"  GPU               : {torch.cuda.get_device_name(0)}")
    MODEL_LOADED = True
    MODEL_BACKEND = "unsloth"

except Exception as unsloth_error:
    print(f"\n  ⚠️  Unsloth load failed: {unsloth_error}")
    print("  Trying transformers fallback...")

    try:
        from typing import TypedDict
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import transformers.utils as tf_utils

        # Compatibility shim for some Phi remote-code versions.
        if not hasattr(tf_utils, "LossKwargs"):
            class _LossKwargs(TypedDict, total=False):
                pass
            tf_utils.LossKwargs = _LossKwargs

        load_path = MODEL_PATH if os.path.exists(MODEL_PATH) else BASE_MODEL
        tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=False)

        if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs = {
            "trust_remote_code": False,
            "low_cpu_mem_usage": True,
        }

        if torch.cuda.is_available():
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = torch.float32

        model = AutoModelForCausalLM.from_pretrained(load_path, **model_kwargs)

        if not torch.cuda.is_available():
            model.to("cpu")

        # Optional LoRA adapter load for base model fallback
        if load_path == BASE_MODEL and os.path.exists(LORA_PATH):
            try:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, LORA_PATH)
                print("  LoRA adapters loaded from:", LORA_PATH)
            except Exception as lora_error:
                print(f"  ⚠️  Could not load LoRA adapters: {lora_error}")

        model.eval()
        MODEL_LOADED = True
        MODEL_BACKEND = "transformers"
        print(f"  Model loaded from : {load_path}")
        print("  Backend           : transformers")
        print(f"  Device            : {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        if torch.cuda.is_available():
            print(f"  GPU               : {torch.cuda.get_device_name(0)}")

    except Exception as fallback_error:
        print(f"\n  ⚠️  Could not load model with transformers: {fallback_error}")
        print("  Running in DEMO mode — responses will be placeholder text.")
        model, tokenizer = None, None
        MODEL_LOADED = False
        MODEL_BACKEND = "demo"

print("=" * 60)


def _model_device() -> torch.device:
    """Return a valid device for input tensors."""
    if model is None:
        return torch.device("cpu")
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Inference Helper ─────────────────────────────────────────────────────────
def generate_answer(question: str, max_tokens: int = MAX_NEW_TOKENS) -> dict:
    """Run inference on the fine-tuned Phi-4 Mini model."""

    if not MODEL_LOADED:
        # Demo fallback — useful for UI testing without a GPU
        return {
            "answer": (
                f"[DEMO MODE — model not loaded]\n\n"
                f"Your question was: \"{question}\"\n\n"
                "To get real answers, run this server on the machine where your "
                "fine-tuned model is saved (./outputs/phi4-medical-final or ./outputs/phi4-medical-lora).\n\n"
                "⚠️ This system is for educational purposes only. Always consult a "
                "licensed healthcare professional for personal medical advice."
            ),
            "confidence": 0,
            "source": "Demo Mode",
        }

    prompt = (
        f"<|system|>\n{SYSTEM_MSG}<|end|>\n"
        f"<|user|>\n{question}<|end|>\n"
        f"<|assistant|>\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(_model_device())

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens      = max_tokens,
            temperature         = 0.7,
            do_sample           = True,
            top_p               = 0.9,
            repetition_penalty  = 1.1,
            pad_token_id        = tokenizer.eos_token_id,
        )

    answer = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()

    # Confidence heuristic: longer + non-hedging = higher confidence
    hedge_words = ["i'm not sure", "i don't know", "cannot", "unclear", "may or may not"]
    hedge_penalty = sum(1 for w in hedge_words if w in answer.lower()) * 8
    length_score  = min(len(answer.split()) / 80, 1.0) * 40
    confidence    = max(20, min(98, 55 + length_score - hedge_penalty))

    return {
        "answer"    : answer,
        "confidence": round(confidence),
        "source"    : "MedQuAD (NIH) — Fine-tuned Phi-4 Mini",
    }


def generate_summary(text: str) -> str:
    """Summarize a medical text passage."""
    if not MODEL_LOADED:
        return "[DEMO MODE] Summary unavailable without the loaded model."

    prompt = (
        f"<|system|>\n"
        "You are a medical summarization assistant. "
        "Summarize the provided medical text clearly and concisely in 3–5 sentences. "
        "Preserve all key medical facts and recommendations."
        "<|end|>\n"
        f"<|user|>\nSummarize the following medical text:\n\n{text}<|end|>\n"
        f"<|assistant|>\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(_model_device())

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens     = 200,
            temperature        = 0.4,
            do_sample          = True,
            top_p              = 0.85,
            repetition_penalty = 1.1,
            pad_token_id       = tokenizer.eos_token_id,
        )

    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()


def generate_related_questions(question: str, answer: str) -> list:
    """Generate 3 follow-up questions using simple templates."""
    q_lower = question.lower()
    templates = {
        "symptom": ["How is {topic} treated?", "What causes {topic}?", "Types of {topic}"],
        "treat"  : ["What are the side effects of {topic} treatment?",
                    "How long does {topic} treatment take?",
                    "Are there alternatives to {topic} treatment?"],
        "cause"  : ["What are the symptoms of {topic}?",
                    "How is {topic} diagnosed?",
                    "Can {topic} be prevented?"],
        "default": ["How is this condition diagnosed?",
                    "What are the treatment options?",
                    "How can this be prevented?"],
    }

    if   any(w in q_lower for w in ["symptom", "sign", "feel"]):              key = "symptom"
    elif any(w in q_lower for w in ["treat", "cure", "medication", "drug"]):  key = "treat"
    elif any(w in q_lower for w in ["cause", "why", "risk"]):                 key = "cause"
    else:                                                                     key = "default"

    stop  = {"what", "how", "why", "when", "where", "is", "are", "does", "do",
             "can", "the", "a", "an", "of", "for", "about"}
    words = [w for w in re.findall(r"\b[A-Za-z]+\b", question) if w.lower() not in stop]
    topic = " ".join(words[:2]) if words else "this condition"

    return [t.format(topic=topic) for t in templates[key][:3]]


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status"      : "ok",
        "model_loaded": MODEL_LOADED,
        "backend"     : MODEL_BACKEND,
        "model_path"  : MODEL_PATH if os.path.exists(MODEL_PATH) else BASE_MODEL,
        "gpu"         : torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    })


@app.route("/chat", methods=["POST"])
def chat():
    data     = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()

    if not question:               return jsonify({"error": "Missing 'question' field."}), 400
    if len(question) > 1000:       return jsonify({"error": "Question too long (max 1000 characters)."}), 400

    result  = generate_answer(question)
    related = generate_related_questions(question, result["answer"])

    return jsonify({
        "answer"           : result["answer"],
        "confidence"       : result["confidence"],
        "source"           : result["source"],
        "related_questions": related,
    })


@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if not text:           return jsonify({"error": "Missing 'text' field."}), 400
    if len(text) > 5000:   return jsonify({"error": "Text too long (max 5000 characters)."}), 400

    return jsonify({"summary": generate_summary(text)})


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n Starting Flask server on http://localhost:5000")
    print(" Open index.html in your browser to use the chatbot.\n")
    print(" Note: running on port 8000 to match Kaggle notebook and tunnel.")
    app.run(host="0.0.0.0", port=8000, debug=False)
