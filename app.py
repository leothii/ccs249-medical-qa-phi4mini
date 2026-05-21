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

from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from flask_cors import CORS
import torch
import os
import re
import traceback
import json
import threading

# Avoid TorchInductor/Triton compile path on Windows when versions mismatch.
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

# Favor faster math on RTX-class GPUs.
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

app = Flask(__name__)
CORS(app)  # Allow requests from the HTML frontend
APP_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(APP_DIR, "frontend")

# ─── Model Configuration ──────────────────────────────────────────────────────
MODEL_PATH    = os.path.join(APP_DIR, "models", "phi4-medical-final")   # Merged fine-tuned model (unused)
LORA_PATH     = os.path.join(APP_DIR, "models", "phi4-medical-lora")     # LoRA adapters
BASE_MODEL    = "microsoft/Phi-4-mini-instruct"
MAX_SEQ_LEN   = 512
MAX_NEW_TOKENS = 250
MAX_SUMMARY_TOKENS = 200
MAX_GENERATION_SECONDS = 120

SYSTEM_MSG = (
    "You are a helpful medical assistant trained on NIH (National Institutes of Health) data. "
    "Answer medical questions accurately and clearly. "
    "Always remind users to consult a licensed healthcare professional for personal medical advice."
)

URGENT_PATTERN = re.compile(
    r"\b(chest pain|trouble breathing|difficulty breathing|shortness of breath|"
    r"stroke|face droop|fainting|seizure|suicidal|suicide|overdose|"
    r"severe bleeding|anaphylaxis|severe allergic|blue lips|"
    r"loss of consciousness|heart attack)\b",
    re.IGNORECASE,
)

URGENT_WARNING = (
    "Urgent safety note: your question may describe symptoms that need immediate care. "
    "If this is happening now, call your local emergency number or go to the nearest "
    "emergency department. MedAI can provide general information, but it cannot assess emergencies."
)


def load_local_tokenizer(load_path: str):
    """Load tokenizer, falling back when exported config names a non-importable class."""
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    try:
        return AutoTokenizer.from_pretrained(load_path, trust_remote_code=False)
    except Exception as tokenizer_error:
        tokenizer_file = os.path.join(load_path, "tokenizer.json")
        if not os.path.exists(tokenizer_file):
            raise tokenizer_error

        print(f"  Tokenizer fallback: loading tokenizer.json directly ({tokenizer_error})")
        return PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_file,
            bos_token="<|endoftext|>",
            eos_token="<|end|>",
            unk_token="<|endoftext|>",
            pad_token="<|end|>",
        )

# ─── Load Model ───────────────────────────────────────────────────────────────
print("=" * 60)
print(" CCS 249 Medical QA — Loading model...")
print("=" * 60)

MODEL_BACKEND = "demo"
MODEL_LOAD_ERROR = ""
MODEL_CPU_OFFLOAD = False
MODEL_DEVICE_MAP_SUMMARY = None

try:
    from unsloth import FastLanguageModel

    # Try merged model first, fall back to base + LoRA
    load_path = BASE_MODEL

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
    MODEL_LOAD_ERROR = f"Unsloth load failed: {unsloth_error}"
    print(f"\n  ⚠️  Unsloth load failed: {unsloth_error}")
    print("  Trying transformers fallback...")

    try:
        from typing import TypedDict
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import transformers.utils as tf_utils

        # Compatibility shim for some Phi remote-code versions.
        if not hasattr(tf_utils, "LossKwargs"):
            class _LossKwargs(TypedDict, total=False):
                pass
            tf_utils.LossKwargs = _LossKwargs

        load_path = BASE_MODEL
        tokenizer = load_local_tokenizer(load_path)

        if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs = {
            "trust_remote_code": False,
            "low_cpu_mem_usage": True,
        }

        if torch.cuda.is_available():
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["quantization_config"] = quant_config
            try:
                model_kwargs["device_map"] = {"": 0}
                model_kwargs["max_memory"] = {0: "6GiB"}
                model = AutoModelForCausalLM.from_pretrained(load_path, **model_kwargs)
            except RuntimeError as load_error:
                if "out of memory" in str(load_error).lower():
                    torch.cuda.empty_cache()
                    model_kwargs["device_map"] = "auto"
                    model_kwargs["max_memory"] = {0: "5GiB", "cpu": "24GiB"}
                    model = AutoModelForCausalLM.from_pretrained(load_path, **model_kwargs)
                else:
                    raise
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
        if hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict):
            summary = {}
            for dev in model.hf_device_map.values():
                summary[str(dev)] = summary.get(str(dev), 0) + 1
            MODEL_DEVICE_MAP_SUMMARY = summary
            MODEL_CPU_OFFLOAD = "cpu" in summary
            print(f"  Device map         : {summary}")
            if MODEL_CPU_OFFLOAD:
                print("  ⚠️  CPU offload detected — responses may be slow.")
        MODEL_LOADED = True
        MODEL_BACKEND = "transformers"
        MODEL_LOAD_ERROR = ""
        print(f"  Model loaded from : {load_path}")
        print("  Backend           : transformers")
        print(f"  Device            : {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        if torch.cuda.is_available():
            print(f"  GPU               : {torch.cuda.get_device_name(0)}")

    except Exception as fallback_error:
        MODEL_LOAD_ERROR = f"{MODEL_LOAD_ERROR} | Transformers fallback failed: {fallback_error}"
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


def _estimate_confidence(answer: str) -> int:
    # Confidence heuristic: longer + non-hedging = higher confidence
    hedge_words = ["i'm not sure", "i don't know", "cannot", "unclear", "may or may not"]
    hedge_penalty = sum(1 for w in hedge_words if w in answer.lower()) * 8
    length_score = min(len(answer.split()) / 80, 1.0) * 40
    return max(20, min(98, 55 + length_score - hedge_penalty))


def load_local_tokenizer(load_path: str):
    """Load tokenizer, falling back when exported config names a non-importable class."""
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    try:
        return AutoTokenizer.from_pretrained(load_path, trust_remote_code=False)
    except Exception as tokenizer_error:
        tokenizer_file = os.path.join(load_path, "tokenizer.json")
        if not os.path.exists(tokenizer_file):
            raise tokenizer_error

        print(f"  Tokenizer fallback: loading tokenizer.json directly ({tokenizer_error})")
        return PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_file,
            bos_token="<|endoftext|>",
            eos_token="<|end|>",
            unk_token="<|endoftext|>",
            pad_token="<|end|>",
        )


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

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
    inputs.pop("token_type_ids", None)
    inputs = inputs.to(_model_device())

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens      = max_tokens,
            max_time            = MAX_GENERATION_SECONDS,
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

    return {
        "answer"    : answer,
        "confidence": round(_estimate_confidence(answer)),
        "source"    : "MedQuAD (NIH) — Fine-tuned Phi-4 Mini",
    }


def _stream_answer(question: str, max_tokens: int = MAX_NEW_TOKENS):
    """Yield answer text chunks as they are generated."""
    if not MODEL_LOADED:
        demo = (
            f"[DEMO MODE — model not loaded]\n\n"
            f"Your question was: \"{question}\"\n\n"
            "To get real answers, run this server on the machine where your "
            "fine-tuned model is saved (./outputs/phi4-medical-final or ./outputs/phi4-medical-lora).\n\n"
            "⚠️ This system is for educational purposes only. Always consult a "
            "licensed healthcare professional for personal medical advice."
        )
        yield demo
        return

    from transformers import TextIteratorStreamer

    prompt = (
        f"<|system|>\n{SYSTEM_MSG}<|end|>\n"
        f"<|user|>\n{question}<|end|>\n"
        f"<|assistant|>\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
    inputs.pop("token_type_ids", None)
    inputs = inputs.to(_model_device())

    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)

    def _run_generation():
        with torch.inference_mode():
            model.generate(
                **inputs,
                max_new_tokens      = max_tokens,
                max_time            = MAX_GENERATION_SECONDS,
                temperature         = 0.7,
                do_sample           = True,
                top_p               = 0.9,
                repetition_penalty  = 1.1,
                pad_token_id        = tokenizer.eos_token_id,
                streamer            = streamer,
            )

    worker = threading.Thread(target=_run_generation, daemon=True)
    worker.start()

    for text in streamer:
        if text:
            yield text

    worker.join(timeout=1)


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

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
    inputs.pop("token_type_ids", None)
    inputs = inputs.to(_model_device())

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens     = MAX_SUMMARY_TOKENS,
            max_time           = MAX_GENERATION_SECONDS,
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
@app.route("/", methods=["GET"])
def splash():
    return send_from_directory(FRONTEND_DIR, "splashscreen.html")


@app.route("/app", methods=["GET"])
def frontend():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:path>", methods=["GET"])
def static_assets(path):
    clean_path = path.split("?", 1)[0]
    if clean_path in {"index.html", "splashscreen.html", "styles.css", "script.js"} or clean_path.startswith("assets/"):
        return send_from_directory(FRONTEND_DIR, clean_path)
    return jsonify({"error": "Not found"}), 404


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status"      : "ok",
        "model_loaded": MODEL_LOADED,
        "backend"     : MODEL_BACKEND,
        "load_error"  : MODEL_LOAD_ERROR,
        "model_path"  : BASE_MODEL,
        "lora_path"   : LORA_PATH if os.path.exists(LORA_PATH) else "",
        "gpu"         : torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "cpu_offload" : MODEL_CPU_OFFLOAD,
        "device_map"  : MODEL_DEVICE_MAP_SUMMARY,
    })


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data     = request.get_json(silent=True) or {}
        question = (data.get("question") or "").strip()

        if not question:               return jsonify({"error": "Missing 'question' field."}), 400
        if len(question) > 1000:       return jsonify({"error": "Question too long (max 1000 characters)."}), 400

        is_urgent = bool(URGENT_PATTERN.search(question))
        result  = generate_answer(question)
        related = generate_related_questions(question, result["answer"])

        return jsonify({
            "answer"           : result["answer"],
            "confidence"       : result["confidence"],
            "source"           : result["source"],
            "related_questions": related,
            "urgent_warning"   : URGENT_WARNING if is_urgent else "",
        })
    except Exception as error:
        print(traceback.format_exc())
        return jsonify({
            "error": "Model generation failed.",
            "detail": str(error),
        }), 500


@app.route("/chat_stream", methods=["POST"])
def chat_stream():
    data     = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "Missing 'question' field."}), 400
    if len(question) > 1000:
        return jsonify({"error": "Question too long (max 1000 characters)."}), 400

    is_urgent = bool(URGENT_PATTERN.search(question))

    def event_stream():
        answer_chunks = []
        try:
            for chunk in _stream_answer(question):
                answer_chunks.append(chunk)
                payload = json.dumps({"delta": chunk})
                yield f"data: {payload}\n\n"

            answer = "".join(answer_chunks).strip()
            related = generate_related_questions(question, answer)
            done_payload = json.dumps({
                "answer": answer,
                "confidence": round(_estimate_confidence(answer)),
                "source": "MedQuAD (NIH) — Fine-tuned Phi-4 Mini",
                "related": related,
                "urgent_warning": URGENT_WARNING if is_urgent else "",
            })
            yield f"event: done\ndata: {done_payload}\n\n"
        except Exception as error:
            err_payload = json.dumps({"error": str(error)})
            yield f"event: error\ndata: {err_payload}\n\n"

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    return Response(stream_with_context(event_stream()), headers=headers, mimetype="text/event-stream")


@app.route("/summarize", methods=["POST"])
def summarize():
    try:
        data = request.get_json(silent=True) or {}
        text = (data.get("text") or "").strip()

        if not text:           return jsonify({"error": "Missing 'text' field."}), 400
        if len(text) > 5000:   return jsonify({"error": "Text too long (max 5000 characters)."}), 400

        return jsonify({"summary": generate_summary(text)})
    except Exception as error:
        print(traceback.format_exc())
        return jsonify({
            "error": "Model summarization failed.",
            "detail": str(error),
        }), 500


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n Starting Flask server on http://localhost:8000")
    print(" Open http://localhost:8000 to start at the MedAI splash screen.\n")
    print(" API endpoints are available on the same port.")
    try:
        from waitress import serve
        print(" Using waitress server (recommended for long requests).")
        serve(app, host="0.0.0.0", port=8000, threads=4, channel_timeout=300)
    except Exception as server_error:
        print(f" Waitress not available ({server_error}). Falling back to Flask dev server.")
        app.run(host="0.0.0.0", port=8000, debug=False, threaded=True)
