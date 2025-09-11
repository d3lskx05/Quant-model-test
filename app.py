# app.py
import streamlit as st
import numpy as np
import time
from pathlib import Path
from typing import Tuple

import torch
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import HfHubHTTPError

from quant_model import QuantModel

st.set_page_config(page_title="Quantized vs FP32 tester", layout="wide")
st.title("Compare FP32 (HF) vs ONNX(INT8) quantized — USER-BGE-M3")

st.markdown("Enter HF repo ids. The quantized ONNX repo must contain `model_quantized.onnx` + tokenizer files.")

with st.sidebar:
    st.header("Settings")
    orig_id = st.text_input("Original HF model id", value="deepvk/USER-BGE-M3")
    quant_id = st.text_input("Quant HF repo id (ONNX)", value="skatzR/USER-BGE-M3-ONNX-INT8")
    bench_batch = st.number_input("Batch size for benchmark", min_value=1, max_value=256, value=32)
    eval_n = st.number_input("Number of texts for cosine eval", min_value=1, max_value=2000, value=200)
    btn = st.button("Run comparison")

# -------------------------
# Кешированные загрузчики — не выполняются при импорте модуля
# -------------------------
@st.cache_resource(show_spinner=False)
def load_orig_model_and_tokenizer(hf_id: str):
    """Load HF original model + tokenizer. Runs once and is cached."""
    try:
        tok = AutoTokenizer.from_pretrained(hf_id, use_fast=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer for original model '{hf_id}': {e}")
    try:
        # NOTE: do not use device_map="auto" here (would require accelerate in environment)
        model = AutoModel.from_pretrained(hf_id)
        # ensure cpu and eval mode
        model.to("cpu")
        model.eval()
    except Exception as e:
        raise RuntimeError(f"Failed to load original HF model '{hf_id}': {e}")
    return model, tok


@st.cache_resource(show_spinner=False)
def load_quant(hf_repo_id: str):
    """Load QuantModel from quant_model.py (uses hf_hub_download inside)."""
    try:
        q = QuantModel(hf_repo_id, use_cpu=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load quant model from '{hf_repo_id}': {e}")
    return q

# -------------------------
# Encoding helpers
# -------------------------
def encode_orig_batch(model, tokenizer, texts: list[str], batch_size: int = 32) -> np.ndarray:
    all = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        input_ids = inputs["input_ids"].to("cpu")
        att = inputs["attention_mask"].to("cpu")
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=att)
        token_embs = out.last_hidden_state  # (batch, seq, dim)
        mask = att.unsqueeze(-1).expand(token_embs.size()).float()
        summed = (token_embs * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        sent = summed / counts
        sent = torch.nn.functional.normalize(sent, p=2, dim=1)
        all.append(sent.cpu().numpy())
    return np.vstack(all)


# -------------------------
# Run when user presses button
# -------------------------
if btn:
    st.info("Starting test — loading models (cached). This may take some time on first run...")
    try:
        model_orig, tok_orig = load_orig_model_and_tokenizer(orig_id)
    except Exception as e:
        st.error(f"Error loading original model: {e}")
        st.stop()

    try:
        quant = load_quant(quant_id)
    except Exception as e:
        st.error(f"Error loading quant model: {e}")
        st.stop()

    # prepare sample texts
    sample_input = st.text_area("Input texts (one per line). Leave empty -> autosample.",
                               value="Тестовая строка для замера скорости.\nПример использования модели.\nКак дела?")
    lines = [l.strip() for l in sample_input.splitlines() if l.strip()]
    if not lines:
        lines = [f"Test sentence {i}" for i in range(50)]

    # build eval set
    if len(lines) < eval_n:
        times = (eval_n + len(lines) - 1) // len(lines)
        eval_texts = (lines * times)[:eval_n]
    else:
        eval_texts = lines[:eval_n]

    # warmup
    st.info("Warmup runs...")
    for _ in range(2):
        _ = encode_orig_batch(model_orig, tok_orig, eval_texts[:min(8, len(eval_texts))], batch_size=int(bench_batch))
        _ = quant.encode(eval_texts[:min(8, len(eval_texts))], batch_size=int(bench_batch))

    # benchmark
    st.info("Benchmarking...")
    bench_list = eval_texts * 3  # repeat to smooth
    t0 = time.perf_counter()
    _ = encode_orig_batch(model_orig, tok_orig, bench_list, batch_size=int(bench_batch))
    t1 = time.perf_counter()
    orig_time = t1 - t0

    t0 = time.perf_counter()
    _ = quant.encode(bench_list, batch_size=int(bench_batch))
    t1 = time.perf_counter()
    quant_time = t1 - t0

    # compute embeddings for comparison
    st.info("Computing embeddings for cosine comparison...")
    emb_o = encode_orig_batch(model_orig, tok_orig, eval_texts, batch_size=int(bench_batch))
    emb_q = quant.encode(eval_texts, batch_size=int(bench_batch))

    # align dims if needed
    if emb_o.shape[1] != emb_q.shape[1]:
        m = min(emb_o.shape[1], emb_q.shape[1])
        emb_o = emb_o[:, :m]
        emb_q = emb_q[:, :m]

    per_cos = (emb_o * emb_q).sum(axis=1)  # both are normalized
    avg_cos = float(np.mean(per_cos))
    med_cos = float(np.median(per_cos))

    # sizes
    def dir_size_mb(path: Path) -> float:
        if not path.exists():
            return 0.0
        total = 0
        for p in path.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
        return total / (1024 * 1024)

    # quant repo is cached by hf_hub_download — path we can show
    quant_local_path = Path(quant.onnx_path).parent
    quant_size = dir_size_mb(quant_local_path)

    # output
    st.subheader("Results")
    st.table({
        "metric": ["Avg cosine", "Median cosine", "Orig time (s)", "Quant time (s)", "Quant size (MB)"],
        "value": [f"{avg_cos:.6f}", f"{med_cos:.6f}", f"{orig_time:.4f}", f"{quant_time:.4f}", f"{quant_size:.1f}"]
    })

    st.metric("Avg cosine", f"{avg_cos:.4f}")
    st.metric("Orig time (s)", f"{orig_time:.3f}")
    st.metric("Quant time (s)", f"{quant_time:.3f}")
    st.success("Done ✅")
