# app.py
import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from quant_model import QuantModel


# =============================
# ⚡ Загружаем модели
# =============================

@st.cache_resource
def load_models():
    # Оригинальная FP32 модель
    orig_id = "deepvk/USER-bge-m3"
    orig_tokenizer = AutoTokenizer.from_pretrained(orig_id)
    orig_model = AutoModel.from_pretrained(orig_id)

    # Квантованная INT8 модель
    quant = QuantModel("skatzR/USER-BGE-M3-ONNX-INT8")

    return orig_tokenizer, orig_model, quant


orig_tokenizer, orig_model, quant_model = load_models()


# =============================
# ⚡ Утилиты
# =============================

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(1) / torch.clamp(mask.sum(1), min=1e-9)


def encode_orig(texts):
    inputs = orig_tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = orig_model(**inputs)
    emb = mean_pooling(outputs, inputs["attention_mask"])
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.cpu().numpy()


# =============================
# ⚡ Streamlit UI
# =============================

st.title("🧩 DeepVK-BGE-M3 — FP32 vs Quantized ONNX")
st.write("Сравнение эмбеддингов оригинальной и квантованной модели")

text1 = st.text_input("Текст 1", "Привет, мир!")
text2 = st.text_input("Текст 2", "Hello, world!")

if st.button("🔎 Проверить"):
    texts = [text1, text2]

    # FP32
    emb_orig = encode_orig(texts)

    # INT8 ONNX
    emb_quant = quant_model.encode(texts)

    # Cosine similarity внутри моделей
    sim_orig = float(np.dot(emb_orig[0], emb_orig[1]))
    sim_quant = float(np.dot(emb_quant[0], emb_quant[1]))

    # Cosine similarity между моделями
    cross_sim = float(np.dot(emb_orig[0], emb_quant[0]))

    st.subheader("📐 Cosine Similarities")
    st.write(f"**FP32 model:** {sim_orig:.4f}")
    st.write(f"**Quantized INT8 model:** {sim_quant:.4f}")
    st.write(f"**FP32 vs INT8 (cross-check):** {cross_sim:.4f}")
