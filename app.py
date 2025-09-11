import streamlit as st
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from quant_model import QuantModel


# -------------------------------
# Оригинальная FP32 модель
# -------------------------------
ORIG_ID = "deepvk/USER-bge-m3"
orig_tokenizer = AutoTokenizer.from_pretrained(ORIG_ID)
orig_model = AutoModel.from_pretrained(ORIG_ID)


def encode_orig(texts):
    """Получаем эмбеддинги через оригинальную PyTorch модель"""
    inputs = orig_tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = orig_model(**inputs)
    token_embeddings = outputs[0]
    mask = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = (token_embeddings * mask).sum(1)
    counts = torch.clamp(mask.sum(1), min=1e-9)
    embeddings = summed / counts
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy()


# -------------------------------
# Квантованная ONNX модель
# -------------------------------
quant_model = QuantModel("skatzR/USER-BGE-M3-ONNX-INT8")


# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🔬 Compare FP32 vs Quantized (ONNX INT8)")
st.write("Проверка cosine similarity между оригинальной моделью и квантованной.")

text1 = st.text_input("Введите текст 1", "Привет, как дела?")
text2 = st.text_input("Введите текст 2", "Hello, how are you?")

if st.button("Сравнить"):
    with st.spinner("⚡ Считаем эмбеддинги..."):
        emb_orig = encode_orig([text1, text2])
        emb_quant = quant_model.encode([text1, text2])

        sim_orig = float(np.dot(emb_orig[0], emb_orig[1]))
        sim_quant = float(np.dot(emb_quant[0], emb_quant[1]))

    st.subheader("📊 Результаты")
    st.write(f"**Оригинальная FP32 модель:** cosine similarity = `{sim_orig:.4f}`")
    st.write(f"**Квантованная INT8 модель:** cosine similarity = `{sim_quant:.4f}`")
