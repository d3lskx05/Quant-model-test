import streamlit as st
import numpy as np
from quant_model import get_model

st.set_page_config(page_title="Quantized Model Demo", layout="wide")
st.title("⚡ Быстрое подключение квантизированной модели")

# Загружаем модель (кэшируется)
model = get_model()
st.success("✅ Модель загружена и готова!")

# Текстовое поле
text_input = st.text_area(
    "Введите текст(ы) для кодирования (по одному на строку):",
    "Привет, мир!\nЭто тестовое предложение."
)

# Кнопка запуска
if st.button("🚀 Получить эмбеддинги"):
    texts = [t.strip() for t in text_input.split("\n") if t.strip()]
    embeddings = model.encode(texts)

    st.subheader("🔢 Результат")
    st.write(f"Размер эмбеддингов: {embeddings.shape}")
    st.json(embeddings[0][:10].tolist())
