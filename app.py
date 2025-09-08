import streamlit as st
import numpy as np
from quant_model import get_model

# =======================
# 🔧 Настройки страницы
# =======================
st.set_page_config(page_title="Тест квантизированных моделей", layout="wide")
st.title("🧠 Тест квантизированных моделей ONNX (int8)")

# =======================
# 📥 Загрузка модели
# =======================
with st.spinner("Загружаем модель..."):
    model = get_model()

st.success("✅ Модель загружена!")

# =======================
# 📝 Поле ввода текста
# =======================
input_text = st.text_area(
    "Введите один или несколько текстов (каждый с новой строки):",
    "Привет мир\nКвантизированные модели экономят память!"
)

normalize = st.checkbox("Нормализовать эмбеддинги", value=True)

# =======================
# 🚀 Кнопка запуска
# =======================
if st.button("🔍 Получить эмбеддинги"):
    texts = [t.strip() for t in input_text.split("\n") if t.strip()]
    if not texts:
        st.warning("Введите хотя бы один текст")
    else:
        with st.spinner("Вычисляем эмбеддинги..."):
            embeddings = model.encode(texts, normalize_embeddings=normalize)

        st.success("✅ Готово!")
        st.write(f"Форма эмбеддингов: {embeddings.shape}")

        # Показать пример
        st.write("🔢 Первые значения первого эмбеддинга:")
        st.code(np.round(embeddings[0][:10], 4))

        # Таблица всех эмбеддингов
        st.write("📊 Все эмбеддинги:")
        st.dataframe(np.round(embeddings, 4))

# =======================
# 📈 Мониторинг ресурсов
# =======================
st.markdown("---")
st.subheader("📈 Ресурсы системы")
stats = model.profile()
col1, col2, col3 = st.columns(3)
col1.metric("CPU (%)", stats["cpu_percent"])
col2.metric("RAM (%)", stats["ram_percent"])
col3.metric("RAM (MB)", round(stats["ram_mb"], 1))
