# quant_model.py
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download


class QuantModel:
    def __init__(self, repo_id: str = "skatzR/USER-BGE-M3-ONNX-INT8"):
        """
        Универсальный загрузчик квантованной ONNX модели с Hugging Face Hub.
        
        Args:
            repo_id (str): Hugging Face repo ID с моделью.
        """
        self.repo_id = repo_id

        # ✅ Загружаем токенизатор
        self.tokenizer = AutoTokenizer.from_pretrained(repo_id)

        # ✅ Скачиваем сам .onnx файл
        model_path = hf_hub_download(repo_id=repo_id, filename="model_quantized.onnx")

        # ✅ Создаём ONNX Runtime сессию
        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    def encode(self, texts, normalize: bool = True):
        """
        Получение эмбеддингов из ONNX модели.

        Args:
            texts (str | list[str]): Текст или список текстов
            normalize (bool): Нормализовать ли эмбеддинги (L2 norm)

        Returns:
            np.ndarray: эмбеддинги (batch_size, hidden_dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        # Токенизация
        inputs = self.tokenizer(texts, return_tensors="np", padding=True, truncation=True)

        # Прогон через модель
        outputs = self.sess.run(None, dict(inputs))
        embeddings = outputs[0]

        # Нормализация эмбеддингов
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms

        return embeddings
