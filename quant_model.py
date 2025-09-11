import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
from pathlib import Path


class QuantModel:
    def __init__(self, repo_id: str, use_cpu: bool = True):
        """
        Универсальная обёртка для квантованной ONNX модели с Hugging Face Hub.

        Args:
            repo_id (str): ID модели на Hugging Face Hub (например, "skatzR/USER-BGE-M3-ONNX-INT8")
            use_cpu (bool): использовать только CPU (True) или дать ONNX выбрать доступные провайдеры (False)
        """
        self.repo_id = repo_id

        # загружаем onnx модель с HF Hub
        self.onnx_path = hf_hub_download(repo_id=repo_id, filename="model_quantized.onnx")

        # загружаем токенизатор
        self.tokenizer = AutoTokenizer.from_pretrained(repo_id, use_fast=True)

        # создаём inference session
        providers = ["CPUExecutionProvider"] if use_cpu else None
        self.sess = ort.InferenceSession(self.onnx_path, providers=providers)

        # список входов модели
        self.model_inputs = {inp.name for inp in self.sess.get_inputs()}

    def encode(self, texts, batch_size: int = 32, normalize: bool = True):
        """
        Кодируем список текстов в эмбеддинги.

        Args:
            texts (list[str]): список текстов
            batch_size (int): размер батча
            normalize (bool): нормализовать эмбеддинги (L2-норма)

        Returns:
            np.ndarray: эмбеддинги [batch, hidden_dim]
        """
        all_embeddings = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]

            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="np"
            )

            # фильтруем входы под модель
            ort_inputs = {k: v for k, v in inputs.items() if k in self.model_inputs}

            outputs = self.sess.run(None, ort_inputs)
            token_embeddings = outputs[0]

            # если модель вернула [batch, seq_len, hidden_dim], делаем mean pooling
            if token_embeddings.ndim == 3:
                mask = ort_inputs["attention_mask"].astype(np.float32)[..., None]
                summed = (token_embeddings * mask).sum(axis=1)
                counts = np.clip(mask.sum(axis=1), 1e-9, None)
                embeddings = summed / counts
            else:
                embeddings = token_embeddings  # уже [batch, hidden_dim]

            if normalize:
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)
