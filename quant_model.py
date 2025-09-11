# quant_model.py
from pathlib import Path
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download


class QuantModel:
    """
    Простая, надёжная обёртка для квантованной ONNX модели, размещённой на HuggingFace.
    - скачивает model_quantized.onnx (hf_hub_download)
    - загружает локальный токенизатор (из репо)
    - создаёт ONNX InferenceSession
    - метод encode(texts) возвращает np.ndarray shape (N, hidden_dim), уже L2-нормализованные
    """

    def __init__(self, hf_repo_id: str, use_cpu: bool = True, cache_dir: str | None = None):
        self.repo_id = hf_repo_id

        # 1) скачиваем ONNX файл (hf_hub_download) — бросит понятную ошибку, если файла нет
        try:
            # ожидаем, что в репозитории есть точное имя "model_quantized.onnx"
            self.onnx_path = hf_hub_download(repo_id=self.repo_id,
                                             filename="model_quantized.onnx",
                                             cache_dir=cache_dir)
        except Exception as e:
            raise RuntimeError(f"Cannot download model_quantized.onnx from '{self.repo_id}': {e}")

        # 2) токенизатор — пытаемся загрузить локально из репо (чтобы использовать те же файлы, что и в архиве)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.repo_id, use_fast=True)
        except Exception as e:
            raise RuntimeError(f"Cannot load tokenizer from '{self.repo_id}': {e}")

        # 3) создаём ONNX session
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ["CPUExecutionProvider"] if use_cpu else None
        try:
            self.sess = ort.InferenceSession(self.onnx_path, sess_options=so, providers=providers)
        except Exception as e:
            raise RuntimeError(f"ONNXRuntime failed to create session from '{self.onnx_path}': {e}")

        # кэш имён входов модели (чтобы отфильтровать лишние поля типа token_type_ids)
        self.input_names = {inp.name for inp in self.sess.get_inputs()}

    def encode(self, texts: list[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        """
        Вернуть эмбеддинги (N, hidden_dim) для списка texts.
        Гарантирует:
         - корректное сопоставление входов (фильтрация token_type_ids, если модель их не ожидает)
         - pooled embeddings (mean pooling по attention_mask) если модель вернула токенные эмбединги
         - L2-нормализацию (если normalize=True)
        """
        if isinstance(texts, str):
            texts = [texts]
        all_embs = []

        model_input_names = self.input_names

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            # токенизация -> numpy (transformers возвращает numpy при return_tensors="np")
            inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="np")

            # фильтруем входы, оставляя только те, что ожидает ONNX модель
            ort_inputs = {k: v for k, v in inputs.items() if k in model_input_names}

            # если attention_mask отсутствует — создадим единичную (без паддинга) — защищаем pooling
            if "attention_mask" not in ort_inputs:
                seq_len = inputs["input_ids"].shape[1]
                ort_inputs["attention_mask"] = np.ones((len(batch), seq_len), dtype=np.int64)

            # инференс
            outputs = self.sess.run(None, ort_inputs)
            emb = outputs[0]  # ожидаем (batch, seq, dim) или (batch, dim)

            # pooling если нужно
            if emb.ndim == 3:
                mask = ort_inputs["attention_mask"].astype(np.float32)[..., None]  # (batch, seq, 1)
                summed = (emb * mask).sum(axis=1)  # (batch, dim)
                counts = mask.sum(axis=1)  # (batch, 1)
                counts = np.clip(counts, 1e-9, None)
                pooled = summed / counts
            elif emb.ndim == 2:
                pooled = emb
            else:
                # fallback
                pooled = emb.mean(axis=1)

            if normalize:
                pooled = pooled / (np.linalg.norm(pooled, axis=1, keepdims=True) + 1e-12)

            all_embs.append(pooled)

        return np.vstack(all_embs)
