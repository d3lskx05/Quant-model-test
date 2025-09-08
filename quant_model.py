# universal_model.py

import os
import zipfile
import gdown
import numpy as np
import onnxruntime as ort
from pathlib import Path
from functools import lru_cache
from typing import List, Union

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer


class UniversalModel:
    """
    Универсальный класс для подключения моделей:
    - ONNX квантизованные модели (GDrive, HF, локально)
    - Обычные модели (transformers)
    - Модели SentenceTransformer

    Поддержка кэширования в памяти.
    """

    def __init__(self,
                 model_id: str,
                 model_type: str = "onnx",
                 source: str = "gdrive",
                 model_dir: str = "onnx_model",
                 model_file: str = "model_quantized.onnx",
                 tokenizer_name: str = None):
        """
        Args:
            model_id: ID модели (GDrive ID, HF repo_id или локальный путь)
            model_type: "onnx", "transformers", "sentence-transformers"
            source: "gdrive", "hf", "local"
            model_dir: Папка для локального хранения
            model_file: ONNX файл
            tokenizer_name: Название токенайзера (если нужно отдельно)
        """
        self.model_id = model_id
        self.model_type = model_type
        self.source = source
        self.model_dir = model_dir
        self.model_file = model_file
        self.tokenizer_name = tokenizer_name or model_id

        self.session = None
        self.model = None
        self.tokenizer = None

        self._prepare_model()

    def _prepare_model(self):
        """Основная точка загрузки модели."""
        if self.model_type == "onnx":
            self._ensure_model_files()
        self._load_model_and_tokenizer()

    def _ensure_model_files(self):
        """Скачивание и распаковка ONNX модели, если нужно."""
        model_path = Path(self.model_dir) / self.model_file
        if model_path.exists():
            return  # уже есть

        os.makedirs(self.model_dir, exist_ok=True)

        if self.source == "gdrive":
            zip_path = f"{self.model_dir}.zip"
            print(f"📥 Скачиваю модель с Google Drive: {self.model_id}")
            gdown.download(f"https://drive.google.com/uc?id={self.model_id}", zip_path, quiet=False)
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(self.model_dir)
            os.remove(zip_path)

        elif self.source == "hf":
            from huggingface_hub import snapshot_download
            print(f"📥 Скачиваю модель с HF Hub: {self.model_id}")
            snapshot_download(repo_id=self.model_id, local_dir=self.model_dir, local_dir_use_symlinks=False)

        elif self.source == "local":
            print(f"📂 Использую локальную модель из {self.model_dir}")

        else:
            raise ValueError(f"❌ Неизвестный source: {self.source}")

        print("✅ Модель скачана и готова!")

    def _load_model_and_tokenizer(self):
        """Загружаем модель и токенайзер в зависимости от типа."""
        print(f"🚀 Загружаю модель типа {self.model_type}")

        if self.model_type == "onnx":
            model_path = str(Path(self.model_dir) / self.model_file)
            self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        elif self.model_type == "transformers":
            self.model = AutoModel.from_pretrained(self.model_id)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        elif self.model_type == "sentence-transformers":
            self.model = SentenceTransformer(self.model_id)

        else:
            raise ValueError(f"❌ Неизвестный тип модели: {self.model_type}")

        print("✅ Модель загружена!")

    def encode(self, texts: Union[str, List[str]], normalize_embeddings=True) -> np.ndarray:
        """Кодируем текст в эмбеддинги."""
        if isinstance(texts, str):
            texts = [texts]

        if self.model_type == "onnx":
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="np")
            ort_inputs = {k: v for k, v in inputs.items()}
            embeddings = self.session.run(None, ort_inputs)[0]
            embeddings = embeddings.mean(axis=1)

        elif self.model_type == "transformers":
            import torch
            self.model.eval()
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

        elif self.model_type == "sentence-transformers":
            embeddings = self.model.encode(texts, convert_to_numpy=True)

        else:
            raise ValueError(f"❌ Неизвестный тип модели: {self.model_type}")

        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
            embeddings = embeddings / norms
        return embeddings


# =========================
# 🔹 Глобальный доступ к модели
# =========================
@lru_cache(maxsize=1)
def get_model():
    """Возвращает загруженную модель (с кэшем в памяти)."""
    model_id = os.getenv("MODEL_ID", "1lkrvCPIE1wvffIuCSHGtbEz3Epjx5R36")
    model_type = os.getenv("MODEL_TYPE", "onnx")
    source = os.getenv("MODEL_SOURCE", "gdrive")
    model_dir = os.getenv("MODEL_DIR", "onnx-user-bge-m3")
    tokenizer_name = os.getenv("TOKENIZER_NAME", "deepvk/USER-BGE-M3")

    return UniversalModel(
        model_id=model_id,
        model_type=model_type,
        source=source,
        model_dir=model_dir,
        tokenizer_name=tokenizer_name
    )
