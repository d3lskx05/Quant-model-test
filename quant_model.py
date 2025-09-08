import os
import zipfile
import gdown
import numpy as np
import onnxruntime as ort
from pathlib import Path
from functools import lru_cache
from transformers import AutoTokenizer
import huggingface_hub


class QuantModel:
    """
    Универсальный загрузчик квантизированных ONNX моделей.
    Поддержка:
      - Google Drive (по ID)
      - Hugging Face Hub
      - Локальная модель
    """

    def __init__(self, model_id: str, source: str = "gdrive",
                 model_dir: str = "onnx_model", model_file: str = "model_quantized.onnx"):
        self.model_id = model_id
        self.source = source
        self.model_dir = Path(model_dir)
        self.model_file = model_file
        self.model_path = self.model_dir / self.model_file

        self._ensure_model()
        self.session = self._load_session()
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))

    def _ensure_model(self):
        """Скачивание и распаковка модели"""
        if self.model_path.exists():
            return

        os.makedirs(self.model_dir, exist_ok=True)

        if self.source == "gdrive":
            zip_path = f"{self.model_dir}.zip"
            print(f"📥 Скачиваю модель с Google Drive: {self.model_id}")
            gdown.download(f"https://drive.google.com/uc?id={self.model_id}", zip_path, quiet=False)
            print(f"📦 Распаковка {zip_path}...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(self.model_dir)
            os.remove(zip_path)

        elif self.source == "hf":
            print(f"📥 Скачиваю модель с Hugging Face: {self.model_id}")
            huggingface_hub.snapshot_download(
                repo_id=self.model_id,
                local_dir=self.model_dir,
                local_dir_use_symlinks=False
            )

        elif self.source == "local":
            print(f"📂 Использую локальную модель: {self.model_dir}")
        else:
            raise ValueError(f"❌ Неизвестный source: {self.source}")

        print("✅ Модель готова!")

    def _load_session(self):
        """Создаём ONNX Runtime Session"""
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = ["CPUExecutionProvider"]
        try:
            if ort.get_device() == "GPU":
                providers.insert(0, "CUDAExecutionProvider")
        except Exception:
            pass

        print(f"🚀 Загружаю модель {self.model_file} с провайдерами: {providers}")
        return ort.InferenceSession(str(self.model_path), sess_options=so, providers=providers)

    @lru_cache(maxsize=512)
    def _encode_cached(self, text: str, normalize: bool = True):
        """Кэшированное кодирование одной строки"""
        inputs = self.tokenizer([text], padding=True, truncation=True, return_tensors="np")
        ort_inputs = {k: v for k, v in inputs.items()}
        embeddings = self.session.run(None, ort_inputs)[0]

        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
            embeddings = embeddings / norms
        return embeddings[0]

    def encode(self, texts, normalize=True):
        """Кодирование текста в эмбеддинги"""
        if isinstance(texts, str):
            texts = [texts]
        return np.array([self._encode_cached(t, normalize) for t in texts])


@lru_cache(maxsize=1)
def get_model():
    """Глобальный доступ к модели с кэшем в памяти"""
    model_id = os.getenv("MODEL_ID", "1lkrvCPIE1wvffIuCSHGtbEz3Epjx5R36")
    source = os.getenv("MODEL_SOURCE", "gdrive")
    model_dir = os.getenv("MODEL_DIR", "onnx-user-bge-m3")
    model_file = os.getenv("MODEL_FILE", "model_quantized.onnx")
    return QuantModel(model_id, source, model_dir, model_file)
