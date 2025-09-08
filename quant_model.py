import os
import zipfile
import gdown
import numpy as np
import onnxruntime as ort
from functools import lru_cache
from transformers import AutoTokenizer
import huggingface_hub
import psutil


class QuantizedSentenceModel:
    """
    Универсальный класс для работы с квантизированными ONNX моделями
    в стиле SentenceTransformer. Поддержка:
      - Google Drive (по ID)
      - Hugging Face Hub
      - Локальная модель
    """

    def __init__(self, model_id: str, source: str = "gdrive",
                 model_dir: str = "onnx_model", model_file: str = "model_quantized.onnx"):
        self.model_id = model_id
        self.source = source  # "gdrive", "hf", "local"
        self.model_dir = model_dir
        self.model_file = model_file
        self.model_path = os.path.join(self.model_dir, self.model_file)

        self._ensure_model()
        self.session = self._load_session()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)

    def _ensure_model(self):
        """Скачивание и распаковка модели в зависимости от source."""
        if os.path.exists(self.model_path):
            return  # Уже есть

        os.makedirs(self.model_dir, exist_ok=True)

        if self.source == "gdrive":
            zip_path = f"{self.model_dir}.zip"
            print(f"📥 Скачиваю модель с Google Drive: {self.model_id}")
            url = f"https://drive.google.com/uc?id={self.model_id}"
            gdown.download(url, zip_path, quiet=False)
            print(f"📦 Распаковка {zip_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
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
            # Ничего не делаем
        else:
            raise ValueError(f"❌ Неизвестный source: {self.source}")

        print("✅ Модель готова!")

    def _load_session(self):
        """Создаём ONNX Runtime Session с авто-выбором устройства."""
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Если есть GPU, используем CUDA
        providers = ["CPUExecutionProvider"]
        try:
            if ort.get_device() == "GPU":
                providers.insert(0, "CUDAExecutionProvider")
        except Exception:
            pass

        print(f"🚀 Загружаю модель {self.model_file} с провайдерами: {providers}")
        return ort.InferenceSession(self.model_path, sess_options=so, providers=providers)

    @lru_cache(maxsize=1024)
    def _encode_cached(self, texts_tuple, normalize_embeddings):
        """Кэшируемый внутренний метод (в памяти)."""
        texts = list(texts_tuple)
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="np"
        )
        ort_inputs = {k: v for k, v in inputs.items()}
        embeddings = self.session.run(None, ort_inputs)[0]

        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
            embeddings = embeddings / norms
        return embeddings

    def encode(self, texts, normalize_embeddings=True):
        """Публичный метод encode с кэшированием в памяти."""
        if isinstance(texts, str):
            texts = [texts]
        texts_tuple = tuple(texts)
        return self._encode_cached(texts_tuple, normalize_embeddings)

    # Заглушка для дискового кэша
    # def _encode_disk_cache(self, texts, normalize_embeddings=True):
    #     """
    #     TODO: Реализация дискового кэша.
    #     Можно хэшировать тексты, сохранять эмбеддинги в npy-файлы.
    #     """
    #     pass

    @staticmethod
    def profile():
        """Мониторинг ресурсов."""
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        cpu = psutil.cpu_percent(interval=0.1)
        ram = psutil.virtual_memory()
        return {
            "cpu_percent": cpu,
            "ram_percent": ram.percent,
            "ram_mb": mem_info.rss / 1024**2
        }


@lru_cache(maxsize=1)
def get_model():
    """Глобальный доступ к модели с кэшем."""
    model_id = os.getenv("MODEL_ID", "1lkrvCPIE1wvffIuCSHGtbEz3Epjx5R36")
    source = os.getenv("MODEL_SOURCE", "gdrive")  # "gdrive", "hf", "local"
    return QuantizedSentenceModel(model_id, source)
