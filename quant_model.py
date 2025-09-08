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
import huggingface_hub
import torch


class UniversalModel:
    """
    Универсальный загрузчик моделей:
    - ONNX квантизованные (GDrive, HF, локальные)
    - Transformers (HF)
    - Sentence-Transformers (HF)
    - Кэширование модели и эмбеддингов
    """

    def __init__(self, model_id: str, model_type: str = "onnx", source: str = "gdrive",
                 model_dir: str = "onnx_model", tokenizer_name: str = None):
        """
        Args:
            model_id: ID модели (GDrive ID, HF repo_id или путь)
            model_type: "onnx", "transformers", "sentence-transformers"
            source: "gdrive", "hf", "local"
            model_dir: папка для хранения модели
            tokenizer_name: название токенайзера (по умолчанию = model_id)
        """
        self.model_id = model_id
        self.model_type = model_type
        self.source = source
        self.model_dir = Path(model_dir)
        self.tokenizer_name = tokenizer_name or model_id

        self.model_path = None
        self.model = None
        self.session = None
        self.tokenizer = None

        print(f"🔍 Инициализация UniversalModel: type={model_type}, source={source}")
        self._prepare()

    # ========================
    # 🔑 Основная подготовка
    # ========================
    def _prepare(self):
        if self.model_type == "onnx":
            print("🔹 Режим: ONNX модель")
            self._ensure_model_files()
            self.model_path = self._find_onnx_file()
            self.session = self._load_onnx_session()
            self.tokenizer = self._load_tokenizer()
        elif self.model_type == "transformers":
            print("🔹 Режим: Transformers модель")
            self.model = AutoModel.from_pretrained(self.model_id)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        elif self.model_type == "sentence-transformers":
            print("🔹 Режим: Sentence-Transformers модель")
            self.model = SentenceTransformer(self.model_id)
        else:
            raise ValueError(f"❌ Неизвестный тип модели: {self.model_type}")

    # ========================
    # 📥 Загрузка модели
    # ========================
    def _ensure_model_files(self):
        os.makedirs(self.model_dir, exist_ok=True)
        if not any(self.model_dir.glob("*.onnx")):
            if self.source == "gdrive":
                zip_path = f"{self.model_dir}.zip"
                print(f"📥 Скачиваю модель с Google Drive: {self.model_id}")
                gdown.download(f"https://drive.google.com/uc?id={self.model_id}", zip_path, quiet=False)
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(self.model_dir)
                if os.path.exists(zip_path):  # ✅ Безопасное удаление
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
                raise ValueError(f"❌ Неизвестный источник: {self.source}")
        else:
            print(f"✅ Файлы модели уже есть в {self.model_dir}")

    def _find_onnx_file(self):
        onnx_files = list(self.model_dir.rglob("*.onnx"))
        if not onnx_files:
            raise FileNotFoundError(f"❌ В {self.model_dir} не найден .onnx файл!")
        print(f"✅ Найден ONNX файл: {onnx_files[0]}")
        return onnx_files[0]

    def _load_onnx_session(self):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ["CPUExecutionProvider"]
        try:
            if ort.get_device() == "GPU":
                providers.insert(0, "CUDAExecutionProvider")
        except Exception:
            pass
        print(f"🚀 Загружаю ONNX модель на провайдерах: {providers}")
        return ort.InferenceSession(str(self.model_path), sess_options=so, providers=providers)

    def _load_tokenizer(self):
        try:
            return AutoTokenizer.from_pretrained(self.tokenizer_name)
        except Exception:
            try:
                return AutoTokenizer.from_pretrained(str(self.model_dir))
            except Exception:
                print("⚠️ Не удалось загрузить токенайзер, используем deepvk/USER-BGE-M3")
                return AutoTokenizer.from_pretrained("deepvk/USER-BGE-M3")

    # ========================
    # 🔥 Кодирование
    # ========================
    @lru_cache(maxsize=1024)
    def _encode_single(self, text: str, normalize: bool = True):
        if self.model_type == "onnx":
            inputs = self.tokenizer([text], padding=True, truncation=True, return_tensors="np")
            ort_inputs = {k: v for k, v in inputs.items()}
            emb = self.session.run(None, ort_inputs)[0]
        elif self.model_type == "transformers":
            self.model.eval()
            inputs = self.tokenizer([text], padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
                emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        elif self.model_type == "sentence-transformers":
            emb = self.model.encode([text], convert_to_numpy=True)
        else:
            raise ValueError("❌ Неизвестный тип модели")

        if normalize:
            norm = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10
            emb = emb / norm
        return emb[0]

    def encode(self, texts: Union[str, List[str]], normalize=True):
        if isinstance(texts, str):
            texts = [texts]
        return np.array([self._encode_single(t, normalize) for t in texts])


# ========================
# 🔗 Глобальный доступ
# ========================
@lru_cache(maxsize=1)
def get_model():
    model_id = os.getenv("MODEL_ID", "1lkrvCPIE1wvffIuCSHGtbEz3Epjx5R36")
    model_type = os.getenv("MODEL_TYPE", "onnx")  # onnx / transformers / sentence-transformers
    source = os.getenv("MODEL_SOURCE", "gdrive")
    model_dir = os.getenv("MODEL_DIR", "onnx-user-bge-m3")
    tokenizer = os.getenv("TOKENIZER_NAME", None)
    return UniversalModel(model_id, model_type, source, model_dir, tokenizer)
