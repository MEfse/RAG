import torch
from sentence_transformers import SentenceTransformer
from settings.config import Settings


class MiniLMEmbedding:
    def __init__(self, device: str = 'cuda') -> None:
        self.settings = Settings()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = SentenceTransformer(self.settings.MINILM, device=self.device)

    def encode(self, texts, batch_size: int = 512):     
        if isinstance(texts, str):
            texts = [texts]

        return self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
    
# Загружаем модель один раз при старте контейнера
embedding_model = MiniLMEmbedding(device='cuda')



class MiniLMOnnxEmbedding:
    def __init__(self, model_path: str = "minilm_onnx"):
        self.model = SentenceTransformer(model_path, device="cpu")  # ONNX обычно CPU

    def encode(self, texts, batch_size: int = 256):
        if isinstance(texts, str):
            texts = [texts]

        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )