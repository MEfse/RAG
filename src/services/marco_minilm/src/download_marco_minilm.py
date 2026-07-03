import torch
from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

class MarcoMiniLM:
    def __init__(self, device: str = 'cuda') -> None:
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = CrossEncoder(MODEL_NAME, device=self.device)

    def predict(self, pairs):     
        scores = self.model.predict(pairs)
        return scores
    
# Загружаем модель один раз при старте контейнера
cross_encoder_model = MarcoMiniLM(device='cuda')