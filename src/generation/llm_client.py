import torch
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import logging
logger = logging.getLogger(__name__)

#generator = pipeline('text-generation', model='gpt-2')

class LLMClient:
    def __init__(self, model_name="google/flan-t5-large", max_new_tokens=256):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        logger.info(f"Загрузка LLM-модели: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        logger.info(f"LLM загружена на устройство: {self.device}")

    def generate_answer(self, prompt):
        if not prompt or not isinstance(prompt, str):
            raise ValueError("prompt должен быть непустой строкой.")
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=96,
            num_beams=5,
            do_sample=False,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        logger.info(f"Сгенерирован ответ длиной {len(answer)} символов")

        return answer