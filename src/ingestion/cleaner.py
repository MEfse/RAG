import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

from src.settings.config import Columns

import logging
logger = logging.getLogger(__name__)

class DataTransformer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def preprocess_data(self, 
                        questions: pd.DataFrame, 
                        answers: pd.DataFrame, 
                        tags: pd.DataFrame) -> pd.DataFrame:
        """
        Выполняет полный preprocessing входных данных.

        Последовательность:
            - объединяет questions, answers и tags
            - оставляет только лучший ответ для каждого question_id
            - очищает HTML-теги из текста
            - формирует итоговое поле document_text

        Args:
            questions (pd.DataFrame): Таблица вопросов.
            answers (pd.DataFrame): Таблица ответов.
            tags (pd.DataFrame): Таблица тегов.

        Returns:
            pd.DataFrame: Предобработанный DataFrame с очищенным текстом и собранным document_text.
        """

        # Оставляем только лучший ответ
        data = self.select_best_answers(data)
        logger.info("После select_best_answers: shape=%s", data.shape)

        # Шаг 3. Убираем в тексте теги
        data[Columns.BODY_QUESTION.value] = data[Columns.BODY_QUESTION.value].apply(self.clean_html)
        data[Columns.BODY_ANSWER.value] = data[Columns.BODY_ANSWER.value].apply(self.clean_html)
        logger.info("Данные от HTML очищены")

        # Собираем документ
        data[Columns.DOCUMENT_TEXT.value] = ("Title: " + data[Columns.TITLE.value] + "\n\n" +
                                             "Tags: " + data[Columns.TAGS.value].fillna("") + "\n\n" +
                                             "Question: " + data[Columns.BODY_QUESTION.value] + "\n\n" +
                                             "Best Answer: " + data[Columns.BODY_ANSWER.value])
        
        logger.info("document_text сформирован: shape=%s", data.shape)

        return data
    
    def select_best_answers(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Оставляет только лучший ответ для каждого question_id.

        Логика:
            - сортирует ответы по убыванию answer_score
            - для каждого question_id оставляет первый (лучший) ответ

        Args:
            data (pd.DataFrame): Объединённый DataFrame с вопросами и ответами.

        Returns:
            pd.DataFrame: DataFrame, содержащий по одному (лучшему) ответу на каждый вопрос.
        """

        best_answers = (
            data.sort_values([Columns.QUESTION_ID.value, Columns.ANSWER_SCORE.value], 
                             ascending=[True, False])
            .drop_duplicates(subset=Columns.QUESTION_ID.value, keep='first')
            .reset_index(drop=True))
        
        logger.info("Оставлен только лучший ответ: shape=%s", best_answers.shape)
        return best_answers

    def clean_html(self, text) -> str:
        """
        Удаляет HTML-теги из текста.

        Args:
            text (str): Входной текст, содержащий HTML.

        Returns:
            str: Очищенный текст без HTML-тегов.
                Если входное значение NaN или произошла ошибка — возвращает пустую строку.

        Raises:
            Exception: Логируется в случае ошибки парсинга HTML.
        """

        if pd.isna(text):
            return ""
        
        try:
            text = BeautifulSoup(text, "html.parser").get_text(" ", strip=True)
            return text
        except Exception as e:
            logger.error("Не удалось очистить HTML. Ошибка: %s", e, exc_info=True)
            return ""
        
    def text_lower(self, text):
        #logger.info(f"Текст приведен к нижнему регистру.")
        return text.lower()
    
    def delete_space(self, text):
        #logger.info(f"В тексте удалены лишние пробелы.")
        return re.sub(r'\s+', ' ', text).strip()
    
    def delete_stopwords(self, text):
        #logger.info(f"В тексте удалены стоп слова через nltk.")
        return ' '.join([word for word in text.split() if word not in self.stop_words])
    
    def lemmatize(self, text):
        #logger.info(f"Текст лемматизирован.")
        return ' '.join([self.lemmatizer.lemmatize(word) for word in text.split()])


    #def _add_len_words(self, data):
        #data['doc_length_words'] = data[Columns.DOCUMENT_TEXT.value].apply(lambda x: len(word_tokenize(x)))
