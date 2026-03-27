import pandas as pd
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize

from src.ingestion.chunker import Chunker
from src.core.constants import Columns

import logging
logger = logging.getLogger(__name__)

class Preprocessing:
    def __init__(self):
        self.chunker = Chunker()

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

        # Объединение данных
        data = self.merge_data(questions, answers, tags)
        logger.info("После merge_data: shape=%s", data.shape)

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


    def merge_data(self, 
                   questions: pd.DataFrame, 
                   answers: pd.DataFrame, 
                   tags: pd.DataFrame) -> pd.DataFrame:
        """
        Объединяет таблицы questions, answers и tags в единый DataFrame.

        Ожидается:
            - questions: содержит вопросы
            - answers: содержит ответы, связанные по question_id
            - tags: содержит теги, связанные по question_id

        Args:
            questions (pd.DataFrame): Таблица вопросов.
            answers (pd.DataFrame): Таблица ответов.
            tags (pd.DataFrame): Таблица тегов.

        Returns:
            pd.DataFrame: Объединённый DataFrame questions + answers + aggregated tags.

        Raises:
            ValueError: Если входные DataFrame пустые или не содержат обязательные колонки.
            Exception: Если произошла ошибка при объединении.
        """

        try:
            logger.info("Размеры данных: questions_shape=%s, answers_shape=%s, tags_shape=%s",
                        questions.shape, answers.shape, tags.shape)
            
            if questions.empty:
                raise ValueError("questions пустой DataFrame.")
            if answers.empty:
                raise ValueError("answers пустой DataFrame.")
            if tags.empty:
                raise ValueError("tags пустой DataFrame.")



            merged_qa = pd.merge(questions, 
                                  answers, 
                                  on=Columns.QUESTION_ID.value,
                                  how='inner',
                                  suffixes=('_question', '_answer'),
                                  validate="one_to_many")
            
            merged_qa = merged_qa.rename(columns={
                "Score_question": Columns.QUESTION_SCORE.value,
                "Score_answer": Columns.ANSWER_SCORE.value
                })
            
            logger.info("Questions и answers объединены: shape=%s", merged_qa.shape)

            # Агрегация тегов: один Id -> список тегов или строка
            tags_grouped = (
                tags.groupby(Columns.QUESTION_ID.value)[Columns.TAGS.value]
                .agg(lambda x: ', '.join(sorted(set(x.dropna().astype(str)))))
                .reset_index()
            )
            
            logger.info("Теги агрегированы: shape=%s", tags_grouped.shape)
            merged_data = pd.merge(merged_qa,
                                   tags_grouped,
                                   on=Columns.QUESTION_ID.value, 
                                   how='left')

            logger.info("Questions, answers и tags объединены: shape=%s", merged_data.shape)
            
            return merged_data

        except Exception as e:
            logger.error("Не удалось объединить данные. Ошибка: %s", e, exc_info=True)
            raise
    
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

    #def _add_len_words(self, data):
        #data['doc_length_words'] = data[Columns.DOCUMENT_TEXT.value].apply(lambda x: len(word_tokenize(x)))
