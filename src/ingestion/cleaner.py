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

    def preprocess_data(self, questions, answers, tags):
        # Объединение данных
        data = self.merge_data(questions, answers, tags)
        if data is None or data.empty:
            logger.error('После объединения данные пустые.')
            return None

        # Оставляем только лучший ответ
        data = self.select_best_answers(data)

        # Шаг 3. Убираем в тексте теги
        data[Columns.BODY_QUESTION] = data[Columns.BODY_QUESTION].apply(self.clean_html)
        data[Columns.BODY_ANSWER] = data[Columns.BODY_ANSWER].apply(self.clean_html)

        # Собираем документ
        data[Columns.DOCUMENT_TEXT] = data.apply(self.build_documents, axis=1)

        return data


    def merge_data(self, questions, answers, tags):
        try:
            merged_qa = (pd.merge(questions, answers, left_on='Id', right_on='ParentId', suffixes=('_question', '_answer'))
                         .drop(columns=['ParentId']))
            logger.info('Questions и answers объединены.')

            # Агрегация тегов: один Id -> список тегов или строка
            tags_grouped = (tags
                            .groupby('Id', as_index=False)['Tag']
                            .apply(lambda x: ', '.join(sorted(set(x.dropna().astype(str))))))

            merged_data = (pd.merge(merged_qa,tags_grouped,left_on=Columns.QUESTION_ID,right_on='Id',how='left')
                           .drop(columns=['Id']))

            logger.info('Questions, answers и tags объединены.')
            
            return merged_data

        except Exception as e:
            logger.error(f'Не удалось объединить данные. Ошибка: {e}')
            return None
        
    def to_datetime(self, data, column):
        return pd.to_datetime(data[column], format='%Y-%m-%dT%H:%M:%SZ', errors='coerce')
    
    def select_best_answers(self, data):
        best_answers = (data
                        .sort_values(['Id_question', 'Score_answer'], ascending=[True, False])
                        .drop_duplicates(subset='Id_question', keep='first')
                        .reset_index(drop=True))
        
        logger.info(f'Оставлен только лучший ответ.')
        return best_answers

    def clean_html(self, text):
        if pd.isna(text):
            return ""
        
        try:
            text = BeautifulSoup(text, "html.parser").get_text(" ", strip=True)
            return text
        except Exception as e:
            logger.error(f'Не удалось отчистить данные. Ошибка: {e}')
            return ""

    def build_documents(self, row):
        return (
        f"Title: {row['Title']}\n\n"
        f"Tags: {row['Tag']}\n\n"
        f"Question: {row['Body_question']}\n\n"
        f"Best Answer: {row['Body_answer']}"
    )

    def build_chunk(self, data):
        chunks_data = []

        for _, row in data.iterrows():
            # Разбиваем document_text на чанки
            chunks = self.chunker.chunk_document(row[Columns.DOCUMENT_TEXT])
            
            # Для каждого чанка создаем запись
            for idx, chunk in enumerate(chunks):
                chunk_data = {
                    Columns.CHUNK_ID : f"q{row['Id_question']}_a{row['Id_answer']}_c{idx}",
                    Columns.QUESTION_ID : row['Id_question'],
                    Columns.ANSWER_ID : row['Id_answer'],
                    Columns.CHUNK_INDEX : idx,
                    Columns.TITLE : row['Title'],
                    Columns.TAGS : row['Tag'],
                    Columns.QUESTION_SCORE : row['Score_question'],
                    Columns.ANSWER_SCORE: row['Score_answer'],
                    Columns.CHUNK_TEXT: chunk
                }
                chunks_data.append(chunk_data)

        # Преобразуем список в DataFrame
        chunks_df = pd.DataFrame(chunks_data)

        # Вставляем данные в БД
        self.loader.insert_chunks_to_db(chunks_df)

    def _add_len_words(self, data):
        data['doc_length_words'] = data[Columns.DOCUMENT_TEXT].apply(lambda x: len(word_tokenize(x)))
