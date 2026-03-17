import pandas as pd
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
import logging

from src.ingestion.chunker import Chunker
from src.ingestion.loader import DataLoader
#from src.indexing.embedder import encode

logger = logging.getLogger(__name__)

class Preprocessing:
    def __init__(self):
        self.datetime_columns = ['CreationDate_answer', 'ClosedDate', 'CreationDate_question']
        self.chunker = Chunker()
        self.loader_data = DataLoader()
        self.columns = ['OwnerUserId_question', 'OwnerUserId_answer',  'CreationDate_question',  'CreationDate_answer',
                           'ClosedDate', 'Body_question',  'Body_answer']

    def preprocess_data(self, questions, answers, tags):
        # Объединение данных
        data = self.merge_data(questions, answers, tags)
        if data is None or data.empty:
            logger.error('После объединения данные пустые.')
            return None

        # Оставляем только лучший ответ
        data = self.select_best_answers(data)

        # Шаг 3. Убираем в тексте теги
        data["Body_question"] = data["Body_question"].apply(self.clean_html)
        data["Body_answer"] = data["Body_answer"].apply(self.clean_html)

        # Собираем документ
        data["document_text"] = data.apply(self.build_documents, axis=1)

        # Облегчение данных
        data = data.drop(columns=self.columns, axis=1)

        return data


    def merge_data(self, questions, answers, tags):
        try:
            merged_qa = (pd.merge(questions, answers, left_on='Id', right_on='ParentId', suffixes=('_question', '_answer'))
                         .drop(columns=['ParentId']))
            logger.info('Questions и answers объединены.')

            # Агрегация тегов: один Id -> список тегов или строка
            tags_grouped = (
                tags.groupby('Id', as_index=False)['Tag']
                .apply(lambda x: ', '.join(sorted(set(x.dropna().astype(str)))))
            )

            merged_data = (pd.merge(merged_qa,tags_grouped,left_on='Id_question',right_on='Id',how='left')
                           .drop(columns=['Id']))

            for col in self.datetime_columns:
                if col in merged_data.columns:
                    merged_data[col] = self.to_datetime(merged_data, col)

            merged_data.rename(columns={'Id_question': 'Id'}, inplace=True)

            logger.info('Questions, answers и tags объединены.')
            
            return merged_data

        except Exception as e:
            logger.error(f'Не удалось объединить данные. Ошибка: {e}')
            return None
        
    def to_datetime(self, data, column):
        return pd.to_datetime(data[column], format='%Y-%m-%dT%H:%M:%SZ', errors='coerce')
    
    def select_best_answers(self, data):
        best_answers = (data
                        .sort_values(['Id', 'Score_answer'], ascending=[True, False])
                        .drop_duplicates(subset='Id', keep='first')
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
            chunks = self.chunker.chunk_document(row['document_text'])
            
            # Для каждого чанка создаем запись
            for idx, chunk in enumerate(chunks):
                chunk_data = {
                    'chunk_id': f"q{row['Id']}_a{row['Id_answer']}_c{idx}",
                    'question_id': row['Id'],
                    'answer_id': row['Id_answer'],
                    'chunk_index': idx,
                    'title': row['Title'],
                    'tags': row['Tag'],
                    'question_score': row['Score_question'],
                    'answer_score': row['Score_answer'],
                    'chunk_text': chunk
                }
                chunks_data.append(chunk_data)

        # Преобразуем список в DataFrame
        chunks_df = pd.DataFrame(chunks_data)

        # Вставляем данные в БД
        self.loader.insert_chunks_to_db(chunks_df)

    def _add_len_words(self, data):
        data['doc_length_words'] = data['document_text'].apply(lambda x: len(word_tokenize(x)))
