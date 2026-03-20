# Шаг 2: Предобработка данных
data = self.preprocessing.preprocess_data(questions, answers, tags) # type: ignore

# Шаг 3: Генерация эмбеддингов
embeddings = self.embedding_generator.generate_embeddings(data['document_text'].tolist()) # type: ignore

# Шаг 4: Разбиение на чанки
chunks_data = []
for _, row in data.iterrows():
    chunks = self.chunker.chunk_document(row['document_text']) # type: ignore
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
            'chunk_text': chunk,
            'embedding': embeddings[idx] 
        }

        chunks_data.append(chunk_data)

# Сохраняем чанки в базу данных
chunks_df = pd.DataFrame(chunks_data) # type: ignore
#self.db_saver.insert_to_db(chunks_df)




# Шаг 2: Предобработка
data = self.preprocessing.preprocess_data(questions, answers, tags) # type: ignore

# Шаг 3: Чанкинг
chunks_data = []

for _, row in data.iterrows():
    chunks = self.chunker.chunk_document(row["document_text"]) # type: ignore

    for idx, chunk in enumerate(chunks):
        chunks_data.append({
            "chunk_id": f"q{row['Id']}_a{row['Id_answer']}_c{idx}",
            "question_id": row["Id"],
            "answer_id": row["Id_answer"],
            "chunk_index": idx,
            "title": row["Title"],
            "tags": row["Tag"],
            "question_score": row["Score_question"],
            "answer_score": row["Score_answer"],
            "chunk_text": chunk,
        })

# Шаг 4: Генерация эмбеддингов (по чанкам!)
chunk_texts = [row["chunk_text"] for row in chunks_data]
embeddings = self.embedding_generator.generate_embeddings(chunk_texts) # type: ignore

# Присваиваем embeddings
for i in range(len(chunks_data)):
    chunks_data[i]["embedding"] = embeddings[i]

# Шаг 5: DataFrame
chunks_df = pd.DataFrame(chunks_data) # type: ignore

return chunks_df # type: ignore
