from src.ingestion.loader import CSVLoader, DataLoader
from src.core.config import Settings
from src.ingestion.cleaner import Preprocessing

# 
loader = CSVLoader()
settings = Settings()
preprocessing = Preprocessing()

# Загрузка данных
data_question = loader.load_csv(settings.path_question)
data_answers = loader.load_csv(settings.path_answers)
data_tags = loader.load_csv(settings.path_tags)

# Обработка
merged_data = preprocessing.execute(data_question, data_answers, data_tags)
print(merged_data)
#clean_data = preprocessing.clean_html()