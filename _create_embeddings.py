# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import umap
import os
os.environ ['HF_token'] = "YOUR_HF_TOKEN"

# Загрузим данные
df = pd.read_csv('vacancy_with_descriptions.csv')

# Фильтруем только успешные вакансии
df_clean = df[~df['description'].str.contains('удалена|закрыта|Ошибка|403|не найдена|Описание не найдено', case=False, na=False)].copy()
df_clean.reset_index(drop=True, inplace=True)

print(f"Работаем с {len(df_clean)} вакансиями\n")

# Создаём объединённый текст для анализа
df_clean['full_text'] = (
    df_clean['title'].fillna('') + '. ' + 
    df_clean['description'].fillna('') + '. ' + 
    'Навыки: ' + df_clean['skills'].fillna('')
)

# Посмотрим на длину текстов
df_clean['text_length'] = df_clean['full_text'].str.len()
print(f"Средняя длина текста: {df_clean['text_length'].mean():.0f} символов")
print(f"Мин: {df_clean['text_length'].min()}, Макс: {df_clean['text_length'].max()}")

print("\n" + "="*80)
print("Данные подготовлены! Теперь создаём embeddings...")
print("Это займёт 1-3 минуты в зависимости от железа...")

# Загружаем multilingual модель для русского текста
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Создаём embeddings
embeddings = model.encode(df_clean['full_text'].tolist(), 
                          show_progress_bar=True,
                          batch_size=8)

print(f"\nEmbeddings созданы! Размерность: {embeddings.shape}")

# Сохраняем
np.save('vacancy_embeddings.npy', embeddings)
df_clean.to_csv('vacancy_clean.csv', index=False, encoding='utf-8-sig')

print("\nФайлы сохранены:")
print("  - vacancy_embeddings.npy (векторы)")
print("  - vacancy_clean.csv (чистые данные)")
