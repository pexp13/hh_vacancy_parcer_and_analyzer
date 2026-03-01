# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from transformers import pipeline
from tqdm import tqdm
import torch

# Загружаем данные
df_clean = pd.read_csv('vacancy_with_clusters.csv')
embeddings = np.load('vacancy_embeddings.npy')

print(f"Загружено {len(df_clean)} вакансий\n")

# ════════════════════════════════════════════════════════════════
# ZERO-SHOT КЛАССИФИКАЦИЯ (ЛеГКАЯ МОДЕЛЬ)
# ════════════════════════════════════════════════════════════════

print("Загружаем легкую модель для zero-shot классификации...")
print("(~400 МБ, потребление ОЗУ ~1.5 ГБ)\n")

# Проверяем наличие GPU
device = 0 if torch.cuda.is_available() else -1
print(f"Используем: {'GPU (CUDA)' if device == 0 else 'CPU'}\n")

# Загружаем zero-shot classifier (легкая multilingual модель)
classifier = pipeline(
    "zero-shot-classification",
    model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    device=device
)

# ════════════════════════════════════════════════════════════════
# ОПРЕДЕЛЯЕМ МЕТКИ ДЛЯ КЛАССИФИКАЦИИ
# ════════════════════════════════════════════════════════════════

# Две категории для классификации (можно на русском - модель multilingual)
# ⬇️ Замените на свои описания категорий
candidate_labels = [
    "описание первой категории вакансий",   # → будет отображаться как "Категория А"
    "описание второй категории вакансий"    # → будет отображаться как "Категория Б"
]

# Короткие названия для отображения
label_names = {
    candidate_labels[0]: "Категория А",
    candidate_labels[1]: "Категория Б"
}

# ════════════════════════════════════════════════════════════════
# КЛАССИФИЦИРУЕМ КАЖДУЮ ВАКАНСИЮ
# ════════════════════════════════════════════════════════════════

print("Классифицируем вакансии...")
print("="*80)

results = []

for idx, row in tqdm(df_clean.iterrows(), total=len(df_clean), desc="Обработка"):
    # Берем только начало текста (модель имеет ограничение на длину)
    text = row['full_text'][:800]  # Первые 800 символов (достаточно)
    
    # Классифицируем
    result = classifier(text, candidate_labels, multi_label=False)
    
    # Извлекаем результаты
    top_label = result['labels'][0]  # Самая вероятная метка
    top_score = result['scores'][0]  # Ее вероятность (0-1)
    
    # Сохраняем
    results.append({
        'label': label_names[top_label],
        'confidence': top_score,
        'is_linguistic': top_label == candidate_labels[0]
    })

# Добавляем результаты в таблицу
df_clean['category'] = [r['label'] for r in results]
df_clean['confidence'] = [r['confidence'] for r in results]
df_clean['is_linguistic'] = [r['is_linguistic'] for r in results]

# ════════════════════════════════════════════════════════════════
# АНАЛИЗ РЕЗУЛЬТАТОВ
# ════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("РЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ")
print("="*80)

ling_count = df_clean['is_linguistic'].sum()
non_ling_count = len(df_clean) - ling_count

print(f"\n📊 Статистика:")
print(f"   Категория А: {ling_count} вакансий ({ling_count/len(df_clean)*100:.1f}%)")
print(f"   Категория Б: {non_ling_count} вакансий ({non_ling_count/len(df_clean)*100:.1f}%)")

# Средняя уверенность модели
avg_confidence = df_clean['confidence'].mean()
print(f"\n💯 Средняя уверенность модели: {avg_confidence:.2%}")

# ════════════════════════════════════════════════════════════════
# ПРИМЕРЫ Классификация 1 ВАКАНСИЙ
# ════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("📁 ВАКАНСИИ — КАТЕГОРИЯ А:")
print("="*80)

ling_jobs = df_clean[df_clean['is_linguistic']].sort_values('confidence', ascending=False)

if len(ling_jobs) > 0:
    for i, row in enumerate(ling_jobs.head(10).itertuples(), 1):
        print(f"\n{i}. {row.title}")
        print(f"   Уверенность: {row.confidence:.1%}")
        print(f"   Зарплата: {row.salary}")
        print(f"   Кластер: {row.cluster}")
        print(f"   Описание (краткое): {row.description[:150]}...")
else:
    print("Вакансий в категории А не найдено")

# ════════════════════════════════════════════════════════════════
# ПРИМЕРЫ Классификация 2 ВАКАНСИЙ
# ════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("📁 ВАКАНСИИ — КАТЕГОРИЯ Б:")
print("="*80)

non_ling_jobs = df_clean[~df_clean['is_linguistic']].sort_values('confidence', ascending=False)

if len(non_ling_jobs) > 0:
    for i, row in enumerate(non_ling_jobs.head(10).itertuples(), 1):
        print(f"\n{i}. {row.title}")
        print(f"   Уверенность: {row.confidence:.1%}")
        print(f"   Зарплата: {row.salary}")
        print(f"   Кластер: {row.cluster}")
        print(f"   Описание (краткое): {row.description[:150]}...")
else:
    print("Вакансий в категории Б не найдено")

# ════════════════════════════════════════════════════════════════
# СРАВНЕНИЕ С КЛАСТЕРАМИ
# ════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("📊 СРАВНЕНИЕ: КЛАСТЕРЫ vs КЛАССИФИКАЦИЯ")
print("="*80)

comparison = pd.crosstab(
    df_clean['cluster'], 
    df_clean['category'],
    margins=True
)
print(f"\n{comparison}")

# ════════════════════════════════════════════════════════════════
# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ════════════════════════════════════════════════════════════════

df_clean.to_csv('vacancy_classified.csv', index=False, encoding='utf-8-sig')
print("\n✅ Результаты сохранены в vacancy_classified.csv")

# Создаем отчет для портфолио
report = f"""
ОТЧЕТ: АНАЛИЗ ВАКАНСИЙ С МАШИННЫМ ОБУЧЕНИЕМ
============================================

ДАТАСЕТ:
- Всего вакансий: {len(df_clean)}
- Источник: HeadHunter (hh.ru)

МЕТОДЫ:
1. Sentence Transformers (paraphrase-multilingual-MiniLM-L12-v2)
   - Векторизация текстов в пространство 384 измерений
   
2. UMAP (Uniform Manifold Approximation and Projection)
   - Снижение размерности 384D → 2D для визуализации
   
3. K-Means кластеризация
   - Автоматическое разбиение на 2 кластера
   
4. Zero-shot classification (mDeBERTa-v3-base-xnli)
   - Классификация без обучающей выборки
   - Multilingual модель с поддержкой русского языка

РЕЗУЛЬТАТЫ:
- Категория А: {ling_count} ({ling_count/len(df_clean)*100:.1f}%)
- Категория Б: {non_ling_count} ({non_ling_count/len(df_clean)*100:.1f}%)
- Средняя уверенность модели: {avg_confidence:.1%}

ФАЙЛЫ:
- vacancy_classified.csv - полная таблица с классификацией
- vacancy_map.html - интерактивная визуализация
- vacancy_embeddings.npy - векторные представления

ТОП-5 ВАКАНСИЙ — КАТЕГОРИЯ А:
{chr(10).join([f"{i+1}. {row.title} (уверенность: {row.confidence:.1%})" for i, row in enumerate(ling_jobs.head(5).itertuples())])}

ТОП-5 ВАКАНСИЙ — КАТЕГОРИЯ Б:
{chr(10).join([f"{i+1}. {row.title} (уверенность: {row.confidence:.1%})" for i, row in enumerate(non_ling_jobs.head(5).itertuples())])}
"""

with open('REPORT.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print("\n Отчет сохранен в REPORT.txt")
print("\n Анализ завершен! Проверьте файлы:")
print("   - vacancy_classified.csv (итоговая таблица)")
print("   - vacancy_map.html (интерактивный график)")
print("   - REPORT.txt (отчет для портфолио)")
