# -*- coding: utf-8 -*-

import pandas as pd
import re

def clean_salary(s):
    if pd.isna(s) or s in ['Не указана', 'Не указано', 'Ошибка', '']:
        return 'Не указана'
    
    s = s.replace('\xa0', ' ').replace('\u202f', ' ')
    
    # Самые частые слипания из твоих примеров
    s = re.sub(r'([отдо])(\d)', r'\1 \2', s)                # от55 → от 55
    s = re.sub(r'(\d)(до)', r'\1 до', s)                    # 000до → 000 до
    s = re.sub(r'(\d)(₽)', r'\1 ₽', s)                      # 000₽ → 000 ₽
    s = re.sub(r'(₽)(за|до|на)', r'\1 \2', s)               # ₽за → ₽ за
    s = re.sub(r'(месяц)(на|до|за)', r'\1 \2', s)           # месяцна → месяц на
    s = re.sub(r'(на руки)', r' на руки', s)
    s = re.sub(r'(до вычета налогов)', r' до вычета налогов', s)
    
    # Убираем лишние пробелы и чистим
    s = re.sub(r'\s+', ' ', s)
    s = s.strip(',.; ')
    
    return s

def clean_experience(s):
    if pd.isna(s) or s in ['Не указано', 'Ошибка', '']:
        return 'Не указано'
    
    s = s.replace('\xa0', ' ').replace('\u202f', ' ')
    s = re.sub(r'(\d)–(\d)', r'\1 – \2', s)                 # 1–3 → 1 – 3
    s = re.sub(r'(не требуется)', r'не требуется', s)
    return re.sub(r'\s+', ' ', s).strip()

def clean_employment(s):
    if pd.isna(s) or s in ['Не указано', 'Ошибка', '']:
        return 'Не указано'
    s = s.replace('\xa0', ' ').replace('\u202f', ' ')
    return re.sub(r'\s+', ' ', s).strip()

def clean_schedule(s):
    if pd.isna(s) or s in ['Не указано', 'Ошибка', '']:
        return 'Не указан'
    
    s = s.replace('\xa0', ' ').replace('\u202f', ' ')
    s = s.replace('График:', 'График: ').strip()
    
    if s == 'График:':
        return 'Не указан'
    
    return re.sub(r'\s+', ' ', s).strip()

def normalize_format(s):
    if pd.isna(s) or s in ['Не указано', 'Ошибка', '']:
        return 'Не указано'
    
    s = s.replace('\xa0', ' ').replace('\u202f', ' ')
    low = s.lower().strip()
    
    mapping = {
        'удалённо':                     'Удалённая работа',
        'удалённая работа':             'Удалённая работа',
        'remote':                       'Удалённая работа',
        'гибрид':                       'Гибридный график',
        'удалённо или гибрид':          'Удалённая работа или гибрид',
        'в офисе':                      'Офис (на месте)',
        'офис':                         'Офис (на месте)',
        'на месте':                     'Офис (на месте)',
        'on-site':                      'Офис (на месте)',
    }
    
    for k, v in mapping.items():
        if k in low:
            return v
    
    return re.sub(r'\s+', ' ', s).strip() or 'Не указано'

def clean_title(s):
    if pd.isna(s) or s in ['Не удалось получить название', '']:
        return 'Без названия'
    s = s.replace('\xa0', ' ').replace('\u202f', ' ')
    return re.sub(r'\s+', ' ', s).strip()

# ────────────────────────────────────────────────

input_file = 'vacancy_parsed_v3_final.csv'     # ← имя твоего файла после парсинга
output_file = 'vacancy_clean_v2.csv'

df = pd.read_csv(input_file)

# Применяем очистку
df['title']        = df['title'].apply(clean_title)
df['salary']       = df['salary'].apply(clean_salary)
df['experience']   = df['experience'].apply(clean_experience)
df['employment']   = df['employment'].apply(clean_employment)
df['schedule']     = df['schedule'].apply(clean_schedule)
df['work_format']  = df['work_format'].apply(normalize_format)

# Сохраняем
df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"Готово! Очень красивая версия сохранена в:\n{output_file}")

# Показываем примеры в консоли
print("\nРезультаты после очистки (первые 3 строки):\n")
print("─" * 100)

for i, row in df.head(3).iterrows():
    print(f"Вакансия:   {row['title']}")
    print(f"Зарплата:   {row['salary']}")
    print(f"Опыт:       {row['experience']}")
    print(f"Занятость:  {row['employment']}")
    print(f"График:     {row['schedule']}")
    print(f"Формат:     {row['work_format']}")
    print("─" * 100)
