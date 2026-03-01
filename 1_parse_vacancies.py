# -*- coding: utf-8 -*-

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random

def parse_hh_vacancy(url):
    headers = {
        'User-Agent': random.choice([
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0',
        ]),
        'Accept-Language': 'ru-RU,ru;q=0.9,en;q=0.8',
    }

    try:
        time.sleep(random.uniform(4.0, 9.0))
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'html.parser')

        title = soup.find('h1', {'data-qa': 'vacancy-title'})
        title = title.get_text(strip=True) if title else 'Не удалось получить название'

        salary_elem = soup.find('div', {'data-qa': 'vacancy-salary'})
        salary = salary_elem.get_text(strip=True).replace('\xa0', ' ').replace('\u202f', ' ') if salary_elem else 'Не указана'

        exp_elem = soup.find('span', {'data-qa': 'vacancy-experience'})
        experience = exp_elem.get_text(strip=True) if exp_elem else 'Не указано'

        # ────────────────────────────────────────────────
        # Занятость / График / Формат — простая и надёжная логика
        employment = 'Не указано'
        schedule = 'Не указано'
        work_format = 'Не указано'

        # Основной блок + ограничение по длине
        main = (
            soup.find('div', {'data-qa': 'vacancy-view'}) or
            soup.find('div', class_='vacancy-view-content') or
            soup.find('div', class_='bloko-column_m-9') or
            soup.find('div', class_='bloko-grid-container') or
            soup
        )
        text_block = main.get_text(separator='\n', strip=True)[:3000]
        lines = [l.strip() for l in text_block.split('\n') if l.strip() and len(l.strip()) < 120]  # короткие строки — характеристики

        for line in lines:
            low = line.lower()

            if employment == 'Не указано' and any(w in low for w in ['полная занятость', 'частичная занятость', 'проектная работа', 'проект или разовое задание', 'стажировка']):
                employment = line

            if schedule == 'Не указано' and any(w in low for w in ['полный день', 'сменный график', 'гибкий график', 'ненормированный', 'вахтовый', '5/2', '2/2', 'график']):
                schedule = line

            # Формат работы — ищем отдельно
            if work_format == 'Не указано' and any(w in low for w in ['удалённая работа', 'удалённо', 'remote', 'гибрид', 'гибридный', 'в офисе', 'офис', 'на месте', 'on-site']):
                work_format = line

        # Если формат содержит "или" — оставляем как есть
        # Если ничего не нашли — остаётся "Не указано"

        return {
            'url': url,
            'title': title,
            'salary': salary,
            'experience': experience,
            'employment': employment,
            'schedule': schedule,
            'work_format': work_format
        }

    except Exception as e:
        print(f"Ошибка {url}: {str(e)}")
        return {
            'url': url,
            'title': 'Ошибка',
            'salary': 'Ошибка',
            'experience': 'Ошибка',
            'employment': 'Ошибка',
            'schedule': 'Ошибка',
            'work_format': 'Ошибка'
        }

# ────────────────────────────────────────────────
# Запуск
# ────────────────────────────────────────────────

input_file = 'vacancy.csv'
output_file = 'vacancy_parsed'

df = pd.read_csv(input_file)

link_column = next(
    (col for col in df.columns if any(w in col.lower() for w in ['url', 'ссылк', 'link', 'vacancy', 'hh', 'адрес'])),
    None
)

if not link_column:
    print("Колонка со ссылками не найдена. Колонки:", list(df.columns))
    exit()

urls = [str(v).strip() for v in df[link_column] if pd.notna(v) and 'hh.ru/vacancy/' in str(v)]

print(f"Найдено {len(urls)} ссылок")

results = []
for i, url in enumerate(urls, 1):
    print(f"[{i:2d}/{len(urls)}] {url}")
    res = parse_hh_vacancy(url)
    results.append(res)

pd.DataFrame(results).to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\nГотово! Результаты в {output_file}")
