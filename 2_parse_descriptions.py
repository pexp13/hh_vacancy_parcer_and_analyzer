# -*- coding: utf-8 -*-

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random

def parse_hh_description(url):
    headers = {
        'User-Agent': random.choice([
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
        ]),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Cache-Control': 'max-age=0',
    }

    try:
        time.sleep(random.uniform(5.0, 10.0))
        
        session = requests.Session()
        r = session.get(url, headers=headers, timeout=20)
        
        # Проверяем статус код
        if r.status_code == 404:
            print(f"  Вакансия не найдена (404) - вероятно удалена")
            return {
                'url': url,
                'description': 'Вакансия удалена/не найдена',
                'skills': 'Вакансия удалена/не найдена'
            }
        
        if r.status_code == 403:
            print(f"  Доступ запрещён (403)")
            return {
                'url': url,
                'description': 'Заблокировано (403)',
                'skills': 'Заблокировано (403)'
            }
        
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'html.parser')

        # Проверяем, не архивная ли вакансия
        archived = soup.find('div', class_='vacancy-archive-warning')
        if archived:
            print(f"  Вакансия в архиве/закрыта")
            return {
                'url': url,
                'description': 'Вакансия закрыта/в архиве',
                'skills': 'Вакансия закрыта/в архиве'
            }

        # Описание вакансии
        description = ''
        desc_block = soup.find('div', {'data-qa': 'vacancy-description'})
        if desc_block:
            description = desc_block.get_text(separator=' ', strip=True)
        
        # Ключевые навыки
        skills = []
        skills_container = soup.find('div', {'data-qa': 'skills-element'})
        if skills_container:
            skill_tags = skills_container.find_all('span', {'data-qa': 'bloko-tag__text'})
            skills = [tag.get_text(strip=True) for tag in skill_tags]
        
        skills_text = ', '.join(skills) if skills else ''

        # Если описание пустое, но страница загрузилась
        if not description:
            print(f"  Предупреждение: описание не найдено")
            description = 'Описание не найдено на странице'

        return {
            'url': url,
            'description': description,
            'skills': skills_text if skills_text else 'Не указаны'
        }

    except requests.exceptions.HTTPError as e:
        status = e.response.status_code
        print(f"  HTTP ошибка {status}")
        return {
            'url': url,
            'description': f'HTTP ошибка {status}',
            'skills': f'HTTP ошибка {status}'
        }
        
    except Exception as e:
        print(f"  Ошибка: {str(e)}")
        return {
            'url': url,
            'description': 'Ошибка парсинга',
            'skills': 'Ошибка парсинга'
        }

# ────────────────────────────────────────────────
# Запуск
# ────────────────────────────────────────────────

input_file = 'vacancy_parsed.csv'
output_file = 'vacancy_with_descriptions.csv'

df = pd.read_csv(input_file)

if 'url' not in df.columns:
    print("Колонка 'url' не найдена. Колонки:", list(df.columns))
    exit()

urls = [str(v).strip() for v in df['url'] if pd.notna(v) and 'hh.ru/vacancy/' in str(v)]

print(f"Найдено {len(urls)} ссылок для парсинга описаний\n")

results = []
for i, url in enumerate(urls, 1):
    print(f"[{i:2d}/{len(urls)}] {url}")
    res = parse_hh_description(url)
    results.append(res)
    
    # Дополнительная пауза каждые 10 вакансий
    if i % 10 == 0 and i < len(urls):
        extra_pause = random.uniform(15.0, 25.0)
        print(f"\n  >>> Пауза после 10 вакансий: {extra_pause:.1f}s\n")
        time.sleep(extra_pause)

desc_df = pd.DataFrame(results)
final_df = df.merge(desc_df, on='url', how='left')

final_df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\nГотово! Результаты в {output_file}")

# Статистика
deleted = len(final_df[final_df['description'].str.contains('удалена|не найдена', case=False, na=False)])
archived = len(final_df[final_df['description'].str.contains('закрыта|архив', case=False, na=False)])
blocked = len(final_df[final_df['description'].str.contains('403', case=False, na=False)])
errors = len(final_df[final_df['description'].str.contains('Ошибка|HTTP ошибка', case=False, na=False)])
success = len(final_df) - deleted - archived - blocked - errors

print(f"\nСтатистика:")
print(f"  ✅ Успешно спарсено: {success}")
print(f"  🗑️  Удалены/не найдены: {deleted}")
print(f"  📦 Закрыты/архив: {archived}")
print(f"  🚫 Заблокировано (403): {blocked}")
print(f"  ❌ Другие ошибки: {errors}")

if success > 0:
    print("\n" + "="*80)
    print("Пример первой успешной вакансии:")
    successful = final_df[~final_df['description'].str.contains('удалена|закрыта|Ошибка|403|не найдена', case=False, na=False)]
    if len(successful) > 0:
        first = successful.iloc[0]
        print(f"Название: {first['title']}")
        print(f"Описание (первые 300 символов):\n{first['description'][:300]}...")
        print(f"Навыки: {first['skills']}")
