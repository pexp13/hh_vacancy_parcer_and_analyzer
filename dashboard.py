# -*- coding: utf-8 -*-
# dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ════════════════════════════════════════════════════════════════
# НАСТРОЙКА СТРАНИЦЫ
# ════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Анализ вакансий ML",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ════════════════════════════════════════════════════════════════
# ЗАГРУЗКА ДАННЫХ
# ════════════════════════════════════════════════════════════════

@st.cache_data
def load_data():
    """Загружаем данные (с кэшированием для скорости)"""
    # Пробуем загрузить разные версии файла
    for filename in ['vacancy_classified_manual.csv', 'vacancy_classified_v2.csv', 'vacancy_classified.csv']:
        try:
            df = pd.read_csv(filename)
            embeddings = np.load('vacancy_embeddings.npy')
            return df, embeddings, filename
        except FileNotFoundError:
            continue
    
    st.error("❌ Файл с классификацией не найден!")
    st.stop()

df, embeddings, current_file = load_data()

# ════════════════════════════════════════════════════════════════
# ФУНКЦИЯ СОХРАНЕНИЯ ИЗМЕНЕНИЙ
# ════════════════════════════════════════════════════════════════

def save_changes(df, filename):
    """Сохраняет изменения в CSV"""
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    st.cache_data.clear()  # Очищаем кэш чтобы перезагрузить данные

# ════════════════════════════════════════════════════════════════
# ЗАГОЛОВОК
# ════════════════════════════════════════════════════════════════

st.title("📊 Анализ вакансий с Machine Learning")
st.markdown(f"*Файл данных: `{current_file}`*")
st.markdown("---")

# ════════════════════════════════════════════════════════════════
# САЙДБАР - ФИЛЬТРЫ
# ════════════════════════════════════════════════════════════════

st.sidebar.header("🔍 Фильтры")

# Фильтр по категории
category_filter = st.sidebar.multiselect(
    "Категория вакансий:",
    options=df['category'].unique(),
    default=df['category'].unique()
)

# Фильтр по формату работы
work_format_filter = st.sidebar.multiselect(
    "Формат работы:",
    options=df['work_format'].unique(),
    default=df['work_format'].unique()
)

# Фильтр по уверенности модели
confidence_threshold = st.sidebar.slider(
    "Минимальная уверенность модели:",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.05,
    format="%.0f%%"
)

# Применяем фильтры
df_filtered = df[
    (df['category'].isin(category_filter)) &
    (df['work_format'].isin(work_format_filter)) &
    (df['confidence'] >= confidence_threshold)
]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Показано вакансий:** {len(df_filtered)} из {len(df)}")

# ════════════════════════════════════════════════════════════════
# СТАТИСТИКА (МЕТРИКИ)
# ════════════════════════════════════════════════════════════════

st.header("📈 Общая статистика")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Всего вакансий",
        len(df),
        delta=None
    )

with col2:
    ling_count = df['is_linguistic'].sum()
    st.metric(
        "Классификация 1",
        ling_count,
        delta=f"{ling_count/len(df)*100:.1f}%"
    )

with col3:
    non_ling_count = len(df) - ling_count
    st.metric(
        "Классификация 2",
        non_ling_count,
        delta=f"{non_ling_count/len(df)*100:.1f}%"
    )

with col4:
    avg_conf = df['confidence'].mean()
    st.metric(
        "Средняя уверенность",
        f"{avg_conf:.1%}",
        delta=None
    )

st.markdown("---")

# ════════════════════════════════════════════════════════════════
# ИНТЕРАКТИВНАЯ КАРТА
# ════════════════════════════════════════════════════════════════

st.header("🗺️ Карта вакансий (UMAP визуализация)")

with st.expander("ℹ️ Как читать карту", expanded=False):
    st.markdown("""
    **Что показывает карта:**
    - Каждая точка = одна вакансия
    - **Близкие точки** = похожие вакансии по содержанию
    - **Далёкие точки** = разные типы работ
    - Цвет = категория
    
    **Что означают оси X и Y:**
    - Это **абстрактные координаты** после сжатия 384-мерного пространства в 2D
    - Сами числа (-5, 0, +3) **не имеют прямого значения**
    - Важно только **взаимное расположение** точек (близко/далеко)
    
    **Как это работает:**
    1. Каждое описание вакансии превращается в вектор из 384 чисел (embeddings)
    2. UMAP "сжимает" эти 384 числа в 2 координаты (X, Y)
    3. При сжатии сохраняется главное: похожие вакансии остаются рядом
    
    **Пример интерпретации:**
    - Плотная группа точек слева → все про редактуру и тексты
    - Плотная группа справа → все про данные и аналитику
    - Расстояние между группами → насколько разные эти типы работ
    """)

# Создаём hover text
df_filtered['hover_text'] = (
    '<b>' + df_filtered['title'] + '</b><br>' +
    'Категория: ' + df_filtered['category'] + '<br>' +
    'Уверенность: ' + (df_filtered['confidence']*100).round(1).astype(str) + '%<br>' +
    'Зарплата: ' + df_filtered['salary'].astype(str) + '<br>' +
    'Опыт: ' + df_filtered['experience'].astype(str) + '<br>' +
    'Формат: ' + df_filtered['work_format'].astype(str)
)

# Создаём scatter plot
fig = px.scatter(
    df_filtered,
    x='x',
    y='y',
    color='category',
    hover_data={'hover_text': True, 'x': False, 'y': False, 'category': False},
    color_discrete_map={
        'Классификация 1': '#FF6B6B',
        'Классификация 2': '#4ECDC4'
    },
    title='',
    height=600
)

fig.update_traces(
    marker=dict(size=14, opacity=0.8, line=dict(width=1, color='white')),
    hovertemplate='%{customdata[0]}<extra></extra>'
)

fig.update_layout(
    xaxis_title='UMAP Измерение 1 (абстрактная координата)',
    yaxis_title='UMAP Измерение 2 (абстрактная координата)',
    font=dict(size=12),
    plot_bgcolor='white',
    hovermode='closest',
    legend_title_text='Категория'
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ════════════════════════════════════════════════════════════════
# РАСПРЕДЕЛЕНИЕ ПО КАТЕГОРИЯМ
# ════════════════════════════════════════════════════════════════

st.header("📊 Распределение вакансий")

col1, col2 = st.columns(2)

with col1:
    category_counts = df_filtered['category'].value_counts()
    
    fig_pie = px.pie(
        values=category_counts.values,
        names=category_counts.index,
        title='Распределение по категориям',
        color_discrete_map={
            'Классификация 1': '#FF6B6B',
            'Классификация 2': '#4ECDC4'
        },
        hole=0.4
    )
    
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    fig_hist = px.histogram(
        df_filtered,
        x='confidence',
        nbins=20,
        title='Распределение уверенности модели',
        labels={'confidence': 'Уверенность', 'count': 'Количество'},
        color_discrete_sequence=['#6C5CE7']
    )
    
    fig_hist.update_layout(
        xaxis_title='Уверенность модели',
        yaxis_title='Количество вакансий',
        showlegend=False
    )
    
    st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("---")

# ════════════════════════════════════════════════════════════════
# ТАБЛИЦА ВАКАНСИЙ
# ════════════════════════════════════════════════════════════════

st.header("📋 Список вакансий")

table_category = st.radio(
    "Показать:",
    options=['Все', 'Классификация 1', 'Классификация 2'],
    horizontal=True
)

if table_category == 'Классификация 1':
    df_table = df_filtered[df_filtered['is_linguistic']]
elif table_category == 'Классификация 2':
    df_table = df_filtered[~df_filtered['is_linguistic']]
else:
    df_table = df_filtered

df_table = df_table.sort_values('confidence', ascending=False)

st.dataframe(
    df_table[[
        'title', 'category', 'confidence', 'salary', 
        'experience', 'work_format', 'employment'
    ]].rename(columns={
        'title': 'Название',
        'category': 'Категория',
        'confidence': 'Уверенность',
        'salary': 'Зарплата',
        'experience': 'Опыт',
        'work_format': 'Формат работы',
        'employment': 'Занятость'
    }),
    use_container_width=True,
    hide_index=True
)

csv = df_table.to_csv(index=False, encoding='utf-8-sig')
st.download_button(
    label="📥 Скачать отфильтрованные вакансии (CSV)",
    data=csv,
    file_name='filtered_vacancies.csv',
    mime='text/csv'
)

st.markdown("---")

# ════════════════════════════════════════════════════════════════
# ДЕТАЛИ ВАКАНСИИ + РЕДАКТИРОВАНИЕ
# ════════════════════════════════════════════════════════════════

st.header("🔍 Просмотр и редактирование вакансии")

vacancy_titles = df_filtered['title'].tolist()
selected_vacancy = st.selectbox(
    "Выберите вакансию:",
    options=vacancy_titles,
    key='vacancy_selector'
)

if selected_vacancy:
    vacancy_idx = df_filtered[df_filtered['title'] == selected_vacancy].index[0]
    vacancy_data = df.loc[vacancy_idx]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(vacancy_data['title'])
        
        # Показываем текущую категорию с индикатором
        if vacancy_data['confidence'] < 0.7 and vacancy_data.get('classification_method') == 'model':
            st.warning(f"⚠️ **Категория:** {vacancy_data['category']} (уверенность: {vacancy_data['confidence']:.1%}) - низкая уверенность!")
        else:
            st.success(f"✅ **Категория:** {vacancy_data['category']} (уверенность: {vacancy_data['confidence']:.1%})")
        
        st.markdown(f"**Зарплата:** {vacancy_data['salary']}")
        st.markdown(f"**Опыт:** {vacancy_data['experience']}")
        st.markdown(f"**Формат работы:** {vacancy_data['work_format']}")
        st.markdown(f"**Занятость:** {vacancy_data['employment']}")
        st.markdown(f"**График:** {vacancy_data['schedule']}")
        
        if 'classification_method' in vacancy_data:
            method_labels = {
                'manual': '✍️ Ручная разметка',
                'model': '🤖 Модель ML',
                'rule': '📏 Правила',
                'manual_review': '✍️ Ручная проверка'
            }
            method = vacancy_data['classification_method']
            st.markdown(f"**Метод классификации:** {method_labels.get(method, method)}")
        
        st.markdown("---")
        st.markdown("**Описание:**")
        st.write(vacancy_data['description'])
        
    with col2:
        st.markdown("**Навыки:**")
        if pd.notna(vacancy_data['skills']) and vacancy_data['skills'] != 'Не указаны':
            skills = vacancy_data['skills'].split(', ')
            for skill in skills:
                st.markdown(f"- {skill}")
        else:
            st.write("Не указаны")
        
        st.markdown("---")
        st.markdown(f"**[Открыть вакансию на HH.ru]({vacancy_data['url']})**")
    
    # ════════════════════════════════════════════════════════════════
    # БЛОК РЕДАКТИРОВАНИЯ
    # ════════════════════════════════════════════════════════════════
    
    st.markdown("---")
    st.subheader("✏️ Исправить классификацию")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        new_category = st.selectbox(
            "Изменить категорию на:",
            options=['Без изменений', 'Классификация 1', 'Классификация 2'],
            key='category_change'
        )
    
    with col2:
        st.write("")  # Отступ
        st.write("")  # Отступ
        if st.button("💾 Сохранить изменение", type="primary"):
            if new_category != 'Без изменений':
                # Обновляем данные
                df.at[vacancy_idx, 'category'] = new_category
                df.at[vacancy_idx, 'is_linguistic'] = (new_category == 'Классификация 1')
                df.at[vacancy_idx, 'confidence'] = 1.0
                df.at[vacancy_idx, 'classification_method'] = 'manual_review'
                
                # Сохраняем
                save_changes(df, current_file)
                
                st.success(f"✅ Категория изменена на '{new_category}'!")
                st.info("🔄 Обновите страницу (F5) чтобы увидеть изменения")
            else:
                st.warning("⚠️ Выберите новую категорию")
    
    with col3:
        st.write("")  # Отступ
        st.write("")  # Отступ
        if st.button("🔄 Сбросить"):
            st.rerun()

st.markdown("---")

# ════════════════════════════════════════════════════════════════
# МАССОВОЕ РЕДАКТИРОВАНИЕ (ДОПОЛНИТЕЛЬНО)
# ════════════════════════════════════════════════════════════════

with st.expander("🔧 Массовое редактирование (для продвинутых)", expanded=False):
    st.markdown("**Изменить категорию для нескольких вакансий по ключевым словам**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        keyword = st.text_input("Ключевое слово в названии:", placeholder="например: 1С, редактор, NLP")
        new_mass_category = st.selectbox(
            "Новая категория:",
            options=['Классификация 1', 'Классификация 2'],
            key='mass_category'
        )
    
    with col2:
        if keyword:
            matching = df[df['title'].str.contains(keyword, case=False, na=False)]
            st.info(f"Найдено вакансий: {len(matching)}")
            
            if len(matching) > 0:
                st.dataframe(matching[['title', 'category']], use_container_width=True, hide_index=True)
                
                if st.button("✅ Применить ко всем найденным", type="primary", key='mass_apply'):
                    for idx in matching.index:
                        df.at[idx, 'category'] = new_mass_category
                        df.at[idx, 'is_linguistic'] = (new_mass_category == 'Классификация 1')
                        df.at[idx, 'confidence'] = 1.0
                        df.at[idx, 'classification_method'] = 'manual_review'
                    
                    save_changes(df, current_file)
                    st.success(f"✅ Обновлено {len(matching)} вакансий!")
                    st.info("🔄 Обновите страницу (F5)")

st.markdown("---")

# ════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════

st.markdown("""
---
**Использованные технологии:**
- `Sentence Transformers` - векторизация текстов (paraphrase-multilingual-MiniLM-L12-v2)
- `UMAP` - снижение размерности для визуализации (384D → 2D)
- `K-Means` - кластеризация вакансий
- `Zero-shot classification` - классификация без обучения (mDeBERTa-v3-base-xnli)
- `Streamlit` - интерактивный дашборд
- `Plotly` - интерактивная визуализация
""")