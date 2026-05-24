import os
import sqlite3
import pandas as pd
import streamlit as st
from google import genai
from google.genai import errors

# 1. НАСТРОЙКА БАЗЫ ДАННЫХ
DB_NAME = "feedback_v1.db"

def init_db():
    """Создает таблицу, если её еще нет."""
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                review_text TEXT,
                rating TEXT,  -- Изменили на TEXT, чтобы можно было записать "-"
                sentiment TEXT DEFAULT 'Не определено',
                issue_category TEXT DEFAULT 'Не определено'
            )
        """)
        conn.commit()

def save_to_db(text, rating, sentiment):
    """Сохраняет отзыв и результаты анализа в базу данных."""
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO reviews (review_text, rating, sentiment) VALUES (?, ?, ?)",
            (text, rating, sentiment)
        )
        conn.commit()

def load_data():
    """Выгружает все данные в Pandas DataFrame."""
    with sqlite3.connect(DB_NAME) as conn:
        df = pd.read_sql_query("SELECT * FROM reviews", conn)
    return df

# 2. ИНТЕГРАЦИЯ GEMINI AI
def analyze_review_with_gemini(review_text):
    """
    Отправляет текст в Gemini. 
    Возвращает кортеж: (оценка_строкой, тональность)
    """
    # Ищем ключ в переменных окружения (Streamlit подтянет из .env или secrets)
    api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        # Если ключа нет, сразу возвращаем дефолтные значения
        return "-", "Ошибка: Нет API ключа"
        
    try:
        # Инициализируем официальный клиент google-genai
        client = genai.Client(api_key=api_key)
        
        prompt = f"""
        Проанализируй следующий отзыв пользователя.
        Выдай ответ строго в формате JSON с двумя ключами:
        "rating": число от 1 до 5 (где 1 - ужасно, 5 - отлично), основанное на тоне отзыва.
        "sentiment": одно слово на русском языке ('Позитивный', 'Нейтральный' или 'Негативный').
        
        Отзыв: "{review_text}"
        
        Ответ должен содержать только чистый JSON, без разметки markdown (без ```json).
        """
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        
        # Парсим ответ от ИИ
        import json
        result = json.loads(response.text.strip())
        
        # Приводим оценку к строке для единообразия с дефолтным "-"
        return str(result.get("rating", "-")), result.get("sentiment", "Не определено")
        
    except Exception as e:
        # Если упала сеть, неверный ключ или ИИ вернул некорректный JSON
        st.sidebar.error(f"Сбой ИИ: {e}") # Незаметно выведем ошибку в боковую панель
        return "-", "Сбой связи с ИИ"

# Инициализируем БД при запуске приложения
init_db()


# 3. ИНТЕРФЕЙС STREAMLIT
st.set_page_config(page_title="Мониторинг отзывов", layout="wide")
st.title("📊 Демо-проект: Сбор и ИИ-анализ отзывов")

# Разделяем интерфейс на две колонки: Форма и Дашборд
col_form, col_dash = st.columns([1, 2], gap="large")

# --- КОЛОНКА 1: ФОРМА ОПРОСНИКА ---
with col_form:
    st.header("📝 Оставить отзыв")
    
    with st.form("feedback_form", clear_on_submit=True):
        review_input = st.text_area(
            "Ваш отзыв", 
            placeholder="Напишите, что вы думаете о товаре или услуге..."
        )
        # Оценка из формы теперь резервная (на случай, если ИИ вернет "-")
        backup_rating = st.selectbox(
            "Ваша оценка (резервная, если ИИ недоступен)", 
            options=["-", 1, 2, 3, 4, 5], 
            index=0
        )
        
        submit_button = st.form_submit_button("Отправить на ИИ-анализ")
        
        if submit_button:
            if review_input.strip() == "":
                st.warning("Пожалуйста, напишите текст отзыва.")
            else:
                with st.spinner("Gemini анализирует отзыв..."):
                    # Вызываем ИИ
                    ai_rating, ai_sentiment = analyze_review_with_gemini(review_input)
                    
                    # Если ИИ сломался и вернул "-", берем оценку пользователя из формы
                    final_rating = ai_rating if ai_rating != "-" else str(backup_rating)
                    
                    # Сохраняем в базу данных
                    save_to_db(review_input, final_rating, ai_sentiment)
                    
                st.success(f"Анализ завершен! Оценка ИИ: {ai_rating}, Тональность: {ai_sentiment}")
                st.rerun()


# --- КОЛОНКА 2: АНАЛИТИКА И ДАШБОРД ---
with col_dash:
    st.header("📈 Аналитическая панель")
    
    df_data = load_data()
    
    if df_data.empty:
        st.info("В базе данных пока нет отзывов. Заполните форму слева, чтобы ИИ построил графики.")
    else:
        # Метрики
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric(label="Всего отзывов", value=len(df_data))
        with col_m2:
            # Считаем сколько раз ИИ не смог определить оценку
            errors_count = len(df_data[df_data['rating'] == '-'])
            st.metric(label="Сбоев ИИ (оценки '-')", value=errors_count)
        
        # График 1: Распределение оценок
        st.subheader("Распределение оценок (включая '-')")
        
        # Чтобы график не падал из-за смешанных типов, приводим всё к строке
        df_data['rating'] = df_data['rating'].astype(str)
        rating_counts = df_data['rating'].value_counts().reset_index()
        rating_counts.columns = ['Оценка', 'Количество']
        
        st.bar_chart(data=rating_counts, x='Оценка', y='Количество')
        
        # Вывод таблицы с новыми колонками
        st.subheader("Последние записи и вердикт Gemini")
        st.dataframe(
            df_data[['id', 'rating', 'sentiment', 'review_text']].sort_values(by='id', ascending=False),
            use_container_width=True,
            hide_index=True
        )