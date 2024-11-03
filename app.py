# pip install streamlit gigachain-community PyMuPDF
# streamlit run app.py

# Импортирование необходимых библиотек
import streamlit as st
import fitz  # PyMuPDF
import tempfile
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.chat_models.gigachat import GigaChat

# Заголовок приложения
st.title("Ваш персональный ассистент для работы с научной литературой")

# Получение ключа авторизации пользователя
Authorization_key = st.text_area('Для начала работы, введите ваш ключ авторизации Gigachat API')

# Авторизация в сервисе GigaChat
chat = GigaChat(credentials=Authorization_key, scope='GIGACHAT_API_PERS', model='GigaChat', streaming=True, verify_ssl_certs=False)

# Загрузка файла
uploaded_file = st.file_uploader("Загрузите свой файл", type=["pdf",'txt','doc'])


# Функция для извлечения текста из PDF
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")  # Открываем PDF-файл
    text = ""
    for page in doc:
        text += page.get_text()  # Извлекаем текст из каждой страницы
    return text

# Функция для извлечения текста из doc
def extract_text_from_doc(file):
    doc = fitz.open(stream=file.read(), filetype="doc")  # Открываем doc-файл
    text = ""
    for page in doc:
        text += page.get_text()  # Извлекаем текст из каждой страницы
    return text

# Функция для извлечения текста из txt
def extract_text_from_txt(file):
    doc = fitz.open(stream=file.read(), filetype="txt")  # Открываем txt-файл
    text = ""
    for page in doc:
        text += page.get_text()  # Извлекаем текст из каждой страницы
    return text

if uploaded_file is not None:
    # Извлекаем расширение
    file_name = uploaded_file.name
    file_extension = file_name.split('.')[-1] if '.' in file_name else None

    if file_extension == 'pdf':
        # Извлечение текста из PDF
        full_text = extract_text_from_pdf(uploaded_file)
    elif file_extension == 'doc':
        # Извлечение текста из doc
        full_text = extract_text_from_doc(uploaded_file)
    elif file_extension == 'txt':
        # Извлечение текста из txt
        full_text = extract_text_from_txt(uploaded_file)
    else:
        st.write('Похоже, произошла какая-то ошибка, попробуйте загрузить файл ещё раз')

    if st.button("Задать вопрос по содержанию статьи"):
        question = st.text_area('Введи свой вопрос по статье: >>>')
        st.write(question)
    elif st.button('Сформировать литературный обзор статьи'):
        try:
            # Проверяем длину текста
            if full_text:
                messages = [
                    SystemMessage(
                        content="Вы являетесь помощником для резюмирования текста. Выдавайте ответы всегда на русском, независимо от языка самой статьи. Используйте не болeе 200 слов и будьте лаконичны в ответе."),
                    HumanMessage(content=full_text)
                ]

                # Получаем краткое содержание
                res = chat(messages)

                # Вывод ответа
                st.subheader("Литературный обзор:")
                st.write(res.content)
            else:
                st.write("Не удалось извлечь текст.")
        except Exception as e:
            st.subheader(
                'Похоже, произошла какая-то ошибка, проверьте правильность ввода ключа авторизации и поддерживаемость формата статьи (pdf, doc, txt)')
    else:
        st.write('Для начала работы выбери одну из функций и нажми на соответственную кнопку')