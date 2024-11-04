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

# Функция для извлечения текста из PDF, doc, txt
def extract_text(file, file_extension):
    doc = fitz.open(stream=file.read(), filetype=file_extension)  # Открываем PDF-файл
    text = ""
    for page in doc:
        text += page.get_text()  # Извлекаем текст из каждой страницы
    return text

# Функция для вызова GigaChat API
def call_gigachat_api(prompt, text_content):
    messages = [SystemMessage(content=prompt+' Выдавайте ответы всегда на русском и будьте лаконичны в ответе.'), HumanMessage(content=text_content)]
    res = chat(messages)
    return res.content





from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter





# Работа самого приложения
if uploaded_file is not None:
    file_name = uploaded_file.name
    file_extension = file_name.split('.')[-1] if '.' in file_name else None
    text_content = ''

    try:
        text_content = extract_text(uploaded_file, file_extension)
    except Exception as e:
        st.write('Похоже, произошла какая-то ошибка, попробуйте загрузить файл ещё раз')

    if text_content:
        # Кнопки функционала
        functions = [
            "Сформировать литературный обзор",
            "Задать вопрос по содержанию текста",
            "Объяснение терминов",
            "Адаптивная суммаризация"
        ]
        selected_function = st.selectbox("Выберите функцию", functions)

        if selected_function == functions[0]:  # Литературный обзор
            if st.button("Сформировать литературный обзор", key="review_button"):
                prompt = "Сформируйте литературный обзор на основе следующего текста. Используйте не более 200 слов."
                review = call_gigachat_api(prompt, text_content)
                st.write(review)

        elif selected_function == functions[1]:  # Вопрос по содержанию текста
            if 'question_history' not in st.session_state:
                st.session_state.question_history = []

            question = st.text_input("Введите ваш вопрос:", key="question_input")
            if st.button("Задать вопрос", key="ask_question_button"):
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
                split_docs = text_splitter.split_documents(text_content)
                model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
                encode_kwargs = {'normalize_embeddings': False}
                embedding = HuggingFaceEmbeddings(model_name=model_name, encode_kwargs=encode_kwargs)
                vector_store = FAISS.from_documents(split_docs, embedding=embedding)
                embedding_retriever = vector_store.as_retriever(search_kwargs={"k": 3})
                prompt = ChatPromptTemplate.from_template(('''Ответь на вопрос пользователя. \
                Используй при этом только информацию из контекста. Если в контексте нет \
                информации для ответа, сообщи об этом пользователю.
                Контекст: {context}
                Вопрос: {input}
                Ответ:''')
                document_chain = create_stuff_documents_chain(llm=chat, prompt=prompt)
                retrieval_chain = create_retrieval_chain(embedding_retriever, document_chain)
                retrieval_chain.invoke(
                    {'input': q1}
                )
                st.session_state.question_history.append((question, answer))

            for q, a in st.session_state.question_history:
                st.write(f"**Вопрос:** {q}")
                st.write(f"**Ответ:** {a}")

        elif selected_function == functions[2]:  # Объяснение терминов
            if 'term_history' not in st.session_state:
                st.session_state.term_history = []

            term = st.text_input("Введите термин:", key="term_input")
            if st.button("Объяснить термин", key="explain_term_button"):
                definition = call_gigachat_api(f"Дай подробное определение термину: {term}", text_content)
                st.session_state.term_history.append((term, definition))

            for t, d in st.session_state.term_history:
                st.write(f"**Термин:** {t}")
                st.write(f"**Определение:** {d}")

        elif selected_function == functions[3]:  # Адаптивная суммаризация
            if 'summary_history' not in st.session_state:
                st.session_state.summary_history = []

            summary_percentage = st.slider("Выберите процент суммирования:", 10, 90, 50, key="summary_slider")
            if st.button("Суммировать", key="summarize_button"):
                summary_prompt = f"Суммируй текст на {summary_percentage}%. 90% - суммируй до пары предложений, 10% - суммируй до пары абзацев."
                summary = call_gigachat_api(summary_prompt, text_content)
                st.session_state.summary_history.append((summary_percentage, summary))

            for percent, sum_text in st.session_state.summary_history:
                st.write(f"**Процент:** {percent}%")
                st.write(f"**Суммированный текст:** {sum_text}")
