# Импортирование необходимых библиотек
import streamlit as st
import fitz  # PyMuPDF
import tempfile
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.chat_models.gigachat import GigaChat

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.schema import Document
from langchain.schema import AIMessage, HumanMessage

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

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
    messages = [SystemMessage(content=prompt), HumanMessage(content=text_content)]
    res = chat(messages)
    return res.content

# Иницализация эмбеддера
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

# Работа самого приложения
if uploaded_file is not None:
    file_name = uploaded_file.name
    file_extension = file_name.split('.')[-1] if '.' in file_name else None
    text_content = ''

    try:
        text_content = extract_text(uploaded_file, file_extension)
    except Exception as e:
        st.write('Похоже, произошла какая-то ошибка, попробуйте загрузить файл ещё раз')

    # RAG
    # Создании функции по разделению текста на части
    class CustomTextSplitter(SentenceTransformersTokenTextSplitter):

        def split_text(self, text: str) -> list:
            chunks = super().split_text(text)
            chunks_with_prefix = ['search_document: ' + chunk for chunk in chunks]
            return chunks_with_prefix

    # Преоброзование текста в удобный для модели формат
    docs = [Document(page_content=text_content)]

    # Разделение текста на части
    text_splitter = CustomTextSplitter(chunk_size=506, chunk_overlap=50, model_name="ai-forever/ru-en-RoSBERTa")
    split_docs = text_splitter.split_documents(docs)

    # Создание векторной базы данных FAISS
    vector_store = FAISS.from_documents(split_docs, embedding=embeddings)

    # Инициализация ретривера
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    # Создание шаблонов
    contextualize_q_system_prompt = (
        "Учитывай историю чата и последний вопрос пользователя, "
        "который может ссылаться на контекст в истории чата, "
        "сформулируй отдельный вопрос, который может быть понят без истории чата."
        "НЕ отвечай на вопрос, просто переформулируй его при необходимости, "
        "в ином случае верни его как есть."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        chat, retriever, contextualize_q_prompt
    )

    # Создание вопросно-ответной системы
    qa_system_prompt = (
        "Вы являетесь помощником при выполнении заданий, связанных с ответами на вопросы."
        "Используйте следующие фрагменты найденного контекста, чтобы ответить на вопрос."
        "Если вы не знаете ответа, просто скажите, что не знаете."
        "Используйте не более трех предложений и будьте лаконичны в ответе."
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(chat, qa_prompt)

    # Создание RAG для ответов на вопросы
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Создание системы по нахождению определний терминов
    termins_system_prompt = (
        "Вы являетесь помощником при выполнении заданий, связанных с выдачей определений терминов."
        "Используйте следующие фрагменты найденного контекста, чтобы ответить на вопрос."
        "Если вы не знаете ответа, просто скажите, что не знаете."
        "Используйте не более трех предложений и будьте лаконичны в ответе."
        "\n\n"
        "{context}"
    )

    termins_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", termins_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    termins_answer_chain = create_stuff_documents_chain(chat, termins_prompt)

    # Создание RAG для написания определений терминов
    rag_chain_termins = create_retrieval_chain(history_aware_retriever, termins_answer_chain)

    # Инициализация функционала ассистента
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
                prompt = "Вы являетесь помощником для резюмирования текста. Выдавайте ответы всегда на русском, независимо от языка самой статьи. Используйте не более 200 слов и будьте лаконичны в ответе."
                review = call_gigachat_api(prompt, text_content)
                st.write(review)

        elif selected_function == functions[1]:  # Вопрос по содержанию текста
            if 'question_history' not in st.session_state:
                st.session_state.question_history = []

            chat_history = []
            question = st.text_input("Введите ваш вопрос:", key="question_input")

            if st.button("Задать вопрос", key="ask_question_button"):
                result = rag_chain.invoke({"input": question, "chat_history": chat_history})
                answer = result['answer']

                chat_history.append(HumanMessage(content=question))
                chat_history.append(AIMessage(content=answer))

                st.session_state.question_history.append((question, answer))

            for q, a in st.session_state.question_history:
                st.write(f"**Вопрос:** {q}")
                st.write(f"**Ответ:** {a}")

        elif selected_function == functions[2]:  # Объяснение терминов
            if 'term_history' not in st.session_state:
                st.session_state.term_history = []

            chat_history_termins = []
            term = st.text_input("Введите термин:", key="term_input")
            if st.button("Объяснить термин", key="explain_term_button"):
                res = rag_chain_termins.invoke({"input": term, "chat_history": chat_history_termins})
                definition = res['answer']

                chat_history_termins.append(HumanMessage(content=term))
                chat_history_termins.append(AIMessage(content=definition))

                st.session_state.term_history.append((term, definition))

            for t, d in st.session_state.term_history:
                st.write(f"**Термин:** {t}")
                st.write(f"**Определение:** {d}")

        elif selected_function == functions[3]:  # Адаптивная суммаризация
            if 'summary_history' not in st.session_state:
                st.session_state.summary_history = []

            summary_percentage = st.slider("Выберите процент суммирования:", 10, 90, 50, key="summary_slider")
            if st.button("Суммировать", key="summarize_button"):
                summary_prompt = f"Вы являетесь помощником для адаптивной суммаризации текста. Выдавайте ответы всегда на русском, независимо от языка самой статьи. Оставь от исходного статьи {summary_percentage}% текста, содержащего самую главную информацию."
                summary = call_gigachat_api(summary_prompt, text_content)
                st.session_state.summary_history.append((summary_percentage, summary))

            for percent, sum_text in st.session_state.summary_history:
                st.write(f"**Процент:** {percent}%")
                st.write(f"**Суммированный текст:** {sum_text}")
