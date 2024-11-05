# Scientific_literature_assistant (Ассистент для работы с научной литературой)
# О репозитории
Репозиторий содержит файл app.py, содержащий код приложения, README.md и requirements.txt (для установки всех требуемых библиотек)
# О приложение
Это Web-приложение позволяет пользователям загружать файлы различного формата(pdf, doc, txt) и работать с ними благодаря различному функционалу. Цель проекта — упростить процесс работы с научной литературой.
# Используемые технологии
- [Документация streamlit](https://docs.streamlit.io/)
- [Документация PyMuPDF](https://pymupdf.readthedocs.io/en/latest/)
- [Документация GigaChat API](https://developers.sber.ru/docs/ru/gigachat/api/overview)
- [Документация GigaChain](https://developers.sber.ru/docs/ru/gigachain/overview)
- [Документация LangChain](https://python.langchain.com/docs/introduction/)
- [Документация HuggingFace](https://huggingface.co/docs)
- [Документация FAISS](https://faiss.ai/)

# Начало работы
Ниже приведена инструкция по началу работы. Чтобы получить локальную копию и запустить её, выполните следующие простые шаги.
## Установка
**1. Клонируйте репозиторий:**
   ```bash
   git clone https://github.com/Aptukov/Scientific_literature_assistant
   ```
**2. Установите необходимые пакеты:**
   ```bash
   pip install -r requirements.txt
   ```
## Использование (В качестве примера используется статья на английском)
### 1. Запустите приложение через терминал:
   ```bash
   streamlit run app.py
   ```
   Перед вами появится окно приложения.
   ![Интерфейс приложения](images/p_start1.png)
   
### 2. Введите свой ключ авторизации GigaChat API.
   ![Ключ авторизации](images/p_start2.png)
   
### 3. Загрузите файл статьи через интерфейс.
   ![Загрузка файла](images/p_start3.png)
   
### 4. Выбери одну из предложенных функций, чтобы начать работу со статьёй.
   ![Функционал](images/p_start4.png)
   
   #### 1) Формирование литературных обзоров
   Нажмите на кнопку "Сформировать литературный обзор", после чего выведется литературный обзор статьи.
   
   ![Функция 1](images/p_fun11.png)
   #### 2) Ответ на вопрос по содержанию статьи
   Напишите свой вопрос в соотвутствующее поле и нажмите кнопку "Задать вопрос", после чего ассистент выдаст ответ, исходя из содержания статьи.
   
   ![Функция 21](images/p_fun21.png)
   ![Функция 22](images/p_fun22.png)
      
   #### 3) Определение термина
   Напишите термин, определение которого хотите узнать, затем нажмите кнопку "Объяснить термин" и ассистент выдаст определение на основе текста статьи.
   
   ![Функция 31](images/p_fun31.png)
   ![Функция 32](images/p_fun322.png)
      
   #### 4) Адаптивная суммаризация
   Выберите процент суммаризации, затем нажмите кнопку "Суммировать" и вам выдастся сжатая на n(от 10 до 90) процентов.
   
   ![Функция 41](images/p_fun41.png)
   ![Функция 42](images/p_fun42.png)

### 5. Сохранение истории взаимодействия.
При переходе с одной функции на другую, история взаимодействия сохраняется. Если снова перейти к той же функции, история чата выводится и сразу же сохраняется
      
### Участие
Вклады — это то, что делает сообщество с открытым исходным кодом таким замечательным местом для обучения, вдохновения и творчества. Любой ваш вклад будет высоко оценен .

Если у вас есть предложение, которое сделает это лучше, пожалуйста, разветвите репозиторий и создайте запрос на извлечение. Вы также можете просто открыть проблему с тегом "улучшение". Не забудьте поставить проекту звезду! Спасибо еще раз!
## Контакты

Аптуков Вадим - [Аккаунт в телеграме](@HackNet11) - vraptukov@gmail.com

Ссылка на проект: [https://github.com/Aptukov/Scientific_literature_assistant](https://github.com/Aptukov/Scientific_literature_assistant)
