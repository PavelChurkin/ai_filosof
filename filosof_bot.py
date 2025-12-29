"""
AI Filosof - Telegram Bot for sharing philosophical thoughts
Transformed from prophecy bot to thought-sharing bot with database and payments
"""
import random
import json
import asyncio
from openai import OpenAI
import os
from dotenv import load_dotenv
import logging
from typing import List, Tuple, Optional, Dict
from datetime import datetime, time as dt_time, timedelta
import pytz
from dataclasses import dataclass
import time

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler,
    ContextTypes, MessageHandler, filters
)
from telegram.constants import ParseMode

from database import Database, ChatState, Thought

# Настройка логирования
class MoscowTimeFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        moscow_tz = pytz.timezone('Europe/Moscow')
        dt = datetime.fromtimestamp(record.created, moscow_tz)
        if datefmt:
            return dt.strftime(datefmt)
        else:
            return dt.isoformat()


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

for handler in logging.root.handlers:
    handler.setFormatter(MoscowTimeFormatter())

# Константы
MOSCOW_TZ = pytz.timezone('Europe/Moscow')

# Цены в рублях
PRICE_DONATION_MIN = 50  # Минимальная сумма пожертвования

# Лимиты запросов
DAILY_REQUEST_LIMIT = 3  # Максимум запросов в день для обычных пользователей
DAILY_REQUEST_LIMIT_DONOR = 6  # Максимум запросов в день для пожертвовавших

# Глобальные переменные
db: Optional[Database] = None
stop_flag = False


def load_env_keys() -> Dict[str, Optional[str]]:
    """Загружает ключи из .env файла"""
    load_dotenv(override=True)
    return {
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        'TG_TOKEN': os.getenv('TG_TOKEN'),
        'ROBOKASSA_MERCHANT_LOGIN': os.getenv('ROBOKASSA_MERCHANT_LOGIN'),
        'ROBOKASSA_PASSWORD1': os.getenv('ROBOKASSA_PASSWORD1'),
        'ROBOKASSA_PASSWORD2': os.getenv('ROBOKASSA_PASSWORD2'),
    }


def get_moscow_time() -> datetime:
    """Возвращает текущее время в московском часовом поясе"""
    return datetime.now(MOSCOW_TZ)


def format_moscow_time(dt: datetime = None, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Форматирует время в московском поясе"""
    if dt is None:
        dt = get_moscow_time()
    return dt.strftime(format_str)


def optimized_choice_lst(lst: list, max_iterations: int = 20000) -> Tuple[list, list]:
    """Оптимизированная версия choice_lst"""
    if not lst:
        return [], []

    unique_elements = set(lst)
    lst_choice = []
    found_elements = set()

    for i in range(max_iterations):
        if len(found_elements) == len(unique_elements):
            break
        choice = random.choice(lst)
        lst_choice.append(choice)
        found_elements.add(choice)

    missing_elements = list(unique_elements - found_elements)
    return lst_choice, random.sample(missing_elements, min(2, len(missing_elements)))


def create_dct(sampled_lst: list) -> List[Tuple[str, int]]:
    """Создает список топ-3 самых частых слов"""
    frequency_dict = {}
    for word in sampled_lst:
        frequency_dict[word] = frequency_dict.get(word, 0) + 1

    sorted_items = sorted(frequency_dict.items(), key=lambda x: x[1], reverse=True)
    return sorted_items[:3]


def get_openai_response(prompt: str, max_retries: int = 3) -> str:
    """Получает ответ от OpenAI API"""
    keys = load_env_keys()
    openai_api_key = keys['OPENAI_API_KEY']

    if not openai_api_key:
        logger.error("OPENAI_API_KEY не найден")
        return "Моя магия слов закончилась. API ключ отсутствует."

    openai_client = OpenAI(
        api_key=openai_api_key,
        base_url="https://api.proxyapi.ru/openai/v1",
        timeout=30
    )

    system_message = f"Ты размышляющий бот" # бот, с изменяющейся ролью (смотри контекст для определения роли)

    for attempt in range(max_retries):
        try:
            logger.info(f"Попытка {attempt + 1} получить ответ от OpenAI...")

            chat_completion = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                timeout=30
            )

            response = chat_completion.choices[0].message.content
            logger.info("Успешно получен ответ от OpenAI")
            return response

        except Exception as e:
            logger.warning(f"Попытка {attempt + 1} не удалась: {e}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5
                logger.info(f"Ожидание {wait_time} секунд...")
                time.sleep(wait_time)
            else:
                logger.error("Все попытки провалились")
                return "Моя магия слов закончилась по техническим причинам."


class ThoughtGenerator:
    """Генератор философских мыслей"""

    def __init__(self):
        # Загружаем словари
        try:
            with open("nouns.json", "r", encoding='utf-8') as fh:
                self.nouns = json.load(fh)
            with open("verbs.json", "r", encoding='utf-8') as fh:
                self.verbs = json.load(fh)
            with open("adject.json", "r", encoding='utf-8') as fh:
                self.adjectives = json.load(fh)

            logger.info(
                f"Загружено: существительных - {len(self.nouns)}, "
                f"глаголов - {len(self.verbs)}, прилагательных - {len(self.adjectives)}"
            )
        except Exception as e:
            logger.error(f"Ошибка загрузки словарей: {e}")
            raise

    async def generate_thought_3_steps(self, chat_id: str, was_paid: bool = False) -> Thought:
        """
        Генерация мысли в 3 этапа:
        1. Выбор слов → образ и роль
        2. Образ → вопрос
        3. Вопрос → ответ
        """
        global db
        try:
            # Шаг 0: Генерация случайных слов
            sample_size = random.randint(100, 20000)

            noun_samples = [random.choice(self.nouns) for _ in range(sample_size)]
            verb_samples = [random.choice(self.verbs) for _ in range(sample_size)]
            adjective_samples = [random.choice(self.adjectives) for _ in range(sample_size)]

            choice_nouns, rare_nouns = optimized_choice_lst(noun_samples)
            choice_verbs, rare_verbs = optimized_choice_lst(verb_samples)
            choice_adjectives, rare_adjectives = optimized_choice_lst(adjective_samples)

            top_nouns = create_dct(choice_nouns)
            top_verbs = create_dct(choice_verbs)
            top_adjectives = create_dct(choice_adjectives)

            # Формирование списка слов
            words_list = f"Существительные: {top_nouns} / {rare_nouns}\n" \
                        f"Глаголы: {top_verbs} / {rare_verbs}\n" \
                        f"Прилагательные: {top_adjectives} / {rare_adjectives}"

            logger.info(f"Сгенерированы слова для чата {chat_id}")

            # Шаг 1: Формирование образа и роли
            prompt1 = f"Даны следующие слова:\n{words_list}\n\n" \
                     f"Выбери из них только слова. По этим словам сформируй образ, не больше 100 слов, с определением своей роли"

            loop = asyncio.get_event_loop()
            step1_image = await loop.run_in_executor(None, get_openai_response, prompt1)
            logger.info(f"Шаг 1 завершен для чата {chat_id}")

            # Шаг 2: Формирование вопроса
            prompt2 = f"{step1_image}\n\nСформируй вопрос"
            step2_question = await loop.run_in_executor(None, get_openai_response, prompt2)
            logger.info(f"Шаг 2 завершен для чата {chat_id}")

            # Шаг 3: Ответ на вопрос
            prompt3 = f"{step2_question}\n\nОтветь на вопрос, не больше 100 слов"
            step3_answer = await loop.run_in_executor(None, get_openai_response, prompt3)
            logger.info(f"Шаг 3 завершен для чата {chat_id}")

            # Сохраняем в базу данных
            thought = await db.save_thought(
                chat_id=chat_id,
                step1_words=words_list,
                step1_image=step1_image,
                step2_question=step2_question,
                step3_answer=step3_answer,
                is_published=False,
                was_paid=was_paid
            )

            logger.info(f"Мысль сохранена в БД с ID {thought.id}")
            return thought

        except Exception as e:
            logger.error(f"Ошибка генерации мысли: {e}")
            raise


# Telegram Bot Handlers

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    chat_id = str(update.effective_chat.id)

    # Определяем тип чата
    if update.effective_chat.type == 'channel':
        chat_type = 'channel'
    elif update.effective_chat.type in ['group', 'supergroup']:
        chat_type = 'group'
    else:
        chat_type = 'private'

    # Создаем или получаем состояние чата
    chat_state = await db.get_or_create_chat_state(chat_id, chat_type)

    # Получаем информацию о балансе
    balance = chat_state.bonus_requests if chat_state.bonus_requests is not None else DAILY_REQUEST_LIMIT

    # Формируем приветственное сообщение (одинаковое для всех типов чатов)
    welcome_text = f"""
👋 Добро пожаловать в AI Filosof!

💎 Ваш баланс запросов: {balance}

Вы можете:
⚡ Получить срочную мысль прямо сейчас
🎲 Создать мысль на основе своих случайных слов
❓ Задать свой вопрос и получить философский ответ
💝 Поддержать проект пожертвованием

⚡ Лимит: минимум {DAILY_REQUEST_LIMIT} запроса в день
💎 Пожертвование: 50₽ = +3 запроса к балансу
💡 Каждый день восстанавливается до {DAILY_REQUEST_LIMIT} запросов (если меньше)
"""

    # Добавляем inline кнопки (одинаковые для всех типов чатов)
    keyboard = [
        [InlineKeyboardButton("⚡ Срочная мысль", callback_data="urgent_thought")],
        [InlineKeyboardButton("🎲 Свои случайные слова", callback_data="custom_words")],
        [InlineKeyboardButton("❓ Ваш вопрос", callback_data="your_question")],
        [InlineKeyboardButton("💝 Пожертвование", callback_data="pay_donation")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(welcome_text, reply_markup=reply_markup)


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик нажатий на inline кнопки"""
    query = update.callback_query
    await query.answer()

    chat_id = str(update.effective_chat.id)
    user_id = str(update.effective_user.id)
    callback_data = query.data

    if callback_data == "pay_donation":
        # Пожертвование
        await handle_donation_payment(query, chat_id, user_id)

    elif callback_data == "donate_custom":
        # Удаляем предыдущее сообщение
        try:
            await query.message.delete()
        except Exception as e:
            logger.warning(f"Не удалось удалить сообщение: {e}")
        # Пользователь хочет ввести свою сумму
        context.user_data['awaiting_input'] = 'donation_amount'
        bot = query.get_bot()
        await bot.send_message(
            chat_id=chat_id,
            text=f"💬 Пожалуйста, введите сумму пожертвования (минимум {PRICE_DONATION_MIN}₽):\n\n"
                 f"ℹ️ Каждые {PRICE_DONATION_MIN}₽ дают +3 запроса к вашему балансу"
        )

    elif callback_data.startswith("donate_"):
        # Пожертвование с конкретной суммой
        amount = int(callback_data.split("_")[1])
        await process_donation(query, chat_id, user_id, amount)

    elif callback_data == "urgent_thought":
        # Удаляем предыдущее сообщение
        try:
            await query.message.delete()
        except Exception as e:
            logger.warning(f"Не удалось удалить сообщение: {e}")

        # Срочная мысль - генерация мысли по стандартному алгоритму (тратит лимит)
        # Проверка лимита
        can_proceed, remaining = await db.check_and_update_daily_limit(chat_id)
        bot = query.get_bot()
        if not can_proceed:
            keyboard = [
                [InlineKeyboardButton("💝 Пожертвование", callback_data="pay_donation")],
                [InlineKeyboardButton("⬅️ Назад", callback_data="back_to_menu")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await bot.send_message(
                chat_id=chat_id,
                text="❌ Вы исчерпали дневной лимит запросов.\n\n"
                     "💝 Сделайте пожертвование, чтобы получить дополнительные запросы!",
                reply_markup=reply_markup
            )
            return

        loading_msg = await bot.send_message(
            chat_id=chat_id,
            text=f"⏳ Генерирую срочную философскую мысль...\n\n"
                 f"⚡ Осталось запросов сегодня: {remaining}"
        )

        # Генерируем мысль
        generator = ThoughtGenerator()
        thought = await generator.generate_thought_3_steps(chat_id, was_paid=False)

        # Удаляем loading сообщение
        try:
            await loading_msg.delete()
        except Exception as e:
            logger.warning(f"Не удалось удалить loading сообщение: {e}")

        # Отправляем результат с кнопками раскрытия деталей
        message = f"🧠 Философская мысль:\n\n{thought.step3_answer}"
        keyboard = [
            [InlineKeyboardButton("🔍 Раскрыть промпт", callback_data=f"reveal_prompt_{thought.id}")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await bot.send_message(chat_id=chat_id, text=message, reply_markup=reply_markup)

    elif callback_data == "custom_words":
        # Удаляем предыдущее сообщение
        try:
            await query.message.delete()
        except Exception as e:
            logger.warning(f"Не удалось удалить сообщение: {e}")

        # Запрос пользовательских случайных слов
        # Проверка лимита
        can_proceed, remaining = await db.check_and_update_daily_limit(chat_id)
        bot = query.get_bot()
        if not can_proceed:
            keyboard = [
                [InlineKeyboardButton("💝 Пожертвование", callback_data="pay_donation")],
                [InlineKeyboardButton("⬅️ Назад", callback_data="back_to_menu")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await bot.send_message(
                chat_id=chat_id,
                text="❌ Вы исчерпали дневной лимит запросов.\n\n"
                     "💝 Сделайте пожертвование, чтобы получить дополнительные запросы!",
                reply_markup=reply_markup
            )
            return

        context.user_data['awaiting_input'] = 'custom_words'
        await bot.send_message(
            chat_id=chat_id,
            text=f"🎲 Введите свои случайные слова (через запятую или пробел):\n\n"
                 f"Например: дерево, океан, мечта, время\n\n"
                 f"⚡ Осталось запросов сегодня: {remaining}"
        )

    elif callback_data == "your_question":
        # Удаляем предыдущее сообщение
        try:
            await query.message.delete()
        except Exception as e:
            logger.warning(f"Не удалось удалить сообщение: {e}")

        # Запрос вопроса от пользователя
        # Проверка лимита
        can_proceed, remaining = await db.check_and_update_daily_limit(chat_id)
        bot = query.get_bot()
        if not can_proceed:
            keyboard = [
                [InlineKeyboardButton("💝 Пожертвование", callback_data="pay_donation")],
                [InlineKeyboardButton("⬅️ Назад", callback_data="back_to_menu")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await bot.send_message(
                chat_id=chat_id,
                text="❌ Вы исчерпали дневной лимит запросов.\n\n"
                     "💝 Сделайте пожертвование, чтобы получить дополнительные запросы!",
                reply_markup=reply_markup
            )
            return

        context.user_data['awaiting_input'] = 'your_question'
        await bot.send_message(
            chat_id=chat_id,
            text=f"❓ Введите ваш вопрос, на который вы хотите получить философский ответ:\n\n"
                 f"⚡ Осталось запросов сегодня: {remaining}"
        )

    elif callback_data == "back_to_menu":
        # Удаляем предыдущее сообщение
        try:
            await query.message.delete()
        except Exception as e:
            logger.warning(f"Не удалось удалить сообщение: {e}")

        # Вернуться в главное меню
        # Получаем информацию о балансе для отображения
        state = await db.get_or_create_chat_state(chat_id)

        welcome_text = (
            f"🤖 Привет! Я AI Filosof — бот, который генерирует философские мысли.\n\n"
            f"⚡ Ваш баланс запросов: {state.bonus_requests}\n"
            f"(Базовый лимит: {DAILY_REQUEST_LIMIT} запроса в день, восстанавливается ежедневно если баланс < {DAILY_REQUEST_LIMIT})\n\n"
            f"Что бы вы хотели?\n\n"
            f"💡 Бесплатные функции (с лимитом):\n"
            f"• Срочная мысль — генерация по стандартному алгоритму\n"
            f"• Свои случайные слова — генерация на основе ваших слов\n"
            f"• Ваш вопрос — прямой ответ на ваш вопрос\n\n"
            f"💝 Пожертвование увеличивает баланс запросов (50₽ = +3 запроса)"
        )

        # Одинаковые кнопки для всех типов чатов
        keyboard = [
            [InlineKeyboardButton("⚡ Срочная мысль", callback_data="urgent_thought")],
            [InlineKeyboardButton("🎲 Свои случайные слова", callback_data="custom_words")],
            [InlineKeyboardButton("❓ Ваш вопрос", callback_data="your_question")],
            [InlineKeyboardButton("💝 Пожертвование", callback_data="pay_donation")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        bot = query.get_bot()
        await bot.send_message(chat_id=chat_id, text=welcome_text, reply_markup=reply_markup)

    elif callback_data.startswith("reveal_question_"):
        # Раскрыть вопрос для конкретной мысли (теперь бесплатно)
        thought_id = int(callback_data.split("_")[2])
        await handle_reveal_specific_question(query, thought_id)

    elif callback_data.startswith("reveal_prompt_"):
        # Раскрыть промпт для конкретной мысли (теперь бесплатно)
        thought_id = int(callback_data.split("_")[2])
        await handle_reveal_specific_prompt(query, thought_id)


async def text_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик текстовых сообщений для кастомных запросов"""
    chat_id = str(update.effective_chat.id)
    user_id = str(update.effective_user.id)
    user_text = update.message.text

    # Проверяем, ожидается ли ввод
    if 'awaiting_input' not in context.user_data:
        return

    input_type = context.user_data.get('awaiting_input')

    if input_type == 'custom_words':
        # Обработка пользовательских слов
        await handle_custom_words_generation(update, context, user_text, chat_id)

    elif input_type == 'your_question':
        # Обработка вопроса пользователя
        await handle_question_generation(update, context, user_text, chat_id)

    elif input_type == 'donation_amount':
        # Обработка суммы пожертвования
        try:
            amount = int(user_text.strip())
            if amount < PRICE_DONATION_MIN:
                await update.message.reply_text(
                    f"❌ Минимальная сумма пожертвования: {PRICE_DONATION_MIN}₽\n\n"
                    f"Попробуйте еще раз."
                )
                return
            # Создаем платеж с пользовательской суммой
            from payments import PaymentService, create_donation_payment
            keys = load_env_keys()
            payment_service = PaymentService(
                merchant_login=keys['ROBOKASSA_MERCHANT_LOGIN'],
                password1=keys['ROBOKASSA_PASSWORD1'],
                password2=keys['ROBOKASSA_PASSWORD2'],
                db=db,
                is_test=False
            )
            payment_url = await create_donation_payment(
                payment_service, chat_id, user_id, amount
            )
            if payment_url:
                # Рассчитываем количество бонусных запросов
                bonus_requests = (amount // PRICE_DONATION_MIN) * 3
                keyboard = [[InlineKeyboardButton("💳 Перейти к оплате", url=payment_url)]]
                reply_markup = InlineKeyboardMarkup(keyboard)
                await update.message.reply_text(
                    f"💝 Спасибо за желание поддержать проект!\n\n"
                    f"Сумма: {amount}₽\n"
                    f"Вы получите: +{bonus_requests} запросов к балансу\n\n"
                    f"Нажмите кнопку ниже для оплаты:",
                    reply_markup=reply_markup
                )
            else:
                await update.message.reply_text(
                    "❌ Ошибка создания платежа. Попробуйте позже."
                )
        except ValueError:
            await update.message.reply_text(
                f"❌ Пожалуйста, введите корректную сумму (число).\n\n"
                f"Минимум: {PRICE_DONATION_MIN}₽"
            )
            return

    # Очищаем состояние ожидания
    context.user_data.pop('awaiting_input', None)


async def handle_custom_words_generation(update: Update, context: ContextTypes.DEFAULT_TYPE,
                                        user_words: str, chat_id: str):
    """Генерация мысли на основе пользовательских слов (3 этапа, сохранение в БД как срочная мысль)"""
    try:
        loading_msg = await update.message.reply_text("⏳ Генерирую философскую мысль на основе ваших слов...")

        # Парсим слова пользователя
        import re
        words_list = re.split(r'[,\s]+', user_words.strip())
        words_list = [w.strip() for w in words_list if w.strip()]

        if len(words_list) < 2:
            # Удаляем loading сообщение перед отправкой ошибки
            try:
                await loading_msg.delete()
            except Exception as e:
                logger.warning(f"Не удалось удалить loading сообщение: {e}")

            await update.message.reply_text(
                "❌ Пожалуйста, введите хотя бы 2 слова.\n\n"
                "Попробуйте снова, нажав кнопку '🎲 Свои случайные слова'"
            )
            return

        # Форматируем список слов
        formatted_words = ', '.join(words_list)

        # Получаем event loop для async операций
        loop = asyncio.get_event_loop()

        # Этап 1: Формируем образ и роль на основе слов пользователя
        prompt1 = f"""Даны следующие слова: {formatted_words}

На основе этих слов сформируй яркий образ или метафору и определи роль мыслителя.
Ответ должен быть не более 100 слов."""

        step1_image = await loop.run_in_executor(None, get_openai_response, prompt1)

        # Этап 2: Формируем вопрос на основе образа
        prompt2 = f"""{step1_image}

Сформируй философский вопрос на основе этого образа."""

        step2_question = await loop.run_in_executor(None, get_openai_response, prompt2)

        # Этап 3: Отвечаем на вопрос (это будет опубликовано)
        prompt3 = f"""{step2_question}

Ответь на этот вопрос, не больше 100 слов."""

        step3_answer = await loop.run_in_executor(None, get_openai_response, prompt3)

        # Сохраняем в базу данных (как срочная мысль)
        thought = await db.save_thought(
            chat_id=chat_id,
            step1_words=f"Пользовательские слова: {formatted_words}",
            step1_image=step1_image,
            step2_question=step2_question,
            step3_answer=step3_answer,
            is_published=False,
            was_paid=False
        )

        # Удаляем loading сообщение
        try:
            await loading_msg.delete()
        except Exception as e:
            logger.warning(f"Не удалось удалить loading сообщение: {e}")

        # Отправляем результат пользователю с inline кнопками
        message = f"🧠 Философская мысль на основе ваших слов:\n\n{step3_answer}"

        # Добавляем inline кнопки для раскрытия деталей
        keyboard = [
            [InlineKeyboardButton("🔍 Раскрыть промпт", callback_data=f"reveal_prompt_{thought.id}")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(message, reply_markup=reply_markup)

        logger.info(f"Сгенерирована мысль на основе пользовательских слов для чата {chat_id}, ID мысли: {thought.id}")

    except Exception as e:
        logger.error(f"Ошибка генерации мысли на основе пользовательских слов: {e}")
        await update.message.reply_text("❌ Произошла ошибка при генерации. Попробуйте позже.")


async def handle_question_generation(update: Update, context: ContextTypes.DEFAULT_TYPE,
                                     user_question: str, chat_id: str):
    """Генерация ответа на вопрос пользователя (только этап 3, сохранение в БД с '-' для пропущенных этапов)"""
    try:
        loading_msg = await update.message.reply_text("⏳ Генерирую философский ответ на ваш вопрос...")

        if len(user_question.strip()) < 5:
            # Удаляем loading сообщение перед отправкой ошибки
            try:
                await loading_msg.delete()
            except Exception as e:
                logger.warning(f"Не удалось удалить loading сообщение: {e}")

            await update.message.reply_text(
                "❌ Вопрос слишком короткий.\n\n"
                "Попробуйте снова, нажав кнопку '❓ Ваш вопрос'"
            )
            return

        # Получаем event loop для async операций
        loop = asyncio.get_event_loop()

        # Сразу генерируем ответ (это соответствует 3 этапу обычной генерации)
        prompt = f"""{user_question}

Ответь на этот вопрос философски, не больше 100 слов."""

        answer = await loop.run_in_executor(None, get_openai_response, prompt)

        # Сохраняем в базу данных с прочерками для пропущенных этапов
        thought = await db.save_thought(
            chat_id=chat_id,
            step1_words="-",
            step1_image="-",
            step2_question=user_question,
            step3_answer=answer,
            is_published=False,
            was_paid=False
        )

        # Удаляем loading сообщение
        try:
            await loading_msg.delete()
        except Exception as e:
            logger.warning(f"Не удалось удалить loading сообщение: {e}")

        # Отправляем результат пользователю
        message = f"💭 Философский ответ на ваш вопрос:\n\n{answer}"

        # Отправляем без кнопок
        await update.message.reply_text(message)

        logger.info(f"Сгенерирован ответ на вопрос пользователя для чата {chat_id}, ID мысли: {thought.id}")

    except Exception as e:
        logger.error(f"Ошибка генерации ответа на вопрос: {e}")
        await update.message.reply_text("❌ Произошла ошибка при генерации. Попробуйте позже.")


async def handle_reveal_specific_question(query, thought_id: int):
    """Раскрыть вопрос для конкретной мысли"""
    try:
        # Получаем мысль по ID
        from sqlalchemy import select
        async with db.async_session() as session:
            result = await session.execute(
                select(Thought).where(Thought.id == thought_id)
            )
            thought = result.scalar_one_or_none()

        if not thought:
            await query.message.reply_text("❌ Мысль не найдена.")
            return

        # Показываем вопрос (бесплатно для пользовательских слов)
        message = f"❓ Вопрос, на который отвечает эта мысль:\n\n{thought.step2_question}"
        await query.message.reply_text(message)

    except Exception as e:
        logger.error(f"Ошибка раскрытия вопроса: {e}")
        await query.message.reply_text("❌ Произошла ошибка. Попробуйте позже.")


async def handle_reveal_specific_prompt(query, thought_id: int):
    """Раскрыть промпт для конкретной мысли"""
    try:
        # Получаем мысль по ID
        from sqlalchemy import select
        async with db.async_session() as session:
            result = await session.execute(
                select(Thought).where(Thought.id == thought_id)
            )
            thought = result.scalar_one_or_none()

        if not thought:
            await query.message.reply_text("❌ Мысль не найдена.")
            return

        # Показываем полный процесс (бесплатно для пользовательских слов)
        message = f"""🔍 Полный процесс генерации этой мысли:

📝 Шаг 1 - Исходные слова:
{thought.step1_words}

🎨 Шаг 2 - Образ и роль:
{thought.step1_image}

❓ Шаг 3 - Вопрос:
{thought.step2_question}

💭 Шаг 4 - Ответ:
{thought.step3_answer}
"""
        await query.message.reply_text(message)

    except Exception as e:
        logger.error(f"Ошибка раскрытия промпта: {e}")
        await query.message.reply_text("❌ Произошла ошибка. Попробуйте позже.")


async def handle_donation_payment(query, chat_id: str, user_id: str):
    """Обработка пожертвования - показываем варианты сумм"""
    try:
        # Удаляем предыдущее сообщение
        try:
            await query.message.delete()
        except Exception as e:
            logger.warning(f"Не удалось удалить сообщение: {e}")

        message = (
            "💝 Выберите сумму пожертвования или введите свою:\n\n"
            "ℹ️ Каждые 50₽ = +3 запроса к балансу\n"
            "├ 50₽ → +3 запроса\n"
            "├ 100₽ → +6 запросов\n"
            "├ 200₽ → +12 запросов\n"
            "├ 500₽ → +30 запросов\n"
            "└ 1000₽ → +60 запросов"
        )

        # Предлагаем варианты сумм
        keyboard = [
            [
                InlineKeyboardButton("50₽ (+3)", callback_data="donate_50"),
                InlineKeyboardButton("100₽ (+6)", callback_data="donate_100"),
                InlineKeyboardButton("200₽ (+12)", callback_data="donate_200")
            ],
            [
                InlineKeyboardButton("500₽ (+30)", callback_data="donate_500"),
                InlineKeyboardButton("1000₽ (+60)", callback_data="donate_1000")
            ],
            [InlineKeyboardButton("💬 Ввести свою сумму", callback_data="donate_custom")],
            [InlineKeyboardButton("⬅️ Назад", callback_data="back_to_menu")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        # Используем update.effective_chat для отправки нового сообщения
        from telegram import Bot
        bot = query.get_bot()
        await bot.send_message(chat_id=chat_id, text=message, reply_markup=reply_markup)

    except Exception as e:
        logger.error(f"Ошибка обработки пожертвования: {e}")
        try:
            from telegram import Bot
            bot = query.get_bot()
            await bot.send_message(chat_id=chat_id, text="❌ Произошла ошибка. Попробуйте позже.")
        except:
            pass


async def process_donation(query, chat_id: str, user_id: str, amount: int):
    """Обработка пожертвования с конкретной суммой"""
    try:
        if amount < PRICE_DONATION_MIN:
            await query.message.reply_text(
                f"❌ Минимальная сумма пожертвования: {PRICE_DONATION_MIN}₽"
            )
            return

        # Интеграция с Robokassa
        from payments import PaymentService, create_donation_payment
        keys = load_env_keys()

        payment_service = PaymentService(
            merchant_login=keys['ROBOKASSA_MERCHANT_LOGIN'],
            password1=keys['ROBOKASSA_PASSWORD1'],
            password2=keys['ROBOKASSA_PASSWORD2'],
            db=db,
            is_test=False
        )

        payment_url = await create_donation_payment(
            payment_service, chat_id, user_id, amount
        )

        if payment_url:
            # Помечаем пользователя как донора (будет активировано после успешной оплаты)
            # Пока просто создаем платеж
            keyboard = [[InlineKeyboardButton("💳 Перейти к оплате", url=payment_url)]]
            reply_markup = InlineKeyboardMarkup(keyboard)

            # Вычисляем количество бонусных запросов
            bonus_requests = (amount // 50) * 3

            message = f"""💝 Спасибо за желание поддержать проект!

Сумма: {amount}₽

После успешной оплаты вы получите:
✨ +{bonus_requests} запросов к вашему балансу

💡 Каждый день восстанавливается минимум 3 запроса.
Пока у вас больше 3 запросов - восстановление не происходит.

Нажмите кнопку ниже для оплаты:"""

            await query.message.reply_text(message, reply_markup=reply_markup)
        else:
            await query.message.reply_text(
                "❌ Ошибка создания платежа. Попробуйте позже."
            )

    except Exception as e:
        logger.error(f"Ошибка обработки пожертвования: {e}")
        await query.message.reply_text("❌ Произошла ошибка. Попробуйте позже.")


async def main():
    """Основная функция запуска бота"""
    global db, stop_flag

    logger.info("Запуск AI Filosof бота...")

    try:
        # Инициализация базы данных
        db = Database()
        await db.init_db()

        # Загрузка ключей
        keys = load_env_keys()
        tg_token = keys['TG_TOKEN']

        if not tg_token:
            logger.error("TG_TOKEN не найден в .env")
            return

        # Создание бота
        app = Application.builder().token(tg_token).build()

        # Регистрация обработчиков
        app.add_handler(CommandHandler("start", start_command))
        app.add_handler(CallbackQueryHandler(button_callback))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_message_handler))

        # Запуск бота
        async with app:
            await app.initialize()
            await app.start()
            await app.updater.start_polling()

            # Ожидаем завершения
            logger.info("Бот запущен и ожидает сообщений...")
            await asyncio.Event().wait()

    except KeyboardInterrupt:
        logger.info("Бот остановлен по Ctrl+C")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
    finally:
        stop_flag = True
        if db:
            await db.close()
        logger.info("Бот завершен")


if __name__ == "__main__":
    asyncio.run(main())
