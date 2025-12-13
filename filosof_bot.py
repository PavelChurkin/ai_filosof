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

from database import Database, ChatState, Thought, GlobalSchedule
from yookassa import Configuration, Payment as YooPayment

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
GENERATION_OFFSET = 600  # 10 минут до публикации
MOSCOW_TZ = pytz.timezone('Europe/Moscow')
MAIN_CHANNEL_ID = "@filosofiya_ot_bota"  # Основной канал для публикаций

# Цены в рублях
PRICE_URGENT_THOUGHT = 100  # Срочная генерация мысли
PRICE_REVEAL_QUESTION = 50  # Раскрыть вопрос
PRICE_REVEAL_PROMPT = 200  # Раскрыть промпт
PRICE_DONATION_MIN = 10  # Минимальная сумма пожертвования (все этапы)

# Глобальные переменные
db: Optional[Database] = None
stop_flag = False


def load_env_keys() -> Dict[str, Optional[str]]:
    """Загружает ключи из .env файла"""
    load_dotenv(override=True)
    return {
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        'TG_TOKEN': os.getenv('TG_TOKEN'),
        'YOOKASSA_SHOP_ID': os.getenv('YOOKASSA_SHOP_ID'),
        'YOOKASSA_SECRET_KEY': os.getenv('YOOKASSA_SECRET_KEY'),
    }


def get_moscow_time() -> datetime:
    """Возвращает текущее время в московском часовом поясе"""
    return datetime.now(MOSCOW_TZ)


def format_moscow_time(dt: datetime = None, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Форматирует время в московском поясе"""
    if dt is None:
        dt = get_moscow_time()
    return dt.strftime(format_str)


def generate_next_publish_time() -> datetime:
    """Генерирует время следующей публикации (завтра в случайное время)"""
    now_moscow = get_moscow_time()
    tomorrow = now_moscow + timedelta(days=1)

    publish_hour = random.randint(0, 23)
    publish_minute = random.randint(0, 59)
    publish_second = random.randint(0, 59)

    publish_time = MOSCOW_TZ.localize(datetime(
        tomorrow.year, tomorrow.month, tomorrow.day,
        publish_hour, publish_minute, publish_second
    ))

    return publish_time


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

    system_message = f"Ты бот философ ({get_moscow_time().ctime()})"

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


class ThoughtScheduler:
    """Планировщик для генерации и публикации мыслей во всех чатах"""

    def __init__(self, bot_app: Application):
        self.bot_app = bot_app
        self.generator = ThoughtGenerator()
        self.is_generating = False
        self.global_thought_id = None  # ID глобальной мысли для групп/каналов

    async def initialize(self):
        """Инициализация планировщика"""
        logger.info("Инициализация планировщика мыслей...")

        # Инициализация глобального расписания для групп/каналов
        global_schedule = await db.get_global_schedule()
        if not global_schedule or not global_schedule.next_publish_time:
            next_publish = generate_next_publish_time()
            next_gen = next_publish - timedelta(seconds=GENERATION_OFFSET)
            await db.update_global_schedule(
                next_publish_time=next_publish,
                next_generation_time=next_gen
            )
            logger.info(
                f"Глобальное расписание: установлено время публикации {format_moscow_time(next_publish)}"
            )

        # Инициализация расписания для приватных чатов
        private_chats = await db.get_all_private_chats()
        for chat_state in private_chats:
            if not chat_state.next_publish_time:
                next_publish = generate_next_publish_time()
                next_gen = next_publish - timedelta(seconds=GENERATION_OFFSET)

                await db.update_chat_state(
                    chat_state.chat_id,
                    next_publish_time=next_publish,
                    next_generation_time=next_gen
                )

                logger.info(
                    f"Приватный чат {chat_state.chat_id}: установлено время публикации {format_moscow_time(next_publish)}"
                )

    async def run(self):
        """Основной цикл планировщика"""
        logger.info("Запуск основного цикла планировщика...")

        while not stop_flag:
            try:
                now = get_moscow_time()

                # Обработка глобального расписания для групп/каналов
                await self._process_global_schedule(now)

                # Обработка индивидуального расписания для приватных чатов
                await self._process_private_chats(now)

                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Ошибка в цикле планировщика: {e}")
                await asyncio.sleep(5)

    async def _process_global_schedule(self, now: datetime):
        """Обработка глобального расписания для групп и каналов"""
        global_schedule = await db.get_global_schedule()
        if not global_schedule:
            return

        # Конвертируем времена в aware datetime
        if global_schedule.next_generation_time:
            if global_schedule.next_generation_time.tzinfo is None:
                next_gen = MOSCOW_TZ.localize(global_schedule.next_generation_time)
            else:
                next_gen = global_schedule.next_generation_time
        else:
            next_gen = None

        if global_schedule.next_publish_time:
            if global_schedule.next_publish_time.tzinfo is None:
                next_pub = MOSCOW_TZ.localize(global_schedule.next_publish_time)
            else:
                next_pub = global_schedule.next_publish_time
        else:
            next_pub = None

        # Проверяем, пора ли генерировать глобальную мысль
        if next_gen and now >= next_gen and self.global_thought_id is None:
            logger.info("Генерация глобальной мысли для всех групп/каналов")
            # Используем специальный chat_id для глобальных мыслей
            thought = await self.generator.generate_thought_3_steps("global", was_paid=False)
            self.global_thought_id = thought.id

            # Сбрасываем время генерации
            await db.update_global_schedule(next_generation_time=None)

        # Проверяем, пора ли публиковать глобальную мысль
        if next_pub and now >= next_pub and self.global_thought_id is not None:
            await self._publish_global_thought()

    async def _process_private_chats(self, now: datetime):
        """Обработка индивидуального расписания для приватных чатов"""
        private_chats = await db.get_all_private_chats()

        for chat_state in private_chats:
            # Конвертируем времена в aware datetime
            if chat_state.next_generation_time:
                if chat_state.next_generation_time.tzinfo is None:
                    next_gen = MOSCOW_TZ.localize(chat_state.next_generation_time)
                else:
                    next_gen = chat_state.next_generation_time
            else:
                next_gen = None

            if chat_state.next_publish_time:
                if chat_state.next_publish_time.tzinfo is None:
                    next_pub = MOSCOW_TZ.localize(chat_state.next_publish_time)
                else:
                    next_pub = chat_state.next_publish_time
            else:
                next_pub = None

            # Проверяем, пора ли генерировать
            if next_gen and now >= next_gen:
                latest_thought = await db.get_latest_thought(chat_state.chat_id)

                # Генерируем только если последняя мысль уже опубликована или её нет
                if not latest_thought or latest_thought.is_published:
                    await self._generate_thought_for_chat(chat_state.chat_id)

                    # Сбрасываем время генерации
                    await db.update_chat_state(
                        chat_state.chat_id,
                        next_generation_time=None
                    )

            # Проверяем, пора ли публиковать
            if next_pub and now >= next_pub:
                await self._publish_thought_for_chat(chat_state.chat_id)

    async def _generate_thought_for_chat(self, chat_id: str):
        """Генерация мысли для конкретного приватного чата"""
        try:
            logger.info(f"Генерация мысли для приватного чата {chat_id}")
            await self.generator.generate_thought_3_steps(chat_id, was_paid=False)
        except Exception as e:
            logger.error(f"Ошибка генерации мысли для чата {chat_id}: {e}")

    async def _publish_global_thought(self):
        """Публикация глобальной мысли во всех группах/каналах"""
        try:
            if self.global_thought_id is None:
                logger.warning("Нет глобальной мысли для публикации")
                return

            # Получаем мысль из БД
            from sqlalchemy import select
            async with db.async_session() as session:
                result = await session.execute(
                    select(Thought).where(Thought.id == self.global_thought_id)
                )
                global_thought = result.scalar_one_or_none()

            if not global_thought:
                logger.error(f"Мысль с ID {self.global_thought_id} не найдена")
                return

            # Генерируем следующее время публикации
            next_publish = generate_next_publish_time()
            next_gen = next_publish - timedelta(seconds=GENERATION_OFFSET)

            # Формируем сообщение с вопросом и ответом
            message = f"❓ Вопрос:\n{global_thought.step2_question}\n\n" \
                     f"💭 Ответ:\n{global_thought.step3_answer}\n\n" \
                     f"⏰ Следующая мысль будет {format_moscow_time(next_publish)} МСК"

            # Добавляем inline кнопки
            keyboard = [
                [
                    InlineKeyboardButton("💭 Срочная мысль", callback_data="pay_urgent"),
                    InlineKeyboardButton("📜 Раскрыть промпт", callback_data="pay_prompt")
                ],
                [InlineKeyboardButton("💝 Пожертвование", callback_data="pay_donation")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            # Получаем все активные группы и каналы
            groups_and_channels = await db.get_all_groups_and_channels()

            # Публикуем в основной канал
            try:
                await self.bot_app.bot.send_message(
                    chat_id=MAIN_CHANNEL_ID,
                    text=message,
                    reply_markup=reply_markup
                )
                logger.info(f"Мысль опубликована в основном канале {MAIN_CHANNEL_ID}")
            except Exception as e:
                logger.error(f"Ошибка публикации в основной канал {MAIN_CHANNEL_ID}: {e}")

            # Публикуем во все зарегистрированные группы/каналы
            for chat_state in groups_and_channels:
                try:
                    await self.bot_app.bot.send_message(
                        chat_id=chat_state.chat_id,
                        text=message,
                        reply_markup=reply_markup
                    )
                    logger.info(f"Мысль опубликована в группе/канале {chat_state.chat_id}")
                except Exception as e:
                    logger.error(f"Ошибка публикации в {chat_state.chat_id}: {e}")

            # Обновляем статус мысли
            await db.update_thought(
                global_thought.id,
                is_published=True,
                published_at=datetime.utcnow()
            )

            # Обновляем глобальное расписание
            await db.update_global_schedule(
                next_publish_time=next_publish,
                next_generation_time=next_gen
            )

            # Сбрасываем ID глобальной мысли
            self.global_thought_id = None

            logger.info("Глобальная мысль успешно опубликована")

        except Exception as e:
            logger.error(f"Ошибка публикации глобальной мысли: {e}")

    async def _publish_thought_for_chat(self, chat_id: str):
        """Публикация мысли в приватный чат"""
        try:
            # Получаем последнюю неопубликованную мысль
            latest_thought = await db.get_latest_thought(chat_id)

            if not latest_thought or latest_thought.is_published:
                logger.warning(f"Нет неопубликованных мыслей для чата {chat_id}")
                return

            # Генерируем следующее время публикации
            next_publish = generate_next_publish_time()
            next_gen = next_publish - timedelta(seconds=GENERATION_OFFSET)

            # Формируем сообщение
            message = f"🧠 Философская мысль:\n\n{latest_thought.step3_answer}\n\n" \
                     f"⏰ Следующая мысль будет {format_moscow_time(next_publish)} МСК"

            # Добавляем inline кнопки
            keyboard = [
                [
                    InlineKeyboardButton("💭 Срочная мысль", callback_data="pay_urgent"),
                    InlineKeyboardButton("❓ Какой был вопрос?", callback_data="pay_question")
                ],
                [
                    InlineKeyboardButton("📜 Раскрыть промпт", callback_data="pay_prompt"),
                    InlineKeyboardButton("💝 Пожертвование", callback_data="pay_donation")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            # Отправляем сообщение
            await self.bot_app.bot.send_message(
                chat_id=chat_id,
                text=message,
                reply_markup=reply_markup
            )

            # Обновляем статус мысли
            await db.update_thought(
                latest_thought.id,
                is_published=True,
                published_at=datetime.utcnow()
            )

            # Обновляем состояние чата
            await db.update_chat_state(
                chat_id,
                next_publish_time=next_publish,
                next_generation_time=next_gen
            )

            logger.info(f"Мысль опубликована в приватном чате {chat_id}")

        except Exception as e:
            logger.error(f"Ошибка публикации мысли для чата {chat_id}: {e}")


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

    # Получаем время следующей публикации
    if chat_type == 'private':
        # Для приватных чатов используем индивидуальное расписание
        if not chat_state.next_publish_time:
            next_publish = generate_next_publish_time()
            next_gen = next_publish - timedelta(seconds=GENERATION_OFFSET)

            await db.update_chat_state(
                chat_id,
                next_publish_time=next_publish,
                next_generation_time=next_gen
            )
        else:
            # Конвертируем в aware datetime если нужно
            if chat_state.next_publish_time.tzinfo is None:
                next_publish = MOSCOW_TZ.localize(chat_state.next_publish_time)
            else:
                next_publish = chat_state.next_publish_time
    else:
        # Для групп и каналов используем глобальное расписание
        global_schedule = await db.get_global_schedule()
        if global_schedule and global_schedule.next_publish_time:
            if global_schedule.next_publish_time.tzinfo is None:
                next_publish = MOSCOW_TZ.localize(global_schedule.next_publish_time)
            else:
                next_publish = global_schedule.next_publish_time
        else:
            next_publish = generate_next_publish_time()

    # Формируем приветственное сообщение
    if chat_type == 'private':
        welcome_text = f"""
👋 Добро пожаловать в AI Filosof!

Я буду делиться философскими мыслями раз в день в случайное время.

⏰ Следующая мысль появится примерно {format_moscow_time(next_publish)} МСК

Вы можете:
💭 Получить мысль срочно (платно)
❓ Узнать вопрос, на который отвечает мысль (платно)
📜 Раскрыть весь процесс генерации (платно)
💝 Поддержать проект пожертвованием
"""
    else:
        welcome_text = f"""
👋 Добро пожаловать в AI Filosof!

Я буду делиться философскими мыслями со всеми группами и каналами одновременно раз в день в случайное время.

⏰ Следующая мысль появится примерно {format_moscow_time(next_publish)} МСК

Вы можете:
💭 Получить мысль срочно (платно)
📜 Раскрыть весь процесс генерации (платно)
💝 Поддержать проект пожертвованием

📢 Подписывайтесь на основной канал: {MAIN_CHANNEL_ID}
"""

    # Добавляем inline кнопки
    keyboard = [
        [InlineKeyboardButton("💭 Срочная мысль", callback_data="pay_urgent")],
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

    if callback_data == "pay_urgent":
        # Срочная генерация мысли
        await handle_urgent_thought_payment(query, chat_id, user_id)

    elif callback_data == "pay_question":
        # Раскрыть вопрос
        await handle_reveal_question_payment(query, chat_id, user_id)

    elif callback_data == "pay_prompt":
        # Раскрыть промпт
        await handle_reveal_prompt_payment(query, chat_id, user_id)

    elif callback_data == "pay_donation":
        # Пожертвование
        await handle_donation_payment(query, chat_id, user_id)

    elif callback_data == "donate_custom":
        # Пользователь хочет ввести свою сумму
        await query.message.reply_text(
            "💬 Пожалуйста, введите сумму пожертвования (минимум 10₽):"
        )
        # Здесь в будущем нужно добавить обработчик текстовых сообщений для получения суммы
        # Пока просто информируем пользователя

    elif callback_data.startswith("donate_"):
        # Пожертвование с конкретной суммой
        amount = int(callback_data.split("_")[1])
        await process_donation(query, chat_id, user_id, amount)


async def handle_urgent_thought_payment(query, chat_id: str, user_id: str):
    """Обработка платежа за срочную мысль"""
    try:
        # В реальной версии здесь должна быть интеграция с YooKassa
        # Для демонстрации просто генерируем мысль

        await query.message.reply_text("⏳ Генерирую срочную мысль...")

        generator = ThoughtGenerator()
        thought = await generator.generate_thought_3_steps(chat_id, was_paid=True)

        # Отправляем результат
        message = f"🧠 Срочная философская мысль:\n\n{thought.step3_answer}"

        # Добавляем inline кнопки
        keyboard = [
            [
                InlineKeyboardButton("💭 Срочная мысль", callback_data="pay_urgent"),
                InlineKeyboardButton("❓ Какой был вопрос?", callback_data="pay_question")
            ],
            [
                InlineKeyboardButton("📜 Раскрыть промпт", callback_data="pay_prompt"),
                InlineKeyboardButton("💝 Пожертвование", callback_data="pay_donation")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await query.message.reply_text(message, reply_markup=reply_markup)

        # Помечаем как опубликованную
        await db.update_thought(thought.id, is_published=True, published_at=datetime.utcnow())

    except Exception as e:
        logger.error(f"Ошибка обработки срочной мысли: {e}")
        await query.message.reply_text("❌ Произошла ошибка. Попробуйте позже.")


async def handle_reveal_question_payment(query, chat_id: str, user_id: str):
    """Обработка платежа за раскрытие вопроса"""
    try:
        # Получаем последнюю опубликованную мысль
        latest_thought = await db.get_latest_thought(chat_id)

        if not latest_thought or not latest_thought.is_published:
            await query.message.reply_text("❌ Нет опубликованных мыслей для раскрытия вопроса.")
            return

        # В реальной версии здесь проверка платежа
        message = f"❓ Вопрос, на который отвечает последняя мысль:\n\n{latest_thought.step2_question}"
        await query.message.reply_text(message)

    except Exception as e:
        logger.error(f"Ошибка раскрытия вопроса: {e}")
        await query.message.reply_text("❌ Произошла ошибка. Попробуйте позже.")


async def handle_reveal_prompt_payment(query, chat_id: str, user_id: str):
    """Обработка платежа за раскрытие промпта"""
    try:
        # Получаем последнюю опубликованную мысль
        latest_thought = await db.get_latest_thought(chat_id)

        if not latest_thought or not latest_thought.is_published:
            await query.message.reply_text("❌ Нет опубликованных мыслей для раскрытия промпта.")
            return

        # В реальной версии здесь проверка платежа
        message = f"""📜 Полный процесс генерации последней мысли:

📝 Шаг 1 - Исходные слова:
{latest_thought.step1_words}

🎨 Шаг 2 - Образ и роль:
{latest_thought.step1_image}

❓ Шаг 3 - Вопрос:
{latest_thought.step2_question}

💭 Шаг 4 - Ответ:
{latest_thought.step3_answer}
"""
        await query.message.reply_text(message)

    except Exception as e:
        logger.error(f"Ошибка раскрытия промпта: {e}")
        await query.message.reply_text("❌ Произошла ошибка. Попробуйте позже.")


async def handle_donation_payment(query, chat_id: str, user_id: str):
    """Обработка пожертвования - показываем варианты сумм"""
    try:
        message = "💝 Выберите сумму пожертвования или введите свою:"

        # Предлагаем варианты сумм
        keyboard = [
            [
                InlineKeyboardButton("50₽", callback_data="donate_50"),
                InlineKeyboardButton("100₽", callback_data="donate_100"),
                InlineKeyboardButton("200₽", callback_data="donate_200")
            ],
            [
                InlineKeyboardButton("500₽", callback_data="donate_500"),
                InlineKeyboardButton("1000₽", callback_data="donate_1000")
            ],
            [InlineKeyboardButton("💬 Ввести свою сумму", callback_data="donate_custom")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await query.message.reply_text(message, reply_markup=reply_markup)

    except Exception as e:
        logger.error(f"Ошибка обработки пожертвования: {e}")
        await query.message.reply_text("❌ Произошла ошибка. Попробуйте позже.")


async def process_donation(query, chat_id: str, user_id: str, amount: int):
    """Обработка пожертвования с конкретной суммой"""
    try:
        if amount < PRICE_DONATION_MIN:
            await query.message.reply_text(
                f"❌ Минимальная сумма пожертвования: {PRICE_DONATION_MIN}₽"
            )
            return

        # В реальной версии здесь интеграция с YooKassa
        # Пока просто подтверждаем
        message = f"""💝 Спасибо за ваше пожертвование {amount}₽!

Ваша поддержка помогает развивать проект.

🙏 Мы ценим вашу помощь!
"""
        await query.message.reply_text(message)

        # Логируем в БД (в будущем здесь будет реальный платеж)
        logger.info(f"Пожертвование от пользователя {user_id}: {amount}₽")

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

        # Создание и инициализация планировщика
        scheduler = ThoughtScheduler(app)
        await scheduler.initialize()

        # Запуск бота и планировщика параллельно
        async with app:
            await app.initialize()
            await app.start()
            await app.updater.start_polling()

            # Запускаем планировщик
            await scheduler.run()

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
