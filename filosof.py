import random
import json
import asyncio
from openai import OpenAI
import os
from dotenv import load_dotenv
import vk_api
import requests
import logging
from typing import List, Tuple, Optional, Dict
from datetime import datetime, time as dt_time, timedelta
import pytz
from dataclasses import dataclass
import time


# Настройка логирования с московским временем
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

# Устанавливаем московское время для всех логов
for handler in logging.root.handlers:
    handler.setFormatter(MoscowTimeFormatter())

# Константы
TG_CHAT_ID = "@prorochestva_ot_bota"
GENERATION_OFFSET = 600  # 10 минут до публикации
STATE_FILE = "prophecy_state.json"  # Файл для сохранения состояния

# Глобальный флаг для остановки
stop_flag = False

# Московский часовой пояс
MOSCOW_TZ = pytz.timezone('Europe/Moscow')


@dataclass
class ProphecySchedule:
    """Расписание для пророчества"""
    generation_time: datetime  # Когда генерировать
    publish_time: datetime  # Когда публиковать
    prophecy: Optional[str] = None  # Сгенерированное пророчество
    generated: bool = False  # Сгенерировано ли


def load_env_keys() -> Dict[str, Optional[str]]:
    """
    Загружает ключи из .env файла при каждом вызове.
    Это позволяет обновлять .env во время работы программы.
    """
    load_dotenv(override=True)  # override=True перезагружает переменные

    keys = {
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        'TG_TOKEN': os.getenv('TG_TOKEN'),
        'VK_TOKEN': os.getenv('VK_TOKEN')

    }

    return keys


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

    # Случайное время на завтра
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

    if missing_elements:
        logger.debug(f"Элементы, не попавшие в выборку: {missing_elements[:5]}")

    return lst_choice, random.sample(missing_elements, min(2, len(missing_elements)))


def create_dct(sampled_lst: list) -> List[Tuple[str, int]]:
    """Создает список топ-3 самых частых слов"""
    frequency_dict = {}
    for word in sampled_lst:
        frequency_dict[word] = frequency_dict.get(word, 0) + 1

    sorted_items = sorted(frequency_dict.items(), key=lambda x: x[1], reverse=True)
    return sorted_items[:3]


def send_to_telegram(message: str) -> bool:
    """Отправляет сообщение в Telegram канал"""
    try:
        # Загружаем ключи при каждом запросе
        keys = load_env_keys()
        tg_token = keys['TG_TOKEN']

        if not tg_token:
            logger.error("TG_TOKEN не найден в .env файле")
            return False

        url = f"https://api.telegram.org/bot{tg_token}/sendMessage"
        payload = {
            'chat_id': TG_CHAT_ID,
            'text': message,
            'parse_mode': 'HTML'
        }

        response = requests.post(url, data=payload, timeout=10)
        response.raise_for_status()
        logger.info("Сообщение успешно отправлено в Telegram")
        return True

    except Exception as e:
        logger.error(f"Ошибка отправки в Telegram: {e}")
        return False


def send_to_vk(message: str) -> bool:
    """Отправляет сообщение в группу VK
    необходимо заменить на отправку сообщения в чат группы"""
    try:
        # Загружаем ключи при каждом запросе
        keys = load_env_keys()
        vk_token = keys['VK_TOKEN']

        if not vk_token:
            logger.error("VK_TOKEN не найден в .env файле")
            return False

        vk_session = vk_api.VkApi(token=vk_token)
        vk = vk_session.get_api()

        group_id = -229101116

        result = vk.wall.post(
            owner_id=group_id,
            message=message,
            from_group=1
        )
        logger.info(f"Ответ VK: {result}")
        logger.info("Сообщение успешно отправлено в VK")
        return True

    except Exception as e:
        logger.error(f"Ошибка отправки в VK: {e}")
        return False


def get_openai_response(prompt: str, max_retries: int = 3) -> str:
    """
    Получает ответ от OpenAI API.
    Ключ загружается при каждом вызове, что позволяет обновлять .env во время работы программы.
    """
    # Загружаем ключи при каждом запросе
    keys = load_env_keys()
    openai_api_key = keys['OPENAI_API_KEY']

    if not openai_api_key:
        logger.error("OPENAI_API_KEY не найден в .env файле")
        return "Моя магия слов закончилась ровно там, где началась ваша надежда услышать нечто волшебное. Пророчествовать не буду, ибо мой ключ API отсутствует."

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
                model="gpt-5.1",
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
                logger.info(f"Ожидание {wait_time} секунд перед повторной попыткой...")
                time.sleep(wait_time)
            else:
                logger.error("Все попытки получить ответ от OpenAI провалились")
                return "Моя магия слов закончилась ровно там, где началась ваша надежда услышать нечто волшебное. Пророчествовать не буду, ибо моя хрустальная сфера сегодня затуманилась по техническим причинам."


async def async_input_listener():
    """Асинхронный слушатель ввода для остановки программы"""
    global stop_flag
    loop = asyncio.get_event_loop()

    while not stop_flag:
        try:
            user_input = await loop.run_in_executor(None, input, "Введите 'stop' для остановки программы: ")

            if user_input.strip().lower() in ['stop', '0', 'exit', 'quit']:
                logger.info("Получена команда остановки")
                stop_flag = True
                break

        except (EOFError, KeyboardInterrupt):
            break
        except Exception as e:
            logger.error(f"Ошибка ввода: {e}")
            await asyncio.sleep(1)


class ProphecyScheduler:
    """Планировщик для генерации и публикации пророчеств"""

    def __init__(self):
        self.next_publish_time: Optional[datetime] = None
        self.next_generation_time: Optional[datetime] = None
        self.current_prophecy: Optional[str] = None
        self.is_generating: bool = False
        self.generated_for_current_cycle: bool = False  # Флаг для предотвращения повторной генерации
        self.planned_next_publish_time: Optional[datetime] = None  # Время, указанное в пророчестве для следующей публикации

        # Загружаем словари один раз
        try:
            with open("nouns.json", "r", encoding='utf-8') as fh:
                self.nouns = json.load(fh)
            with open("verbs.json", "r", encoding='utf-8') as fh:
                self.verbs = json.load(fh)
            with open("adject.json", "r", encoding='utf-8') as fh:
                self.adjectives = json.load(fh)

            logger.info(
                f"Загружено: существительных - {len(self.nouns)}, глаголов - {len(self.verbs)}, прилагательных - {len(self.adjectives)}")
        except Exception as e:
            logger.error(f"Ошибка загрузки словарей: {e}")
            raise

    def save_state(self):
        """
        Сохраняет текущее состояние в файл.
        Это позволяет восстановить состояние после перезапуска программы.
        """
        try:
            state = {
                'next_publish_time': self.next_publish_time.isoformat() if self.next_publish_time else None,
                'next_generation_time': self.next_generation_time.isoformat() if self.next_generation_time else None,
                'current_prophecy': self.current_prophecy,
                'is_generating': self.is_generating,
                'generated_for_current_cycle': self.generated_for_current_cycle,
                'planned_next_publish_time': self.planned_next_publish_time.isoformat() if self.planned_next_publish_time else None
            }

            with open(STATE_FILE, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)

            logger.debug(f"Состояние сохранено в {STATE_FILE}")
        except Exception as e:
            logger.error(f"Ошибка сохранения состояния: {e}")

    def load_state(self) -> bool:
        """
        Загружает состояние из файла, если он существует.
        Возвращает True, если состояние успешно загружено, False в противном случае.
        """
        try:
            if not os.path.exists(STATE_FILE):
                logger.info(f"Файл состояния {STATE_FILE} не найден, начинаем с нуля")
                return False

            with open(STATE_FILE, 'r', encoding='utf-8') as f:
                state = json.load(f)

            # Восстанавливаем времена
            if state['next_publish_time']:
                self.next_publish_time = datetime.fromisoformat(state['next_publish_time'])
            if state['next_generation_time']:
                self.next_generation_time = datetime.fromisoformat(state['next_generation_time'])
            if state.get('planned_next_publish_time'):
                self.planned_next_publish_time = datetime.fromisoformat(state['planned_next_publish_time'])

            # Восстанавливаем пророчество и флаги
            self.current_prophecy = state.get('current_prophecy')
            self.is_generating = state.get('is_generating', False)
            self.generated_for_current_cycle = state.get('generated_for_current_cycle', False)

            logger.info(f"Состояние восстановлено из {STATE_FILE}")
            if self.next_publish_time:
                logger.info(f"Следующая публикация: {format_moscow_time(self.next_publish_time)}")
            if self.next_generation_time:
                logger.info(f"Следующая генерация: {format_moscow_time(self.next_generation_time)}")
            if self.current_prophecy:
                logger.info(f"Найдено сохраненное пророчество (готово к публикации)")

            return True
        except Exception as e:
            logger.error(f"Ошибка загрузки состояния: {e}")
            return False

    async def initialize(self):
        """
        Инициализация при старте программы.
        Проверяет наличие сохраненного состояния и восстанавливает его, если возможно.
        """
        logger.info("Инициализация программы...")

        # Пытаемся загрузить сохраненное состояние
        state_loaded = self.load_state()

        if state_loaded:
            # Проверяем, не устарело ли состояние
            now = get_moscow_time()

            # Если время публикации уже прошло и есть пророчество - публикуем сразу
            if self.current_prophecy and self.next_publish_time and now >= self.next_publish_time:
                logger.info("Найдено непубликованное пророчество, публикуем немедленно...")
                await self._publish_scheduled_prophecy()
            # Если время генерации прошло, но пророчество не сгенерировано - генерируем
            elif not self.current_prophecy and self.next_generation_time and now >= self.next_generation_time and not self.generated_for_current_cycle:
                logger.info("Пропущена генерация, генерируем пророчество немедленно...")
                await self._generate_next_prophecy()
            else:
                logger.info("Состояние актуально, продолжаем работу по расписанию")
                return

        # Если состояние не загружено или нет запланированного времени - создаем новое расписание
        if not self.next_publish_time:
            logger.info("Создание нового расписания - генерация первого пророчества...")

            # Генерируем время следующей публикации (на завтра)
            self.next_publish_time = generate_next_publish_time()
            self.next_generation_time = self.next_publish_time - timedelta(seconds=GENERATION_OFFSET)

            # Сбрасываем флаг генерации для нового цикла
            self.generated_for_current_cycle = False

            # Сохраняем состояние
            self.save_state()

            # Генерируем и публикуем пророчество сразу
            await self._generate_and_publish_immediate()

            logger.info(
                f"Первое пророчество опубликовано. Следующее будет сгенерировано в {format_moscow_time(self.next_generation_time)} и опубликовано в {format_moscow_time(self.next_publish_time)}")

    async def _generate_and_publish_immediate(self):
        """Немедленная генерация и публикация пророчества (при старте программы)"""
        try:
            # Генерируем пророчество
            prophecy = await self._generate_prophecy()

            # Используем уже запланированное время следующей публикации
            next_next_publish_time = self.next_publish_time
            next_next_time_str = format_moscow_time(next_next_publish_time)

            # Формируем сообщение с указанием времени следующей публикации
            current_time_str = format_moscow_time()
            full_message = f"🔮 Пророчество от бота ({current_time_str} МСК):\n\n{prophecy}\n\n" \
                           f"⏰ Следующее пророчество будет опубликовано {next_next_time_str} МСК"

            # Сохраняем состояние
            self.save_state()

            # Логируем пророчество
            with open("prophecies_log.txt", "a", encoding="utf-8") as log_file:
                log_file.write(f"\n{format_moscow_time()} - ПЕРВОЕ ПРОРОЧЕСТВО\n")
                log_file.write(f"Следующая публикация: {next_next_time_str}\n")
                log_file.write(f"{prophecy}\n{'-' * 50}\n")

            # Публикуем
            await self._publish_prophecy(full_message)

            logger.info(
                f"Первое пророчество опубликовано. Следующее будет сгенерировано в {format_moscow_time(self.next_generation_time)} и опубликовано в {next_next_time_str}")

        except Exception as e:
            logger.error(f"Ошибка при немедленной генерации и публикации: {e}")

    async def run(self):
        """Основной цикл планировщика"""
        logger.info("Запуск основного цикла планировщика...")

        while not stop_flag:
            now = get_moscow_time()

            # Проверяем, пора ли генерировать следующее пророчество
            # ВАЖНО: генерируем только если еще не генерировали для текущего цикла
            if (self.next_generation_time and
                now >= self.next_generation_time and
                not self.is_generating and
                not self.generated_for_current_cycle):
                logger.info(f"Пора генерировать следующее пророчество!")
                await self._generate_next_prophecy()

            # Проверяем, пора ли публиковать
            if self.current_prophecy and self.next_publish_time and now >= self.next_publish_time:
                logger.info(f"Пора публиковать пророчество!")
                await self._publish_scheduled_prophecy()

            # Точное ожидание 1 секунды
            await asyncio.sleep(1)

        # Сохраняем состояние при выходе
        logger.info("Сохранение состояния перед выходом...")
        self.save_state()

    async def _generate_next_prophecy(self):
        """Генерация следующего пророчества по расписанию"""
        self.is_generating = True
        self.save_state()  # Сохраняем флаг генерации

        try:
            logger.info("Начало генерации пророчества по расписанию...")

            # Генерируем пророчество
            prophecy = await self._generate_prophecy()

            # Определяем время СЛЕДУЮЩЕЙ публикации (после той, которая сейчас запланирована)
            next_next_publish_time = generate_next_publish_time()
            next_next_time_str = format_moscow_time(next_next_publish_time)

            # Сохраняем запланированное время для использования при публикации
            self.planned_next_publish_time = next_next_publish_time

            # Формируем сообщение для публикации с указанием времени СЛЕДУЮЩЕЙ публикации
            current_publish_time_str = format_moscow_time(self.next_publish_time)
            full_message = f"🔮 Пророчество от бота ({current_publish_time_str} МСК):\n\n{prophecy}\n\n" \
                           f"⏰ Следующее пророчество будет опубликовано {next_next_time_str} МСК"

            self.current_prophecy = full_message
            self.generated_for_current_cycle = True  # Устанавливаем флаг, что генерация выполнена
            self.save_state()  # Сохраняем сгенерированное пророчество и запланированное время

            logger.info(f"Пророчество сгенерировано, готово к публикации в {current_publish_time_str}")
            logger.info(f"Следующее пророчество после этой публикации будет в {next_next_time_str}")

            # Логируем сгенерированное пророчество
            with open("prophecies_log.txt", "a", encoding="utf-8") as log_file:
                log_file.write(f"\n{format_moscow_time()} - СГЕНЕРИРОВАНО ДЛЯ ПУБЛИКАЦИИ\n")
                log_file.write(f"Время публикации: {current_publish_time_str}\n")
                log_file.write(f"Следующая публикация: {next_next_time_str}\n")
                log_file.write(f"{prophecy}\n{'-' * 50}\n")

        except Exception as e:
            logger.error(f"Ошибка генерации пророчества: {e}")
            self.current_prophecy = None
            self.generated_for_current_cycle = False
        finally:
            self.is_generating = False
            self.save_state()

    async def _generate_prophecy(self) -> str:
        """Генерация пророчества на основе случайных слов"""
        try:
            # Генерация случайных выборок
            sample_size = random.randint(100, 20000)

            # Создание случайных выборок
            noun_samples = [random.choice(self.nouns) for _ in range(sample_size)]
            verb_samples = [random.choice(self.verbs) for _ in range(sample_size)]
            adjective_samples = [random.choice(self.adjectives) for _ in range(sample_size)]

            # Анализ частотности
            choice_nouns, rare_nouns = optimized_choice_lst(noun_samples)
            choice_verbs, rare_verbs = optimized_choice_lst(verb_samples)
            choice_adjectives, rare_adjectives = optimized_choice_lst(adjective_samples)

            top_nouns = create_dct(choice_nouns)
            top_verbs = create_dct(choice_verbs)
            top_adjectives = create_dct(choice_adjectives)

            # Формирование промпта
            prompt = f"Существительные: {top_nouns} / {rare_nouns}\n" \
                     f"Глаголы: {top_verbs} / {rare_verbs}\n" \
                     f"Прилагательные: {top_adjectives} / {rare_adjectives}"

            # Логирование промпта
            with open("prophecies_log.txt", "a", encoding="utf-8") as log_file:
                log_file.write(f"\n{format_moscow_time()} - ГЕНЕРАЦИЯ\n{prompt}\n{'+' * 50}\n")

            # Получение ответа от OpenAI (синхронно в отдельном потоке)
            loop = asyncio.get_event_loop()
            prophecy = await loop.run_in_executor(None, get_openai_response, prompt)

            return prophecy

        except Exception as e:
            logger.error(f"Ошибка в процессе генерации: {e}")
            return "Пророчество не удалось сгенерировать. Попробуйте позже."

    async def _publish_scheduled_prophecy(self):
        """Публикация запланированного пророчества"""
        try:
            if not self.current_prophecy:
                logger.error("Нет пророчества для публикации")
                return

            logger.info(f"Публикация запланированного пророчества...")

            # Публикуем
            await self._publish_prophecy(self.current_prophecy)

            # Используем запланированное время, если оно было сохранено, иначе генерируем новое
            if self.planned_next_publish_time:
                next_next_publish_time = self.planned_next_publish_time
            else:
                next_next_publish_time = generate_next_publish_time()

            self.next_publish_time = next_next_publish_time
            self.next_generation_time = self.next_publish_time - timedelta(seconds=GENERATION_OFFSET)

            # Очищаем текущее пророчество, сбрасываем флаги и очищаем запланированное время
            self.current_prophecy = None
            self.generated_for_current_cycle = False
            self.planned_next_publish_time = None

            # Сохраняем новое состояние
            self.save_state()

            logger.info(
                f"Следующее пророчество запланировано: генерация в {format_moscow_time(self.next_generation_time)}, публикация в {format_moscow_time(self.next_publish_time)}")

        except Exception as e:
            logger.error(f"Ошибка публикации пророчества: {e}")

    async def _publish_prophecy(self, message: str):
        """Публикация пророчества в соцсети"""
        try:
            # Отправка в социальные сети
            success_count = 0
            if send_to_telegram(message):
                success_count += 1
            if send_to_vk(message):
                success_count += 1

            logger.info(f"Пророчество опубликовано в {success_count} из 2 социальных сетей")

            # Логирование публикации
            with open("prophecies_log.txt", "a", encoding="utf-8") as log_file:
                log_file.write(f"\n{format_moscow_time()} - ПУБЛИКАЦИЯ\n")
                log_file.write(f"Опубликовано в {success_count} соцсетей\n")
                log_file.write(f"{'-' * 50}\n")

        except Exception as e:
            logger.error(f"Ошибка при публикации: {e}")


async def main():
    """Основная асинхронная функция"""
    global stop_flag

    logger.info("Запуск программы пророчеств (время МСК)...")
    logger.info(f"Текущее время: {format_moscow_time()} МСК")
    logger.info(f"Генерация за {GENERATION_OFFSET} секунд до публикации")

    try:
        # Создаем планировщик
        scheduler = ProphecyScheduler()

        # Инициализируем (восстанавливаем состояние или публикуем первое пророчество)
        await scheduler.initialize()

        # Запускаем планировщик и слушатель ввода параллельно
        await asyncio.gather(
            scheduler.run(),
            async_input_listener()
        )
    except KeyboardInterrupt:
        logger.info("Программа остановлена по Ctrl+C")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
    finally:
        stop_flag = True
        logger.info("Программа завершена")


if __name__ == "__main__":
    # Проверяем наличие pytz
    try:
        import pytz
    except ImportError:
        print("Установите pytz: pip install pytz")
        exit(1)

    asyncio.run(main())