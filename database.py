"""
Database module for managing chat states and thought history
"""
import asyncio
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, Text, JSON
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, Session
from sqlalchemy.pool import StaticPool
import json
import logging

logger = logging.getLogger(__name__)

Base = declarative_base()


class ChatState(Base):
    """Хранит состояние для каждого чата (приватного или группового)"""
    __tablename__ = 'chat_states'

    id = Column(Integer, primary_key=True)
    chat_id = Column(String, unique=True, nullable=False, index=True)  # Telegram chat_id
    chat_type = Column(String, nullable=False)  # 'private' or 'group' or 'channel'
    is_active = Column(Boolean, default=True)
    # Система лимитов запросов
    last_request_date = Column(DateTime, nullable=True)  # Дата последнего запроса для отслеживания смены дня
    bonus_requests = Column(Integer, default=3)  # Оставшиеся запросы (минимум 3 в день, пожертвование: 50₽ = +3)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Thought(Base):
    """Хранит историю мыслей для каждого чата"""
    __tablename__ = 'thoughts'

    id = Column(Integer, primary_key=True)
    chat_id = Column(String, nullable=False, index=True)

    # Трехэтапный процесс генерации
    step1_words = Column(Text, nullable=True)  # Исходные слова
    step1_image = Column(Text, nullable=True)  # Результат 1: образ и роль
    step2_question = Column(Text, nullable=True)  # Результат 2: вопрос
    step3_answer = Column(Text, nullable=True)  # Результат 3: ответ (публикуется)

    # Метаданные
    generated_at = Column(DateTime, default=datetime.utcnow)
    published_at = Column(DateTime, nullable=True)
    is_published = Column(Boolean, default=False)
    was_paid = Column(Boolean, default=False)  # Была ли оплачена преждевременная генерация

    created_at = Column(DateTime, default=datetime.utcnow)


class Payment(Base):
    """Хранит историю платежей через YooKassa"""
    __tablename__ = 'payments'

    id = Column(Integer, primary_key=True)
    chat_id = Column(String, nullable=False, index=True)
    user_id = Column(String, nullable=False)
    payment_id = Column(String, unique=True, nullable=False)  # ID платежа в YooKassa
    amount = Column(Integer, nullable=False)  # Сумма в копейках
    currency = Column(String, default='RUB')
    status = Column(String, nullable=False)  # pending, succeeded, canceled
    payment_type = Column(String, nullable=False)  # 'urgent_thought', 'reveal_question', 'reveal_prompt'
    payment_metadata = Column(JSON, nullable=True)  # Дополнительные данные

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Database:
    """Асинхронная обертка для работы с базой данных"""

    def __init__(self, database_url: str = "sqlite+aiosqlite:///filosof.db"):
        self.database_url = database_url
        self.engine = create_async_engine(
            database_url,
            echo=False,
            poolclass=StaticPool,
            connect_args={"check_same_thread": False} if "sqlite" in database_url else {}
        )
        self.async_session = async_sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

    async def init_db(self):
        """Инициализация базы данных"""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            # Миграция: добавляем недостающие колонки если их нет
            await conn.run_sync(self._migrate_chat_states_columns)
        logger.info("База данных инициализирована")

    def _migrate_chat_states_columns(self, conn):
        """Миграция: добавление всех недостающих колонок в таблицу chat_states"""
        from sqlalchemy import inspect, text

        try:
            inspector = inspect(conn)
            existing_columns = [col['name'] for col in inspector.get_columns('chat_states')]
            logger.info(f"Текущие колонки в chat_states: {existing_columns}")

            # Определяем колонки, которые должны быть в таблице
            required_columns = {
                'last_request_date': 'DATETIME' if 'sqlite' in self.database_url else 'TIMESTAMP',
                'bonus_requests': 'INTEGER DEFAULT 3'
            }

            # Добавляем недостающие колонки
            for column_name, column_type in required_columns.items():
                if column_name not in existing_columns:
                    logger.info(f"Добавление колонки {column_name} в таблицу chat_states")
                    conn.execute(text(f'ALTER TABLE chat_states ADD COLUMN {column_name} {column_type}'))
                    logger.info(f"Колонка {column_name} успешно добавлена")
                else:
                    logger.debug(f"Колонка {column_name} уже существует")

        except Exception as e:
            logger.error(f"Ошибка при миграции таблицы chat_states: {e}")

    async def get_or_create_chat_state(self, chat_id: str, chat_type: str = 'private') -> ChatState:
        """Получить или создать состояние чата"""
        async with self.async_session() as session:
            # Пытаемся найти существующее состояние
            from sqlalchemy import select
            result = await session.execute(
                select(ChatState).where(ChatState.chat_id == chat_id)
            )
            chat_state = result.scalar_one_or_none()

            if not chat_state:
                # Создаем новое состояние с базовым лимитом 3 запроса
                chat_state = ChatState(
                    chat_id=chat_id,
                    chat_type=chat_type,
                    is_active=True,
                    bonus_requests=3  # Базовый дневной лимит
                )
                session.add(chat_state)
                await session.commit()
                await session.refresh(chat_state)
                logger.info(f"Создано новое состояние для чата {chat_id}")

            return chat_state

    async def update_chat_state(self, chat_id: str, **kwargs):
        """Обновить состояние чата"""
        async with self.async_session() as session:
            from sqlalchemy import select, update
            await session.execute(
                update(ChatState)
                .where(ChatState.chat_id == chat_id)
                .values(**kwargs, updated_at=datetime.utcnow())
            )
            await session.commit()

    async def save_thought(self, chat_id: str, step1_words: str = None,
                          step1_image: str = None, step2_question: str = None,
                          step3_answer: str = None, is_published: bool = False,
                          was_paid: bool = False) -> Thought:
        """Сохранить новую мысль"""
        async with self.async_session() as session:
            thought = Thought(
                chat_id=chat_id,
                step1_words=step1_words,
                step1_image=step1_image,
                step2_question=step2_question,
                step3_answer=step3_answer,
                is_published=is_published,
                was_paid=was_paid,
                published_at=datetime.utcnow() if is_published else None
            )
            session.add(thought)
            await session.commit()
            await session.refresh(thought)
            return thought

    async def update_thought(self, thought_id: int, **kwargs):
        """Обновить мысль"""
        async with self.async_session() as session:
            from sqlalchemy import update
            await session.execute(
                update(Thought)
                .where(Thought.id == thought_id)
                .values(**kwargs)
            )
            await session.commit()

    async def get_latest_thought(self, chat_id: str) -> Optional[Thought]:
        """Получить последнюю мысль для чата"""
        async with self.async_session() as session:
            from sqlalchemy import select
            result = await session.execute(
                select(Thought)
                .where(Thought.chat_id == chat_id)
                .order_by(Thought.created_at.desc())
                .limit(1)
            )
            return result.scalar_one_or_none()

    async def get_all_active_chats(self) -> List[ChatState]:
        """Получить все активные чаты"""
        async with self.async_session() as session:
            from sqlalchemy import select
            result = await session.execute(
                select(ChatState).where(ChatState.is_active == True)
            )
            return list(result.scalars().all())

    async def create_payment(self, chat_id: str, user_id: str, payment_id: str,
                           amount: int, payment_type: str, payment_metadata: dict = None) -> Payment:
        """Создать запись о платеже"""
        async with self.async_session() as session:
            payment = Payment(
                chat_id=chat_id,
                user_id=user_id,
                payment_id=payment_id,
                amount=amount,
                status='pending',
                payment_type=payment_type,
                payment_metadata=payment_metadata
            )
            session.add(payment)
            await session.commit()
            await session.refresh(payment)
            return payment

    async def update_payment_status(self, payment_id: str, status: str):
        """Обновить статус платежа"""
        async with self.async_session() as session:
            from sqlalchemy import update
            await session.execute(
                update(Payment)
                .where(Payment.payment_id == payment_id)
                .values(status=status, updated_at=datetime.utcnow())
            )
            await session.commit()

    async def get_payment(self, payment_id: str) -> Optional[Payment]:
        """Получить платеж по ID"""
        async with self.async_session() as session:
            from sqlalchemy import select
            result = await session.execute(
                select(Payment).where(Payment.payment_id == payment_id)
            )
            return result.scalar_one_or_none()

    async def get_all_groups_and_channels(self) -> List[ChatState]:
        """Получить все активные группы и каналы (не приватные чаты)"""
        async with self.async_session() as session:
            from sqlalchemy import select
            result = await session.execute(
                select(ChatState).where(
                    ChatState.is_active == True,
                    ChatState.chat_type.in_(['group', 'channel'])
                )
            )
            return list(result.scalars().all())

    async def get_all_private_chats(self) -> List[ChatState]:
        """Получить все активные приватные чаты"""
        async with self.async_session() as session:
            from sqlalchemy import select
            result = await session.execute(
                select(ChatState).where(
                    ChatState.is_active == True,
                    ChatState.chat_type == 'private'
                )
            )
            return list(result.scalars().all())

    async def check_and_update_daily_limit(self, chat_id: str) -> tuple[bool, int]:
        """
        Проверяет и обновляет дневной лимит запросов для чата

        Логика:
        - bonus_requests хранит оставшиеся бонусные запросы из пожертвований
        - Каждый запрос расходует 1 бонусный запрос (если есть)
        - При сбросе дня (новый день):
          * Если bonus_requests > 3: ничего не делаем (достаточно запросов)
          * Если bonus_requests < 3: восстанавливаем до 3 (минимальный дневной лимит)

        Args:
            chat_id: ID чата

        Returns:
            Tuple (can_make_request: bool, remaining_requests: int)
        """
        from datetime import date
        async with self.async_session() as session:
            from sqlalchemy import select
            result = await session.execute(
                select(ChatState).where(ChatState.chat_id == chat_id)
            )
            chat_state = result.scalar_one_or_none()

            if not chat_state:
                return False, 0

            today = date.today()
            last_request_date = chat_state.last_request_date.date() if chat_state.last_request_date else None

            # Сбрасываем/восстанавливаем лимит если новый день
            if last_request_date != today:
                # Если bonus_requests < 3, восстанавливаем до минимума 3
                if chat_state.bonus_requests < 3:
                    chat_state.bonus_requests = 3
                # Если >= 3, оставляем как есть (не добавляем ничего)

                chat_state.last_request_date = datetime.utcnow()

            # Проверяем, есть ли доступные запросы
            if chat_state.bonus_requests <= 0:
                return False, 0

            # Расходуем один запрос
            chat_state.bonus_requests -= 1
            chat_state.updated_at = datetime.utcnow()

            await session.commit()

            remaining = chat_state.bonus_requests
            return True, remaining

    async def add_bonus_requests(self, chat_id: str, amount: int):
        """
        Добавляет бонусные запросы пользователю за пожертвование
        50₽ = +3 запроса

        Args:
            chat_id: ID чата
            amount: Сумма пожертвования в рублях
        """
        # Вычисляем количество бонусных запросов: каждые 50₽ дают 3 запроса
        bonus_to_add = (amount // 50) * 3

        async with self.async_session() as session:
            from sqlalchemy import select
            result = await session.execute(
                select(ChatState).where(ChatState.chat_id == chat_id)
            )
            chat_state = result.scalar_one_or_none()

            if chat_state:
                chat_state.bonus_requests += bonus_to_add
                chat_state.updated_at = datetime.utcnow()
                await session.commit()
                logger.info(f"Чат {chat_id} получил +{bonus_to_add} бонусных запросов (всего: {chat_state.bonus_requests})")
            else:
                logger.error(f"Чат {chat_id} не найден для добавления бонусных запросов")

    async def close(self):
        """Закрыть соединение с базой данных"""
        await self.engine.dispose()
        logger.info("Соединение с базой данных закрыто")
