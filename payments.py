"""
YooKassa payment integration module
"""
import uuid
import logging
from typing import Optional, Dict
from yookassa import Configuration, Payment as YooPayment
from database import Database

logger = logging.getLogger(__name__)

# Цены в рублях
PRICES = {
    'urgent_thought': 100,  # Срочная генерация мысли
    'reveal_question': 50,  # Раскрыть вопрос
    'reveal_prompt': 200,  # Раскрыть промпт
}


class PaymentService:
    """Сервис для работы с платежами через YooKassa"""

    def __init__(self, shop_id: str, secret_key: str, db: Database):
        """
        Инициализация сервиса платежей

        Args:
            shop_id: ID магазина в YooKassa
            secret_key: Секретный ключ YooKassa
            db: Экземпляр базы данных
        """
        Configuration.account_id = shop_id
        Configuration.secret_key = secret_key
        self.db = db
        logger.info("Сервис платежей инициализирован")

    async def create_payment(
        self,
        chat_id: str,
        user_id: str,
        payment_type: str,
        return_url: str = None
    ) -> Optional[Dict]:
        """
        Создает платеж в YooKassa

        Args:
            chat_id: ID чата
            user_id: ID пользователя
            payment_type: Тип платежа (urgent_thought, reveal_question, reveal_prompt)
            return_url: URL для возврата после оплаты

        Returns:
            Словарь с информацией о платеже или None в случае ошибки
        """
        try:
            if payment_type not in PRICES:
                logger.error(f"Неизвестный тип платежа: {payment_type}")
                return None

            amount = PRICES[payment_type]
            idempotence_key = str(uuid.uuid4())

            # Создаем платеж в YooKassa
            payment = YooPayment.create({
                "amount": {
                    "value": str(amount),
                    "currency": "RUB"
                },
                "confirmation": {
                    "type": "redirect",
                    "return_url": return_url or "https://t.me/your_bot"
                },
                "capture": True,
                "description": self._get_payment_description(payment_type),
                "metadata": {
                    "chat_id": chat_id,
                    "user_id": user_id,
                    "payment_type": payment_type
                }
            }, idempotence_key)

            # Сохраняем платеж в базу данных
            await self.db.create_payment(
                chat_id=chat_id,
                user_id=user_id,
                payment_id=payment.id,
                amount=amount * 100,  # В копейках
                payment_type=payment_type,
                payment_metadata={
                    "confirmation_url": payment.confirmation.confirmation_url
                }
            )

            logger.info(f"Создан платеж {payment.id} для пользователя {user_id}")

            return {
                "payment_id": payment.id,
                "confirmation_url": payment.confirmation.confirmation_url,
                "amount": amount,
                "status": payment.status
            }

        except Exception as e:
            logger.error(f"Ошибка создания платежа: {e}")
            return None

    async def check_payment_status(self, payment_id: str) -> Optional[str]:
        """
        Проверяет статус платежа в YooKassa

        Args:
            payment_id: ID платежа

        Returns:
            Статус платежа (pending, succeeded, canceled) или None
        """
        try:
            payment = YooPayment.find_one(payment_id)

            # Обновляем статус в БД
            await self.db.update_payment_status(payment_id, payment.status)

            logger.info(f"Статус платежа {payment_id}: {payment.status}")
            return payment.status

        except Exception as e:
            logger.error(f"Ошибка проверки статуса платежа {payment_id}: {e}")
            return None

    async def handle_webhook(self, notification_data: dict) -> bool:
        """
        Обрабатывает webhook от YooKassa

        Args:
            notification_data: Данные уведомления

        Returns:
            True если обработка успешна, False иначе
        """
        try:
            event = notification_data.get('event')
            payment_data = notification_data.get('object')

            if not payment_data:
                logger.error("Нет данных о платеже в webhook")
                return False

            payment_id = payment_data.get('id')
            status = payment_data.get('status')

            # Обновляем статус в БД
            await self.db.update_payment_status(payment_id, status)

            logger.info(f"Webhook обработан: платеж {payment_id}, статус {status}")
            return True

        except Exception as e:
            logger.error(f"Ошибка обработки webhook: {e}")
            return False

    @staticmethod
    def _get_payment_description(payment_type: str) -> str:
        """Возвращает описание платежа для YooKassa"""
        descriptions = {
            'urgent_thought': 'Срочная генерация философской мысли',
            'reveal_question': 'Раскрытие вопроса к мысли',
            'reveal_prompt': 'Раскрытие полного промпта',
        }
        return descriptions.get(payment_type, 'Оплата услуги')


# Функции-помощники для интеграции с ботом

async def create_urgent_thought_payment(
    payment_service: PaymentService,
    chat_id: str,
    user_id: str
) -> Optional[str]:
    """
    Создает платеж за срочную мысль и возвращает URL для оплаты

    Args:
        payment_service: Сервис платежей
        chat_id: ID чата
        user_id: ID пользователя

    Returns:
        URL для оплаты или None
    """
    result = await payment_service.create_payment(
        chat_id=chat_id,
        user_id=user_id,
        payment_type='urgent_thought'
    )

    if result:
        return result['confirmation_url']
    return None


async def create_reveal_question_payment(
    payment_service: PaymentService,
    chat_id: str,
    user_id: str
) -> Optional[str]:
    """
    Создает платеж за раскрытие вопроса и возвращает URL для оплаты
    """
    result = await payment_service.create_payment(
        chat_id=chat_id,
        user_id=user_id,
        payment_type='reveal_question'
    )

    if result:
        return result['confirmation_url']
    return None


async def create_reveal_prompt_payment(
    payment_service: PaymentService,
    chat_id: str,
    user_id: str
) -> Optional[str]:
    """
    Создает платеж за раскрытие промпта и возвращает URL для оплаты
    """
    result = await payment_service.create_payment(
        chat_id=chat_id,
        user_id=user_id,
        payment_type='reveal_prompt'
    )

    if result:
        return result['confirmation_url']
    return None
