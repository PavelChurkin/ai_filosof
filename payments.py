"""
Robokassa payment integration module
"""
import hashlib
import logging
from typing import Optional, Dict
from urllib.parse import urlencode
from database import Database

logger = logging.getLogger(__name__)

# Цены в рублях
PRICES = {
    'donation': 50,  # Пожертвование (минимальная сумма)
}


class PaymentService:
    """Сервис для работы с платежами через Robokassa"""

    def __init__(self, merchant_login: str, password1: str, password2: str, db: Database, is_test: bool = False):
        """
        Инициализация сервиса платежей

        Args:
            merchant_login: Идентификатор магазина в Robokassa
            password1: Пароль #1 для формирования подписи
            password2: Пароль #2 для проверки результата оплаты
            db: Экземпляр базы данных
            is_test: Тестовый режим
        """
        self.merchant_login = merchant_login
        self.password1 = password1
        self.password2 = password2
        self.db = db
        self.is_test = is_test
        self.payment_url = "https://auth.robokassa.ru/Merchant/Index.aspx"
        logger.info("Сервис платежей Robokassa инициализирован")

    def _generate_signature(self, amount: float, inv_id: int, **kwargs) -> str:
        """
        Генерирует подпись для Robokassa

        Args:
            amount: Сумма платежа
            inv_id: Номер счета
            **kwargs: Дополнительные параметры

        Returns:
            MD5 подпись
        """
        # Формируем строку для подписи: MerchantLogin:OutSum:InvId:Password1
        data = f"{self.merchant_login}:{amount}:{inv_id}:{self.password1}"

        # Добавляем дополнительные параметры в алфавитном порядке
        if kwargs:
            sorted_params = sorted(kwargs.items())
            for key, value in sorted_params:
                data += f":{key}={value}"

        return hashlib.md5(data.encode()).hexdigest()

    async def create_payment(
        self,
        chat_id: str,
        user_id: str,
        amount: float,
        description: str = "Пожертвование"
    ) -> Optional[Dict]:
        """
        Создает платеж в Robokassa

        Args:
            chat_id: ID чата
            user_id: ID пользователя
            amount: Сумма платежа в рублях
            description: Описание платежа

        Returns:
            Словарь с информацией о платеже или None в случае ошибки
        """
        try:
            if amount < PRICES['donation']:
                logger.error(f"Сумма меньше минимальной: {amount}")
                return None

            # Генерируем уникальный ID счета
            import time
            inv_id = int(time.time() * 1000) % 2147483647  # Максимальное значение для int в некоторых БД

            # Дополнительные параметры
            shp_chat_id = chat_id
            shp_user_id = user_id

            # Генерируем подпись
            signature = self._generate_signature(
                amount,
                inv_id,
                Shp_chat_id=shp_chat_id,
                Shp_user_id=shp_user_id
            )

            # Формируем параметры для URL
            params = {
                'MrchLogin': self.merchant_login,
                'OutSum': amount,
                'InvId': inv_id,
                'Desc': description,
                'SignatureValue': signature,
                'Shp_chat_id': shp_chat_id,
                'Shp_user_id': shp_user_id,
                'IsTest': 1 if self.is_test else 0
            }

            # Формируем URL для оплаты
            payment_url = f"{self.payment_url}?{urlencode(params)}"

            # Сохраняем платеж в базу данных
            await self.db.create_payment(
                chat_id=chat_id,
                user_id=user_id,
                payment_id=str(inv_id),
                amount=int(amount * 100),  # В копейках
                payment_type='donation',
                payment_metadata={
                    "confirmation_url": payment_url,
                    "description": description
                }
            )

            logger.info(f"Создан платеж {inv_id} для пользователя {user_id}")

            return {
                "payment_id": str(inv_id),
                "confirmation_url": payment_url,
                "amount": amount,
                "status": "pending"
            }

        except Exception as e:
            logger.error(f"Ошибка создания платежа: {e}")
            return None

    def _verify_result_signature(self, out_sum: float, inv_id: int, signature: str, **kwargs) -> bool:
        """
        Проверяет подпись результата оплаты

        Args:
            out_sum: Сумма платежа
            inv_id: Номер счета
            signature: Подпись для проверки
            **kwargs: Дополнительные параметры

        Returns:
            True если подпись верна, False иначе
        """
        # Формируем строку для проверки: OutSum:InvId:Password2
        data = f"{out_sum}:{inv_id}:{self.password2}"

        # Добавляем дополнительные параметры в алфавитном порядке
        if kwargs:
            sorted_params = sorted(kwargs.items())
            for key, value in sorted_params:
                data += f":{key}={value}"

        expected_signature = hashlib.md5(data.encode()).hexdigest()
        return expected_signature.upper() == signature.upper()

    async def check_payment_status(self, payment_id: str) -> Optional[str]:
        """
        Проверяет статус платежа в БД (Robokassa не предоставляет прямой API для проверки)

        Args:
            payment_id: ID платежа

        Returns:
            Статус платежа (pending, succeeded, canceled) или None
        """
        try:
            payment = await self.db.get_payment(payment_id)
            if payment:
                logger.info(f"Статус платежа {payment_id}: {payment.status}")
                return payment.status
            return None

        except Exception as e:
            logger.error(f"Ошибка проверки статуса платежа {payment_id}: {e}")
            return None

    async def handle_result_callback(self, result_data: dict) -> bool:
        """
        Обрабатывает Result URL от Robokassa (синхронный ответ пользователю)

        Args:
            result_data: Данные от Robokassa (OutSum, InvId, SignatureValue, и т.д.)

        Returns:
            True если обработка успешна, False иначе
        """
        try:
            out_sum = float(result_data.get('OutSum', 0))
            inv_id = int(result_data.get('InvId', 0))
            signature = result_data.get('SignatureValue', '')

            # Извлекаем дополнительные параметры
            shp_params = {}
            for key, value in result_data.items():
                if key.startswith('Shp_'):
                    shp_params[key] = value

            # Проверяем подпись
            if not self._verify_result_signature(out_sum, inv_id, signature, **shp_params):
                logger.error(f"Неверная подпись для платежа {inv_id}")
                return False

            # Получаем chat_id из дополнительных параметров
            chat_id = shp_params.get('Shp_chat_id')

            # Обновляем статус в БД
            await self.db.update_payment_status(str(inv_id), 'succeeded')

            # Добавляем бонусные запросы пользователю (50₽ = +3 запроса)
            if chat_id:
                amount_rubles = int(out_sum)
                await self.db.add_bonus_requests(chat_id, amount_rubles)
                logger.info(f"Добавлены бонусные запросы для чата {chat_id}, сумма {amount_rubles}₽")

            logger.info(f"Result callback обработан: платеж {inv_id}, статус succeeded")
            return True

        except Exception as e:
            logger.error(f"Ошибка обработки result callback: {e}")
            return False


# Функции-помощники для интеграции с ботом

async def create_donation_payment(
    payment_service: PaymentService,
    chat_id: str,
    user_id: str,
    amount: float,
    description: str = "Пожертвование"
) -> Optional[str]:
    """
    Создает платеж за пожертвование и возвращает URL для оплаты

    Args:
        payment_service: Сервис платежей
        chat_id: ID чата
        user_id: ID пользователя
        amount: Сумма пожертвования
        description: Описание платежа

    Returns:
        URL для оплаты или None
    """
    result = await payment_service.create_payment(
        chat_id=chat_id,
        user_id=user_id,
        amount=amount,
        description=description
    )

    if result:
        return result['confirmation_url']
    return None
