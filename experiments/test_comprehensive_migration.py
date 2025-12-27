#!/usr/bin/env python3
"""
Тест комплексной миграции базы данных
Проверяет, что migration правильно добавляет все недостающие колонки
"""
import asyncio
import os
import sys
import sqlite3
from datetime import datetime

# Добавляем родительскую директорию в путь для импорта модулей
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import Database

# Используем тестовую базу данных
TEST_DB = "test_comprehensive_migration.db"
DB_URL = f"sqlite+aiosqlite:///{TEST_DB}"


def create_old_schema():
    """Создает старую схему базы данных БЕЗ колонок last_request_date и bonus_requests"""
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)

    conn = sqlite3.connect(TEST_DB)
    cursor = conn.cursor()

    # Создаём таблицу chat_states со СТАРОЙ схемой (без last_request_date и bonus_requests)
    cursor.execute("""
        CREATE TABLE chat_states (
            id INTEGER PRIMARY KEY,
            chat_id VARCHAR NOT NULL UNIQUE,
            chat_type VARCHAR NOT NULL,
            is_active BOOLEAN DEFAULT 1,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Добавляем тестовые данные
    cursor.execute("""
        INSERT INTO chat_states (chat_id, chat_type, is_active)
        VALUES ('test_chat_1', 'private', 1)
    """)

    cursor.execute("""
        INSERT INTO chat_states (chat_id, chat_type, is_active)
        VALUES ('test_chat_2', 'group', 1)
    """)

    conn.commit()
    conn.close()
    print("✅ Создана старая схема базы данных без колонок last_request_date и bonus_requests")


def verify_columns():
    """Проверяет наличие всех колонок в таблице"""
    conn = sqlite3.connect(TEST_DB)
    cursor = conn.cursor()

    cursor.execute("PRAGMA table_info(chat_states)")
    columns = cursor.fetchall()
    column_names = [col[1] for col in columns]

    print(f"\n📋 Колонки в таблице chat_states: {column_names}")

    conn.close()
    return column_names


def verify_data():
    """Проверяет, что данные сохранились"""
    conn = sqlite3.connect(TEST_DB)
    cursor = conn.cursor()

    cursor.execute("SELECT chat_id, chat_type, bonus_requests FROM chat_states ORDER BY chat_id")
    rows = cursor.fetchall()

    print("\n📊 Данные в таблице:")
    for row in rows:
        print(f"  - chat_id: {row[0]}, chat_type: {row[1]}, bonus_requests: {row[2]}")

    conn.close()
    return rows


async def test_migration():
    """Основной тест миграции"""
    print("\n🧪 Тест комплексной миграции базы данных\n")

    # Шаг 1: Создаём старую схему
    print("1️⃣ Создание старой схемы...")
    create_old_schema()
    columns_before = verify_columns()

    assert 'last_request_date' not in columns_before, "❌ Колонка last_request_date уже существует!"
    assert 'bonus_requests' not in columns_before, "❌ Колонка bonus_requests уже существует!"
    print("✅ Старая схема создана корректно (без новых колонок)\n")

    # Шаг 2: Запускаем миграцию через init_db
    print("2️⃣ Запуск миграции...")
    db = Database(DB_URL)
    await db.init_db()
    print("✅ Миграция завершена\n")

    # Шаг 3: Проверяем, что колонки добавлены
    print("3️⃣ Проверка добавленных колонок...")
    columns_after = verify_columns()

    assert 'last_request_date' in columns_after, "❌ Колонка last_request_date не добавлена!"
    assert 'bonus_requests' in columns_after, "❌ Колонка bonus_requests не добавлена!"
    print("✅ Все недостающие колонки добавлены\n")

    # Шаг 4: Проверяем, что данные сохранились
    print("4️⃣ Проверка сохранности данных...")
    rows = verify_data()

    assert len(rows) == 2, "❌ Данные потеряны!"
    assert rows[0][0] == 'test_chat_1', "❌ Данные повреждены!"
    assert rows[1][0] == 'test_chat_2', "❌ Данные повреждены!"

    # Проверяем значения по умолчанию для bonus_requests
    for row in rows:
        # bonus_requests должен быть 3 (значение по умолчанию)
        assert row[2] == 3, f"❌ bonus_requests должен быть 3, получено: {row[2]}"

    print("✅ Все данные сохранены, значения по умолчанию установлены\n")

    # Шаг 5: Тестируем идемпотентность (повторный запуск миграции)
    print("5️⃣ Тест идемпотентности (повторный запуск)...")
    await db.init_db()
    columns_rerun = verify_columns()

    assert columns_after == columns_rerun, "❌ Повторный запуск миграции изменил схему!"
    print("✅ Миграция идемпотентна (безопасна для повторного запуска)\n")

    # Шаг 6: Проверяем работу с новыми колонками
    print("6️⃣ Проверка работы с новыми колонками...")
    chat_state = await db.get_or_create_chat_state("test_chat_1", "private")

    assert chat_state.chat_id == "test_chat_1", "❌ Неверный chat_id!"
    assert chat_state.bonus_requests == 3, f"❌ bonus_requests должен быть 3, получено: {chat_state.bonus_requests}"
    print(f"  chat_id: {chat_state.chat_id}")
    print(f"  bonus_requests: {chat_state.bonus_requests}")
    print(f"  last_request_date: {chat_state.last_request_date}")
    print("✅ Работа с новыми колонками корректна\n")

    # Закрываем соединение
    await db.close()

    print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
    print("\n🎉 Комплексная миграция работает корректно!")


async def main():
    try:
        await test_migration()
    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Удаляем тестовую базу данных
        if os.path.exists(TEST_DB):
            os.remove(TEST_DB)
            print(f"\n🧹 Тестовая база данных {TEST_DB} удалена")


if __name__ == "__main__":
    asyncio.run(main())
