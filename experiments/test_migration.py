#!/usr/bin/env python3
"""
Test database migration for last_request_date column
"""
import asyncio
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import Database, ChatState
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.pool import StaticPool

async def test_migration():
    """Test migration with both fresh and existing database"""

    # Test 1: Create old database without last_request_date column
    print("=== Test 1: Creating old database schema ===")
    old_db_path = "test_old.db"
    if os.path.exists(old_db_path):
        os.remove(old_db_path)

    # Create sync engine to manually create old schema
    engine = create_engine(f'sqlite:///{old_db_path}', poolclass=StaticPool)

    # Create old schema without last_request_date
    with engine.connect() as conn:
        conn.execute(text('''
            CREATE TABLE chat_states (
                id INTEGER PRIMARY KEY,
                chat_id VARCHAR NOT NULL UNIQUE,
                chat_type VARCHAR NOT NULL,
                is_active BOOLEAN DEFAULT 1,
                bonus_requests INTEGER DEFAULT 3,
                created_at DATETIME,
                updated_at DATETIME
            )
        '''))
        conn.commit()

        # Insert test data
        conn.execute(text('''
            INSERT INTO chat_states (chat_id, chat_type, is_active, bonus_requests, created_at, updated_at)
            VALUES ('-5050956685', 'group', 1, 5, datetime('now'), datetime('now'))
        '''))
        conn.commit()

    engine.dispose()
    print("Old database created with test data")

    # Check columns before migration
    inspector = inspect(create_engine(f'sqlite:///{old_db_path}'))
    columns_before = [col['name'] for col in inspector.get_columns('chat_states')]
    print(f"Columns before migration: {columns_before}")
    assert 'last_request_date' not in columns_before, "Column should not exist yet"

    # Test 2: Run migration
    print("\n=== Test 2: Running migration ===")
    db = Database(f"sqlite+aiosqlite:///{old_db_path}")
    await db.init_db()

    # Check columns after migration
    inspector = inspect(create_engine(f'sqlite:///{old_db_path}'))
    columns_after = [col['name'] for col in inspector.get_columns('chat_states')]
    print(f"Columns after migration: {columns_after}")
    assert 'last_request_date' in columns_after, "Column should exist after migration"

    # Test 3: Verify data integrity
    print("\n=== Test 3: Verifying data integrity ===")
    chat_state = await db.get_or_create_chat_state('-5050956685', 'group')
    print(f"Chat state retrieved: chat_id={chat_state.chat_id}, bonus_requests={chat_state.bonus_requests}")
    assert chat_state.chat_id == '-5050956685', "Chat ID should match"
    assert chat_state.bonus_requests == 5, "Bonus requests should be preserved"
    print("Data integrity verified")

    # Test 4: Run migration again (should be idempotent)
    print("\n=== Test 4: Testing idempotency ===")
    await db.init_db()
    columns_after_second = [col['name'] for col in inspector.get_columns('chat_states')]
    print(f"Columns after second migration: {columns_after_second}")
    assert columns_after == columns_after_second, "Columns should be the same"
    print("Migration is idempotent")

    await db.close()

    # Clean up
    if os.path.exists(old_db_path):
        os.remove(old_db_path)

    print("\n✅ All migration tests passed!")

if __name__ == '__main__':
    asyncio.run(test_migration())
