"""
Test to verify if database file is created properly
"""
import sys
import os
import asyncio
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import Database
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_db_creation():
    """Test database file creation"""

    # Test 1: Check current directory
    print("🧪 Test 1: Checking current working directory")
    cwd = os.getcwd()
    print(f"   Current directory: {cwd}")

    # Test 2: Create database with relative path
    print("\n🧪 Test 2: Creating database with default URL")
    db = Database("sqlite+aiosqlite:///test_filosof.db")
    print(f"   Database URL: {db.database_url}")

    # Test 3: Initialize database
    print("\n🧪 Test 3: Initializing database...")
    await db.init_db()
    print("   ✅ Database initialized")

    # Test 4: Check if file exists
    print("\n🧪 Test 4: Checking if database file was created")
    db_file_path = Path(cwd) / "test_filosof.db"
    if db_file_path.exists():
        print(f"   ✅ Database file exists at: {db_file_path}")
        print(f"   File size: {db_file_path.stat().st_size} bytes")
    else:
        print(f"   ❌ Database file NOT found at: {db_file_path}")

        # Search for the file in other locations
        print("\n   🔍 Searching for database files...")
        for root, dirs, files in os.walk(cwd):
            for file in files:
                if file.endswith('.db'):
                    print(f"   Found: {os.path.join(root, file)}")

    # Test 5: Try creating a chat state
    print("\n🧪 Test 5: Testing database operations...")
    try:
        chat_state = await db.get_or_create_chat_state("test_chat_123", "private")
        print(f"   ✅ Created chat state: {chat_state.chat_id}")
    except Exception as e:
        print(f"   ❌ Error creating chat state: {e}")

    # Test 6: Check file again after operations
    print("\n🧪 Test 6: Final file check...")
    if db_file_path.exists():
        print(f"   ✅ Database file confirmed at: {db_file_path}")
        print(f"   File size: {db_file_path.stat().st_size} bytes")

        # List all files in directory to confirm visibility
        print("\n   📁 All .db files in current directory:")
        for item in Path(cwd).glob("*.db"):
            print(f"   - {item.name} ({item.stat().st_size} bytes)")
    else:
        print(f"   ❌ Database file still NOT found")

        # Check with SQLite directly
        print("\n   🔍 Testing with direct SQLite3...")
        import sqlite3
        try:
            conn = sqlite3.connect("test_direct.db")
            conn.execute("CREATE TABLE test (id INTEGER)")
            conn.close()

            direct_path = Path(cwd) / "test_direct.db"
            if direct_path.exists():
                print(f"   ✅ Direct SQLite file created successfully at: {direct_path}")
            else:
                print(f"   ❌ Even direct SQLite failed to create visible file")
        except Exception as e:
            print(f"   ❌ Direct SQLite error: {e}")

    # Cleanup
    await db.close()

    print("\n" + "="*60)
    print("✅ TEST COMPLETED")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(test_db_creation())
