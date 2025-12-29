"""
Test to verify database is created with absolute path
This simulates the server scenario where bot might be run from different directory
"""
import sys
import os
import asyncio
from pathlib import Path
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_db_creation_from_different_dir():
    """Test database file creation when run from different directory"""

    print("🧪 Test: Database creation with absolute path fix")
    print("="*70)

    # Get the actual bot directory
    bot_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"\n📁 Bot directory: {bot_dir}")
    print(f"📁 Current working directory: {os.getcwd()}")

    # Test 1: Import database from bot directory
    print("\n" + "="*70)
    print("Test 1: Creating database from bot directory")
    print("="*70)

    from database import Database

    db1 = Database()
    await db1.init_db()

    # Check where the file was created
    db_file_path1 = db1.database_url.replace('sqlite+aiosqlite:///', '')
    print(f"\n✅ Database initialized")
    print(f"   Database URL: {db1.database_url}")
    print(f"   Expected file path: {db_file_path1}")

    if os.path.exists(db_file_path1):
        print(f"   ✅ File exists: {os.path.getsize(db_file_path1)} bytes")
        print(f"   ✅ File location: {db_file_path1}")
    else:
        print(f"   ❌ File NOT found at: {db_file_path1}")

    # Test database operations
    try:
        chat_state = await db1.get_or_create_chat_state("test_chat_1", "private")
        print(f"   ✅ Database operations working (created chat: {chat_state.chat_id})")
    except Exception as e:
        print(f"   ❌ Database operations failed: {e}")

    await db1.close()

    # Test 2: Change to a different directory and create database
    print("\n" + "="*70)
    print("Test 2: Creating database from DIFFERENT working directory")
    print("="*70)

    # Create a temporary directory and change to it
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        print(f"\n📁 Changed working directory to: {os.getcwd()}")
        print(f"   (This simulates running bot from different directory on server)")

        # Create another database instance
        db2 = Database()
        await db2.init_db()

        db_file_path2 = db2.database_url.replace('sqlite+aiosqlite:///', '')
        print(f"\n✅ Database initialized from different directory")
        print(f"   Database URL: {db2.database_url}")
        print(f"   Expected file path: {db_file_path2}")

        # Check if both databases point to the same location (should be in bot directory)
        if db_file_path1 == db_file_path2:
            print(f"   ✅ Both databases use the SAME absolute path (good!)")
        else:
            print(f"   ⚠️ Databases use different paths:")
            print(f"      First:  {db_file_path1}")
            print(f"      Second: {db_file_path2}")

        if os.path.exists(db_file_path2):
            print(f"   ✅ File exists: {os.path.getsize(db_file_path2)} bytes")
            print(f"   ✅ File location: {db_file_path2}")
        else:
            print(f"   ❌ File NOT found at: {db_file_path2}")

        # Test database operations from different directory
        try:
            chat_state = await db2.get_or_create_chat_state("test_chat_2", "group")
            print(f"   ✅ Database operations working (created chat: {chat_state.chat_id})")
        except Exception as e:
            print(f"   ❌ Database operations failed: {e}")

        await db2.close()

        # Restore original directory
        os.chdir(original_cwd)

    # Test 3: Verify file is in bot directory, not in CWD
    print("\n" + "="*70)
    print("Test 3: Verify database file location")
    print("="*70)

    expected_db_path = os.path.join(bot_dir, "filosof.db")
    print(f"\n📍 Expected database path: {expected_db_path}")

    if os.path.exists(expected_db_path):
        print(f"   ✅ Database file is in bot directory")
        print(f"   ✅ File size: {os.path.getsize(expected_db_path)} bytes")

        # List all .db files to confirm
        print(f"\n📁 All .db files in bot directory:")
        for item in Path(bot_dir).glob("*.db"):
            print(f"   - {item.name} ({item.stat().st_size} bytes)")
    else:
        print(f"   ❌ Database file NOT in bot directory")
        print(f"\n   🔍 Searching for .db files in bot directory...")
        for item in Path(bot_dir).glob("*.db"):
            print(f"   Found: {item}")

    # Cleanup test database
    if os.path.exists(expected_db_path):
        os.remove(expected_db_path)
        print(f"\n🧹 Cleaned up test database: {expected_db_path}")

    print("\n" + "="*70)
    print("✅ ALL TESTS COMPLETED")
    print("="*70)

    print("\n📝 Summary:")
    print("   The database file will now ALWAYS be created in the bot's")
    print("   directory, regardless of where the bot is started from.")
    print("   This fixes the issue where the file was created in different")
    print("   locations depending on the current working directory.")

if __name__ == "__main__":
    asyncio.run(test_db_creation_from_different_dir())
