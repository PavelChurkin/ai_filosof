"""Quick test to verify database functionality"""
import asyncio
from database import Database


async def test_basic_operations():
    """Test basic database operations"""
    print("Initializing database...")
    db = Database("sqlite+aiosqlite:///:memory:")
    await db.init_db()
    print("✓ Database initialized")

    print("\nCreating chat state...")
    chat_state = await db.get_or_create_chat_state("test_chat_123", "private")
    print(f"✓ Chat state created: {chat_state.chat_id}, type: {chat_state.chat_type}")

    print("\nSaving thought...")
    thought = await db.save_thought(
        chat_id="test_chat_123",
        step1_words="test words",
        step1_image="test image",
        step2_question="test question?",
        step3_answer="test answer",
        is_published=False,
        was_paid=False
    )
    print(f"✓ Thought saved with ID: {thought.id}")

    print("\nRetrieving latest thought...")
    latest = await db.get_latest_thought("test_chat_123")
    print(f"✓ Latest thought retrieved: {latest.step3_answer}")

    print("\nCreating payment...")
    payment = await db.create_payment(
        chat_id="test_chat_123",
        user_id="user_456",
        payment_id="payment_789",
        amount=10000,
        payment_type="urgent_thought"
    )
    print(f"✓ Payment created with ID: {payment.payment_id}")

    print("\nGetting all active chats...")
    active_chats = await db.get_all_active_chats()
    print(f"✓ Found {len(active_chats)} active chat(s)")

    await db.close()
    print("\n✅ All database operations completed successfully!")


if __name__ == "__main__":
    asyncio.run(test_basic_operations())
