"""
Unit tests for database module
"""
import pytest
import asyncio
from datetime import datetime
from database import Database, ChatState, Thought, Payment


@pytest.fixture
async def test_db():
    """Create a test database instance"""
    db = Database("sqlite+aiosqlite:///:memory:")
    await db.init_db()
    yield db
    await db.close()


@pytest.mark.asyncio
async def test_create_chat_state(test_db):
    """Test creating a new chat state"""
    chat_id = "test_chat_123"
    chat_state = await test_db.get_or_create_chat_state(chat_id, "private")

    assert chat_state is not None
    assert chat_state.chat_id == chat_id
    assert chat_state.chat_type == "private"
    assert chat_state.is_active is True


@pytest.mark.asyncio
async def test_get_existing_chat_state(test_db):
    """Test getting an existing chat state"""
    chat_id = "test_chat_456"

    # Create first
    chat_state1 = await test_db.get_or_create_chat_state(chat_id, "group")

    # Get existing
    chat_state2 = await test_db.get_or_create_chat_state(chat_id, "group")

    assert chat_state1.id == chat_state2.id
    assert chat_state2.chat_id == chat_id


@pytest.mark.asyncio
async def test_update_chat_state(test_db):
    """Test updating chat state"""
    chat_id = "test_chat_789"
    chat_state = await test_db.get_or_create_chat_state(chat_id, "private")

    next_publish = datetime.utcnow()
    await test_db.update_chat_state(
        chat_id,
        next_publish_time=next_publish,
        is_active=False
    )

    # Verify update
    updated_state = await test_db.get_or_create_chat_state(chat_id, "private")
    assert updated_state.is_active is False


@pytest.mark.asyncio
async def test_save_thought(test_db):
    """Test saving a thought"""
    chat_id = "test_chat_101"

    thought = await test_db.save_thought(
        chat_id=chat_id,
        step1_words="test words",
        step1_image="test image",
        step2_question="test question",
        step3_answer="test answer",
        is_published=False,
        was_paid=False
    )

    assert thought is not None
    assert thought.chat_id == chat_id
    assert thought.step1_words == "test words"
    assert thought.step3_answer == "test answer"
    assert thought.is_published is False


@pytest.mark.asyncio
async def test_get_latest_thought(test_db):
    """Test getting the latest thought"""
    chat_id = "test_chat_202"

    # Save first thought
    thought1 = await test_db.save_thought(
        chat_id=chat_id,
        step3_answer="first answer"
    )

    # Save second thought
    await asyncio.sleep(0.01)  # Ensure different timestamps
    thought2 = await test_db.save_thought(
        chat_id=chat_id,
        step3_answer="second answer"
    )

    # Get latest
    latest = await test_db.get_latest_thought(chat_id)

    assert latest is not None
    assert latest.id == thought2.id
    assert latest.step3_answer == "second answer"


@pytest.mark.asyncio
async def test_update_thought(test_db):
    """Test updating a thought"""
    chat_id = "test_chat_303"

    thought = await test_db.save_thought(
        chat_id=chat_id,
        step3_answer="original answer",
        is_published=False
    )

    # Update
    await test_db.update_thought(
        thought.id,
        is_published=True,
        published_at=datetime.utcnow()
    )

    # Verify
    latest = await test_db.get_latest_thought(chat_id)
    assert latest.is_published is True
    assert latest.published_at is not None


@pytest.mark.asyncio
async def test_create_payment(test_db):
    """Test creating a payment record"""
    chat_id = "test_chat_404"
    user_id = "test_user_505"
    payment_id = "test_payment_606"

    payment = await test_db.create_payment(
        chat_id=chat_id,
        user_id=user_id,
        payment_id=payment_id,
        amount=10000,  # 100 rubles in kopecks
        payment_type="urgent_thought",
        metadata={"test": "data"}
    )

    assert payment is not None
    assert payment.chat_id == chat_id
    assert payment.user_id == user_id
    assert payment.payment_id == payment_id
    assert payment.amount == 10000
    assert payment.status == "pending"


@pytest.mark.asyncio
async def test_get_all_active_chats(test_db):
    """Test getting all active chats"""
    # Create multiple chats
    await test_db.get_or_create_chat_state("chat1", "private")
    await test_db.get_or_create_chat_state("chat2", "group")
    await test_db.get_or_create_chat_state("chat3", "private")

    # Deactivate one
    await test_db.update_chat_state("chat2", is_active=False)

    # Get active chats
    active_chats = await test_db.get_all_active_chats()

    assert len(active_chats) == 2
    assert all(chat.is_active for chat in active_chats)
    assert "chat2" not in [chat.chat_id for chat in active_chats]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
