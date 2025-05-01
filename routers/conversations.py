
import logging
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timezone # Added timezone
from pydantic import BaseModel, Field
import uuid

from fastapi import APIRouter, HTTPException, Request, Depends, status
from google.cloud import firestore
from google.cloud.firestore import ArrayUnion, AsyncClient

import AuthAndUser as auth

logger = logging.getLogger('uvicorn.error')

async def get_firestore_client(request: Request) -> AsyncClient: # Ensure AsyncClient type hint
    if not hasattr(request.app.state, 'db') or not request.app.state.db:
        logger.error("Firestore client not initialized or unavailable.")
        raise HTTPException(status_code=503, detail="Database service unavailable")
    if not isinstance(request.app.state.db, AsyncClient):
         logger.error("Firestore client is not an AsyncClient.")
         raise HTTPException(status_code=503, detail="Database service misconfigured")
    return request.app.state.db


class ClientInfo(BaseModel):
    type: Optional[str] = None
    version: Optional[str] = None

class SessionMetadata(BaseModel):
    user_id: str
    start_time: datetime
    model_version: Optional[str] = None
    client_info: Optional[ClientInfo] = None
    extra: Dict[str, Any] = Field(default_factory=dict)

class PartMetadata(BaseModel):
    filename: Optional[str] = None
    duration_ms: Optional[int] = None
    processing_time_ms: Optional[int] = None
    confidence: Optional[float] = None
    extra: Dict[str, Any] = Field(default_factory=dict)

class Part(BaseModel):
    type: str
    content: Union[str, dict, list, Any]
    metadata: Optional[Dict[str, Any]] = None

class Turn(BaseModel):
    # Allow turn_id to be optional in input, generate if missing
    turn_id: str = Field(default_factory=lambda: f"turn-{uuid.uuid4().hex}")
    # Set default timestamp to now in UTC
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    role: str # "user" or "model"
    parts: List[Part]

class Conversation(BaseModel):
    conversation_id: str
    session_metadata: SessionMetadata
    turns: List[Turn]

    model_config = {  # Pydantic v2 style config
        "json_schema_extra": {
            "example": {
                "conversation_id": "unique-conversation-identifier-12345",
                "session_metadata": {
                    "user_id": "user-abc",
                    "start_time": "2025-04-30T21:05:00Z",
                    "model_version": "gemini-1.5-pro",
                    "client_info": {
                        "type": "web",
                        "version": "1.2.0"
                    }
                },
                "turns": [
                    {
                        "turn_id": "turn-001",
                        "timestamp": "2025-04-30T21:05:10Z",
                        "role": "user",
                        "parts": [
                            {"type": "text/plain", "content": "Describe this image."},
                            {"type": "image/jpeg", "content": "gs://bucket/img.jpg",
                             "metadata": {"filename": "photo.jpg"}}
                        ]
                    },
                    {
                        "turn_id": "turn-002",
                        "timestamp": "2025-04-30T21:05:15Z",
                        "role": "model",
                        "parts": [
                            {"type": "text/plain", "content": "It shows a cat."},
                            {"type": "application/json", "content": {"details": "fluffy"},
                             "metadata": {"processing_time_ms": 500}}
                        ]
                    }
                ]
            }
        }
    }

# --- API Router ---

router = APIRouter(
    prefix="/conversations",
    tags=["conversations"],
)

CONVERSATIONS_COLLECTION = "conversations"

@router.post("/", response_model=Conversation, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    conversation: Conversation,
    db: AsyncClient = Depends(get_firestore_client),
    current_user: auth.User = Depends(auth.get_current_active_user)
):
    """
    Creates a new conversation record or overwrites an existing one
    with the same conversation_id.
    """
    if conversation.session_metadata.user_id != current_user.username:
       logger.warning(f"User '{current_user.username}' attempting to save conversation for different user '{conversation.session_metadata.user_id}'.")
       raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User ID in session metadata does not match authenticated user.")

    try:
        conv_ref = db.collection(CONVERSATIONS_COLLECTION).document(conversation.conversation_id)
        conv_data = conversation.model_dump(mode='json')
        await conv_ref.set(conv_data)
        logger.info(f"User '{current_user.username}' saved conversation '{conversation.conversation_id}'")
        return conversation
    except Exception as e:
        logger.exception(f"Error saving conversation '{conversation.conversation_id}' for user '{current_user.username}': {e}")
        raise HTTPException(status_code=500, detail="Internal server error while saving conversation")

@router.get("/{conversation_id}", response_model=Conversation)
async def get_conversation(
    conversation_id: str,
    db: AsyncClient = Depends(get_firestore_client),
    current_user: auth.User = Depends(auth.get_current_active_user)
):
    """
    Retrieves a specific conversation by its ID.
    """
    try:
        conv_ref = db.collection(CONVERSATIONS_COLLECTION).document(conversation_id)
        conv_doc = await conv_ref.get()

        if not conv_doc.exists:
            logger.warning(f"Conversation '{conversation_id}' not found attempted access by user '{current_user.username}'.")
            raise HTTPException(status_code=404, detail=f"Conversation with id '{conversation_id}' not found")

        conv_data = conv_doc.to_dict()

        fetched_user_id = conv_data.get("session_metadata", {}).get("user_id")
        if fetched_user_id != current_user.username:
             logger.warning(f"Forbidden: User '{current_user.username}' attempted to access conversation '{conversation_id}' belonging to user '{fetched_user_id}'.")
             raise HTTPException(status_code=403, detail="Forbidden: You do not have access to this conversation.")

        try:
            return Conversation(**conv_data)
        except Exception as validation_error:
             logger.error(f"Data validation error for conversation {conversation_id}: {validation_error}. Data: {conv_data}")
             raise HTTPException(status_code=500, detail="Error processing conversation data from database")

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.exception(f"Error retrieving conversation '{conversation_id}' for user '{current_user.username}': {e}")
        raise HTTPException(status_code=500, detail="Internal server error while fetching conversation")


# --- Endpoint to Add a Turn ---
@router.post("/{conversation_id}/turns", response_model=Turn, status_code=status.HTTP_201_CREATED)
async def add_turn_to_conversation(
    conversation_id: str,
    turn_payload: Turn, # Expect a Turn object in the request body
    db: AsyncClient = Depends(get_firestore_client),
    current_user: auth.User = Depends(auth.get_current_active_user)
):
    """
    Adds a new turn to an existing conversation identified by conversation_id.
    Validates ownership before adding. Returns the added turn.
    """
    conv_ref = db.collection(CONVERSATIONS_COLLECTION).document(conversation_id)

    try:
        conv_doc = await conv_ref.get()
        if not conv_doc.exists:
            logger.warning(f"Add turn attempt: Conversation '{conversation_id}' not found for user '{current_user.username}'.")
            raise HTTPException(status_code=404, detail=f"Conversation with id '{conversation_id}' not found")

        conv_data = conv_doc.to_dict()
        fetched_user_id = conv_data.get("session_metadata", {}).get("user_id")
        if fetched_user_id != current_user.username:
            logger.warning(f"Forbidden: User '{current_user.username}' attempted to add turn to conversation '{conversation_id}' belonging to user '{fetched_user_id}'.")
            raise HTTPException(status_code=403, detail="Forbidden: You do not have access to add turns to this conversation.")

        # 2. Prepare the new turn data
        # Create a new Turn instance ensuring defaults are applied if needed
        # Override timestamp to be the time of addition (UTC)
        # Use provided turn_id or generate one
        new_turn = Turn(
            turn_id=turn_payload.turn_id or f"turn-{uuid.uuid4().hex}", # Use provided or generate
            timestamp=datetime.now(timezone.utc), # Set timestamp on arrival
            role=turn_payload.role,
            parts=turn_payload.parts # Assumes parts are valid per Part model
        )

        # 3. Update Firestore using ArrayUnion
        turn_dict = new_turn.model_dump(mode='json') # Serialize the validated Turn model
        await conv_ref.update({
            "turns": ArrayUnion([turn_dict])
        })

        logger.info(f"User '{current_user.username}' added turn '{new_turn.turn_id}' to conversation '{conversation_id}'")

        # 4. Return the newly added turn
        return new_turn

    except HTTPException as http_exc:
        # Re-raise HTTP exceptions (like 404, 403)
        raise http_exc
    except Exception as e:
        logger.exception(f"Error adding turn to conversation '{conversation_id}' for user '{current_user.username}': {e}")
        raise HTTPException(status_code=500, detail="Internal server error while adding turn")