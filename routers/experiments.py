import uuid
import logging
from typing import List, Optional, Union, Annotated
from datetime import datetime, timezone  # Ensure timezone is imported
import asyncio

from fastapi import APIRouter, HTTPException, Body, Depends, Request, status, Query  # Added Query
from pydantic import BaseModel, Field, ValidationError
from google.cloud import firestore
from google.cloud.firestore import AsyncClient
from google.cloud import storage

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig, SafetySetting, HarmCategory
from vertexai.preview.vision_models import ImageGenerationModel

from google.auth import exceptions as google_auth_exceptions

import AuthAndUser as auth
from services.ai_story_utils import (
    build_story_generation_prompt,
    generate_story_from_prompt,
    generate_and_upload_image,
    GeminiStructuredResponse
)

logger = logging.getLogger('uvicorn.error')

IMAGE_GEN_MODEL_NAME_CONFIG = "imagen-3.0-generate-002"
TEXT_GEN_MODEL_NAME_CONFIG = "gemini-2.0-flash"
GCS_BUCKET_NAME_CONFIG = "musings-mr.net"
GCP_PROJECT_ID = "clojure-gen-blog"
GCP_LOCATION = "us-central1"

_vertex_ai_initialized = False


def ensure_vertex_ai_initialized():
    global _vertex_ai_initialized
    if not _vertex_ai_initialized:
        try:
            logger.info(f"Initializing Vertex AI with Project ID: {GCP_PROJECT_ID} and Location: {GCP_LOCATION}")
            vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
            _vertex_ai_initialized = True
            logger.info("Vertex AI initialized successfully.")
        except google_auth_exceptions.DefaultCredentialsError as e:
            logger.error(
                f"Vertex AI DefaultCredentialsError: {e}. Ensure ADC is configured or service account is set up.")
            raise HTTPException(status_code=500, detail="Vertex AI authentication failed. Check server configuration.")
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI: {e}")
            raise HTTPException(status_code=500, detail="Vertex AI initialization error.")


async def get_firestore_client(request: Request) -> AsyncClient:
    if not hasattr(request.app.state, 'db') or not request.app.state.db:
        logger.error("Firestore client not initialized or unavailable.")
        raise HTTPException(status_code=503, detail="Database service unavailable")
    if not isinstance(request.app.state.db, AsyncClient):
        logger.error("Firestore client is not an AsyncClient.")
        raise HTTPException(status_code=503, detail="Database service misconfigured")
    return request.app.state.db


async def get_gcs_client(request: Request) -> storage.Client:
    if not hasattr(request.app.state, 'gcs_client') or not request.app.state.gcs_client:
        logger.error("GCS client not initialized or unavailable in app state.")
        raise HTTPException(status_code=503, detail="GCS service unavailable")
    return request.app.state.gcs_client


router = APIRouter(
    prefix="/experiments",
    tags=["experiments"],
    dependencies=[Depends(auth.get_current_active_user)],
)

CAMPFIRE_DAILY_LIMIT = 5
USERS_COLLECTION = "users"
USAGE_SUBCOLLECTION = "usage"
CAMPFIRES_SUBCOLLECTION = "campfires"
RATE_LIMIT_BYPASS_EMAIL = "mrkiouak@gmail.com"


class CampfireChatTurn(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender: str
    text: str
    imageUrl: Optional[str] = Field(default=None)
    promptForUser: Optional[str] = Field(default=None)


class CampfireGetResponse(BaseModel):
    storyContent: str
    chatTurns: List[CampfireChatTurn]
    hasExistingSession: bool  # MODIFIED: Renamed from hasActiveSessionToday


class CampfirePostRequest(BaseModel):
    previousContent: str
    inputText: str
    chatTurns: List[CampfireChatTurn] = Field(default_factory=list)


class CampfirePostResponse(BaseModel):
    storyContent: str
    chatTurns: List[CampfireChatTurn]


class CampfireDateListResponse(BaseModel):
    dates: List[str] = Field(
        description="A list of dates (YYYY-MM-DD) for which campfire stories exist, sorted descending.")


@router.get("/campfire/list", response_model=CampfireDateListResponse)
async def list_campfire_story_dates(
        current_user: Annotated[auth.User, Depends(auth.get_current_active_user)],
        db: AsyncClient = Depends(get_firestore_client)
):
    logger.info(f"User '{current_user.username}' requesting list of their campfire story dates.")
    story_dates = []
    try:
        campfires_collection_ref = db.collection(USERS_COLLECTION) \
            .document(current_user.username) \
            .collection(CAMPFIRES_SUBCOLLECTION)
        async for doc_snapshot in campfires_collection_ref.select([]).stream():
            story_dates.append(doc_snapshot.id)
        story_dates.sort(reverse=True)
        logger.info(f"Found {len(story_dates)} campfire story dates for user '{current_user.username}'.")
        return CampfireDateListResponse(dates=story_dates)
    except Exception as e:
        logger.exception(f"Error listing campfire story dates for user '{current_user.username}': {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve story dates list.")


@router.get("/campfire", response_model=CampfireGetResponse)
async def get_campfire_start(
        current_user: Annotated[auth.User, Depends(auth.get_current_active_user)],
        db: AsyncClient = Depends(get_firestore_client),
        date: Optional[str] = Query(None,
                                    description="Optional date in YYYY-MM-DD format to fetch a specific campfire story. Defaults to today's date.",
                                    alias="date_str")  # MODIFIED: Added query param
):
    logger.info(f"User '{current_user.username}' requesting campfire story. Provided date_str: '{date}'.")

    target_date_str: str
    if date:
        try:
            # Validate YYYY-MM-DD format. strptime raises ValueError if format doesn't match.
            datetime.strptime(date, "%Y-%m-%d")
            target_date_str = date
            logger.info(f"Using provided date: {target_date_str} for user '{current_user.username}'.")
        except ValueError:
            logger.warning(f"Invalid date format '{date}' provided by user '{current_user.username}'.")
            raise HTTPException(status_code=400, detail="Invalid date format. Please use YYYY-MM-DD.")
    else:
        # Default to today's date in UTC
        target_date_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        logger.info(f"Defaulting to today's date: {target_date_str} for user '{current_user.username}'.")

    campfire_log_ref = db.collection(USERS_COLLECTION) \
        .document(current_user.username) \
        .collection(CAMPFIRES_SUBCOLLECTION).document(target_date_str)

    has_existing_session = False
    initial_prompt_for_user = "How does the story begin?"
    initial_story_content = "The campfire crackles, waiting for a tale..."

    try:
        doc = await campfire_log_ref.get()
        if doc.exists:
            logger.info(f"Found existing campfire log for user '{current_user.username}' for date {target_date_str}.")
            saved_data = doc.to_dict()
            has_existing_session = True  # Record found for the target date

            parsed_chat_turns = []
            if "chatTurns" in saved_data and isinstance(saved_data["chatTurns"], list):
                for turn_data_dict in saved_data["chatTurns"]:
                    if not isinstance(turn_data_dict, dict): continue
                    turn_data_dict.setdefault('imageUrl', None)
                    turn_data_dict.setdefault('promptForUser', None)
                    try:
                        parsed_chat_turns.append(CampfireChatTurn(**turn_data_dict))
                    except ValidationError as ve:
                        logger.error(
                            f"Validation error for a GET chat turn (date: {target_date_str}): {ve}. Data: {turn_data_dict}")
                saved_data["chatTurns"] = parsed_chat_turns

            if not saved_data.get("chatTurns"):
                logger.warning(
                    f"Chat turns empty/malformed for user {current_user.username}, date {target_date_str}. Initializing.")
                first_turn = CampfireChatTurn(id=f'start_error_get_{target_date_str}', sender='Storyteller',
                                              text='The adventure begins (or had an issue loading)!', imageUrl=None,
                                              promptForUser=initial_prompt_for_user)
                saved_data["chatTurns"] = [first_turn]

            if "storyContent" not in saved_data or not saved_data["storyContent"]:
                saved_data["storyContent"] = initial_story_content
                if parsed_chat_turns:
                    story_parts = [turn.text for turn in parsed_chat_turns if turn.text]
                    if story_parts: saved_data["storyContent"] = "\n\n".join(story_parts)

            try:
                # Remove fields not in response model to prevent validation error from old data
                keys_to_pop = ["newImageUrl", "prompt", "savedAt", "hasActiveSessionToday"]
                for key_to_pop in keys_to_pop:
                    saved_data.pop(key_to_pop, None)

                # Ensure required fields for CampfireGetResponse are present
                final_story_content = saved_data.get("storyContent", initial_story_content)
                final_chat_turns = saved_data.get("chatTurns", [])
                if not final_chat_turns:  # If somehow still empty, provide a default
                    final_chat_turns = [CampfireChatTurn(id=f'default_final_{target_date_str}', sender='Storyteller',
                                                         text='Welcome to the campfire!', imageUrl=None,
                                                         promptForUser=initial_prompt_for_user)]
                    if final_story_content == initial_story_content:  # And if story content is also placeholder
                        final_story_content = 'Welcome to the campfire!'

                return CampfireGetResponse(
                    storyContent=final_story_content,
                    chatTurns=final_chat_turns,
                    hasExistingSession=has_existing_session  # This is True here
                )
            except ValidationError as e:
                logger.error(
                    f"Final validation error for CampfireGetResponse from saved data (date: {target_date_str}) for user {current_user.username}: {e}. Data before error: {saved_data}")
                has_existing_session = False  # Treat as if no existing session was successfully loaded
        else:
            logger.info(
                f"No existing campfire log for user '{current_user.username}' for date {target_date_str}. Returning default new story state.")
            # has_existing_session remains False (its initial value)

    except Exception as e:
        logger.exception(
            f"Error fetching/processing campfire log for user '{current_user.username}' for date {target_date_str}: {e}. Returning default new story state.")
        has_existing_session = False  # Ensure it's false on any exception during fetch/processing

    # This part is reached if doc doesn't exist OR if there was an error during fetch/parse
    # OR if validation of existing data failed and we decided to fall through.
    default_initial_storyteller_turn = CampfireChatTurn(
        id=f'start_new_{target_date_str}',
        sender='Storyteller',
        text='The adventure begins...',
        imageUrl=None,
        promptForUser=initial_prompt_for_user
    )
    return CampfireGetResponse(
        storyContent=initial_story_content,
        chatTurns=[default_initial_storyteller_turn],
        hasExistingSession=False  # Explicitly false for new/default state
    )


@firestore.async_transactional
async def check_and_update_usage(transaction, usage_ref, current_user: auth.User):
    usage_snapshot = await usage_ref.get(transaction=transaction)
    current_count = 0
    if usage_snapshot.exists: current_count = usage_snapshot.get("campfireCalls") or 0
    logger.debug(f"User '{current_user.username}' - Today's campfire usage count before update: {current_count}")
    apply_limit_check = current_user.email != RATE_LIMIT_BYPASS_EMAIL
    if current_user.email == RATE_LIMIT_BYPASS_EMAIL:
        logger.info(f"User '{current_user.username}' ({current_user.email}) bypassing rate limit check.")
    if apply_limit_check and current_count >= CAMPFIRE_DAILY_LIMIT:
        logger.warning(
            f"User '{current_user.username}' exceeded daily limit of {CAMPFIRE_DAILY_LIMIT} for /campfire POST.")
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                            detail=f"Daily limit of {CAMPFIRE_DAILY_LIMIT} campfire interactions exceeded. Please try again tomorrow.")
    transaction.set(usage_ref, {"campfireCalls": firestore.Increment(1), "lastUpdated": firestore.SERVER_TIMESTAMP},
                    merge=True)
    logger.debug(f"User '{current_user.username}' - Incrementing campfire usage count.")
    return current_count


@router.post("/campfire", response_model=CampfirePostResponse)
async def post_campfire_turn(
        current_user: Annotated[auth.User, Depends(auth.get_current_active_user)],
        payload: CampfirePostRequest = Body(...),
        db: AsyncClient = Depends(get_firestore_client),
        gcs_client: storage.Client = Depends(get_gcs_client)
):
    logger.info(f"User '{current_user.username}' ({current_user.email}) attempting POST /campfire")
    ensure_vertex_ai_initialized()

    # POST always operates on today's date for creating/updating story turns and usage
    today_utc_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')

    usage_doc_ref = db.collection(USERS_COLLECTION).document(current_user.username) \
        .collection(USAGE_SUBCOLLECTION).document(today_utc_str)  # Usage is always for today
    try:
        await check_and_update_usage(db.transaction(), usage_doc_ref, current_user)
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.exception(
            f"Firestore error during usage check/update for user '{current_user.username}' on {today_utc_str}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not verify usage limit.")

    try:
        gemini_text_model = GenerativeModel(TEXT_GEN_MODEL_NAME_CONFIG)
        image_gen_model = ImageGenerationModel.from_pretrained(IMAGE_GEN_MODEL_NAME_CONFIG)
    except Exception as e:
        logger.exception(f"Failed to initialize AI models for user '{current_user.username}': {e}")
        raise HTTPException(status_code=500, detail="AI model initialization failed.")

    safety_settings_text = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: SafetySetting.HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: SafetySetting.HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: SafetySetting.HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: SafetySetting.HarmBlockThreshold.BLOCK_NONE,
    }
    generation_config_text = GenerationConfig(
        temperature=0.8, top_p=0.95, max_output_tokens=1000, response_mime_type="application/json"
    )

    storyteller_texts = [turn.text for turn in payload.chatTurns if turn.sender == "Storyteller" and turn.text]
    storyteller_only_previous_text = "\n\n".join(storyteller_texts)

    prompt_for_story = build_story_generation_prompt(
        previous_story_text=storyteller_only_previous_text,
        user_idea=payload.inputText
    )

    gemini_story_data: GeminiStructuredResponse
    try:
        gemini_story_data = await generate_story_from_prompt(
            gemini_text_model=gemini_text_model, prompt=prompt_for_story,
            generation_config=generation_config_text, safety_settings=safety_settings_text
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception(f"Unexpected error from generate_story_from_prompt for user '{current_user.username}': {e}")
        raise HTTPException(status_code=500, detail="Failed to generate story elements.")

    story_parts_for_response = []
    if payload.previousContent and payload.previousContent != "The campfire crackles, waiting for a tale...":
        story_parts_for_response.append(payload.previousContent)

    story_parts_for_response.append(payload.inputText)
    story_parts_for_response.append(gemini_story_data.storyContinuation)
    full_story_content = "\n\n".join(filter(None, story_parts_for_response))

    if not full_story_content and \
            (
                    payload.previousContent == "The campfire crackles, waiting for a tale..." or not payload.previousContent) and \
            (payload.inputText or gemini_story_data.storyContinuation):
        full_story_content = "\n\n".join(filter(None, [payload.inputText, gemini_story_data.storyContinuation]))

    updated_chat_turns = list(payload.chatTurns)
    updated_chat_turns.append(
        CampfireChatTurn(sender='User', text=payload.inputText, imageUrl=None, promptForUser=None)
    )

    storyteller_turn_image_url: Optional[str] = None
    if gemini_story_data.storyContinuation:
        current_segment_for_image = gemini_story_data.storyContinuation
        previous_context_parts_for_image = []
        if payload.previousContent and payload.previousContent != "The campfire crackles, waiting for a tale...":
            previous_context_parts_for_image.append(payload.previousContent)
        if payload.inputText:
            previous_context_parts_for_image.append(payload.inputText)
        full_story_context_for_image = "\n\n".join(filter(None, previous_context_parts_for_image))
        if not full_story_context_for_image:
            full_story_context_for_image = "The story is just beginning."

        try:
            storyteller_turn_image_url = await generate_and_upload_image(
                image_gen_model=image_gen_model, gcs_client=gcs_client,
                gcs_bucket_name=GCS_BUCKET_NAME_CONFIG,
                current_story_segment=current_segment_for_image,
                full_story_context=full_story_context_for_image,
                username=current_user.username
            )
            if storyteller_turn_image_url:
                logger.info(
                    f"Successfully generated image URL: {storyteller_turn_image_url} for user '{current_user.username}'.")
            else:
                logger.warning(f"Image generation utility returned no URL for user '{current_user.username}'.")
        except Exception as img_gen_exc:
            logger.exception(
                f"Error calling image generation utility for user '{current_user.username}': {img_gen_exc}")

    updated_chat_turns.append(
        CampfireChatTurn(
            sender='Storyteller', text=gemini_story_data.storyContinuation,
            imageUrl=storyteller_turn_image_url, promptForUser=gemini_story_data.nextUserPrompt
        )
    )

    response_data = CampfirePostResponse(
        storyContent=full_story_content,
        chatTurns=updated_chat_turns
    )

    try:
        # POST always saves to today's date log
        campfire_log_ref_post = db.collection(USERS_COLLECTION).document(current_user.username) \
            .collection(CAMPFIRES_SUBCOLLECTION).document(today_utc_str)
        data_to_save = response_data.model_dump(mode='json')
        data_to_save['savedAt'] = firestore.SERVER_TIMESTAMP
        await campfire_log_ref_post.set(data_to_save)  # Use set to overwrite today's log with the latest state
        logger.info(f"Saved/Updated campfire response for user '{current_user.username}' for date {today_utc_str}.")
    except Exception as e:
        logger.exception(
            f"Failed to save campfire response for user '{current_user.username}' for date {today_utc_str}: {e}")

    return response_data