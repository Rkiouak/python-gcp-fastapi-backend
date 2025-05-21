import uuid
import logging
from typing import List, Optional, Union, Annotated, Set  # Added Set
from datetime import datetime, timezone
import asyncio

from fastapi import APIRouter, HTTPException, Body, Depends, Request, status, Query
from pydantic import BaseModel, Field, ValidationError
from google.cloud import firestore
from google.cloud.firestore import AsyncClient, FieldFilter  # Added FieldFilter
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
TEXT_GEN_MODEL_NAME_CONFIG = "gemini-2.5-flash-preview-05-20"  # Or your preferred/available model
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

CAMPFIRE_DAILY_LIMIT = 50
MAX_CAMPFIRE_STORIES = 25  # Maximum number of stories a user can save
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
    storyTitle: Optional[str] = None
    hasExistingSession: bool


class CampfirePostRequest(BaseModel):
    previousContent: str
    inputText: str
    chatTurns: List[CampfireChatTurn] = Field(default_factory=list)
    storyTitle: Optional[str] = None  # User-provided title for the story


class CampfirePostResponse(BaseModel):
    storyContent: str
    chatTurns: List[CampfireChatTurn]
    storyTitle: str  # The title under which the story was saved/updated


class CampfireStoryListResponse(BaseModel):
    titles: List[str] = Field(
        description="A list of unique story titles for which campfire stories exist, sorted alphabetically.")


@router.get("/campfire/list", response_model=CampfireStoryListResponse)
async def list_campfire_story_titles(
        current_user: Annotated[auth.User, Depends(auth.get_current_active_user)],
        db: AsyncClient = Depends(get_firestore_client)
):
    logger.info(f"User '{current_user.username}' requesting list of their campfire story titles.")
    story_titles_set: Set[str] = set()
    try:
        campfires_collection_ref = db.collection(USERS_COLLECTION) \
            .document(current_user.username) \
            .collection(CAMPFIRES_SUBCOLLECTION)

        # Fetch only the 'storyTitle' field if possible, or stream docs and extract
        # Using .select(["storyTitle"]) might be more efficient if supported and only title is needed.
        # For simplicity here, we stream the document and extract.
        async for doc_snapshot in campfires_collection_ref.stream():
            data = doc_snapshot.to_dict()
            if data and "storyTitle" in data and data["storyTitle"]:
                story_titles_set.add(data["storyTitle"])

        sorted_titles = sorted(list(story_titles_set))
        logger.info(f"Found {len(sorted_titles)} unique campfire story titles for user '{current_user.username}'.")
        return CampfireStoryListResponse(titles=sorted_titles)
    except Exception as e:
        logger.exception(f"Error listing campfire story titles for user '{current_user.username}': {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve story titles list.")


@router.get("/campfire", response_model=CampfireGetResponse)
async def get_campfire_start(
        current_user: Annotated[auth.User, Depends(auth.get_current_active_user)],
        db: AsyncClient = Depends(get_firestore_client),
        storyTitle: Optional[str] = Query(None,
                                          description="Optional title of the campfire story to fetch.",
                                          alias="title")
):
    logger.info(f"User '{current_user.username}' requesting campfire story. Provided title: '{storyTitle}'.")

    initial_prompt_for_user = "How does the story begin?"
    initial_story_content = "The campfire crackles, waiting for a tale..."
    has_existing_session = False
    loaded_story_title = storyTitle  # Will be updated if a story is loaded

    if storyTitle:
        campfires_collection_ref = db.collection(USERS_COLLECTION) \
            .document(current_user.username) \
            .collection(CAMPFIRES_SUBCOLLECTION)

        query = campfires_collection_ref.where(filter=FieldFilter("storyTitle", "==", storyTitle)).limit(1)
        docs_stream = query.stream()
        doc_list = [doc async for doc in docs_stream]

        if doc_list:
            doc = doc_list[0]
            logger.info(
                f"Found existing campfire log for user '{current_user.username}' with storyTitle '{storyTitle}'. Doc ID: {doc.id}")
            saved_data = doc.to_dict()
            has_existing_session = True

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
                            f"Validation error for a GET chat turn (storyTitle: {storyTitle}): {ve}. Data: {turn_data_dict}")
                saved_data["chatTurns"] = parsed_chat_turns

            if not saved_data.get("chatTurns"):
                first_turn = CampfireChatTurn(id=str(uuid.uuid4()), sender='Storyteller',
                                              text='The adventure loaded (or had an issue with turns)!', imageUrl=None,
                                              promptForUser=initial_prompt_for_user)
                saved_data["chatTurns"] = [first_turn]

            if "storyContent" not in saved_data or not saved_data["storyContent"]:
                saved_data["storyContent"] = initial_story_content
                if parsed_chat_turns:
                    story_parts = [turn.text for turn in parsed_chat_turns if turn.text]
                    if story_parts: saved_data["storyContent"] = "\n\n".join(story_parts)

            loaded_story_title = saved_data.get("storyTitle", storyTitle)  # Use title from doc

            return CampfireGetResponse(
                storyContent=saved_data.get("storyContent", initial_story_content),
                chatTurns=saved_data.get("chatTurns", []),
                storyTitle=loaded_story_title,
                hasExistingSession=has_existing_session
            )
        else:
            logger.info(
                f"No campfire log found for user '{current_user.username}' with storyTitle '{storyTitle}'. Returning default new story state.")
    else:
        logger.info(f"No storyTitle provided by user '{current_user.username}'. Returning default new story state.")

    default_initial_storyteller_turn = CampfireChatTurn(
        id=str(uuid.uuid4()),
        sender='Storyteller',
        text='The adventure begins...',
        imageUrl=None,
        promptForUser=initial_prompt_for_user
    )
    return CampfireGetResponse(
        storyContent=initial_story_content,
        chatTurns=[default_initial_storyteller_turn],
        storyTitle=None,
        hasExistingSession=False
    )


@firestore.async_transactional
async def check_and_update_usage(transaction, usage_ref, current_user: auth.User):
    usage_snapshot = await usage_ref.get(transaction=transaction)
    current_count = 0
    if usage_snapshot.exists: current_count = usage_snapshot.get("campfireCalls") or 0
    logger.debug(f"User '{current_user.username}' - Today's campfire usage count before update: {current_count}")

    apply_daily_limit_check = current_user.email != RATE_LIMIT_BYPASS_EMAIL
    if current_user.email == RATE_LIMIT_BYPASS_EMAIL:
        logger.info(f"User '{current_user.username}' ({current_user.email}) bypassing daily API call limit check.")

    if apply_daily_limit_check and current_count >= CAMPFIRE_DAILY_LIMIT:
        logger.warning(
            f"User '{current_user.username}' exceeded daily API call limit of {CAMPFIRE_DAILY_LIMIT} for /campfire POST.")
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                            detail=f"Daily limit of {CAMPFIRE_DAILY_LIMIT} campfire interactions exceeded. Please try again tomorrow.")

    transaction.set(usage_ref, {"campfireCalls": firestore.Increment(1), "lastUpdated": firestore.SERVER_TIMESTAMP},
                    merge=True)
    logger.debug(f"User '{current_user.username}' - Incrementing daily campfire API call usage count.")
    return current_count


@router.post("/campfire", response_model=CampfirePostResponse)
async def post_campfire_turn(
        current_user: Annotated[auth.User, Depends(auth.get_current_active_user)],
        payload: CampfirePostRequest = Body(...),
        db: AsyncClient = Depends(get_firestore_client),
        gcs_client: storage.Client = Depends(get_gcs_client)
):
    logger.info(
        f"User '{current_user.username}' ({current_user.email}) attempting POST /campfire with payload storyTitle: '{payload.storyTitle}'")

    story_title_to_save: str
    today_utc_str_for_default_title = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    if payload.storyTitle and payload.storyTitle.strip():
        story_title_to_save = payload.storyTitle.strip()
    else:
        story_title_to_save = f"Untitled Story - {today_utc_str_for_default_title}"
    logger.info(f"User '{current_user.username}' - Effective story_title_to_save: '{story_title_to_save}'")

    campfires_collection_ref = db.collection(USERS_COLLECTION).document(current_user.username) \
        .collection(CAMPFIRES_SUBCOLLECTION)

    query_existing_by_title = campfires_collection_ref.where(
        filter=FieldFilter("storyTitle", "==", story_title_to_save)).limit(1)
    existing_docs_stream = query_existing_by_title.stream()
    existing_doc_list = [doc async for doc in existing_docs_stream]

    if not existing_doc_list:  # This implies a new story creation attempt
        if current_user.email != RATE_LIMIT_BYPASS_EMAIL:
            count_query = campfires_collection_ref.count()
            aggregation_query_result = await count_query.get()
            current_story_count = 0
            if aggregation_query_result and aggregation_query_result[0] and hasattr(aggregation_query_result[0][0],
                                                                                    'value'):
                current_story_count = aggregation_query_result[0][0].value
            else:
                logger.warning(
                    f"Could not retrieve accurate story count for user {current_user.username}. Check Firestore logs/permissions.")

            logger.info(
                f"User '{current_user.username}' has {current_story_count} existing campfire stories. Limit is {MAX_CAMPFIRE_STORIES}.")

            if current_story_count >= MAX_CAMPFIRE_STORIES:
                logger.warning(
                    f"User '{current_user.username}' has reached the maximum story limit of {MAX_CAMPFIRE_STORIES}.")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"You have reached the maximum limit of {MAX_CAMPFIRE_STORIES} stories. Please delete an existing story to create a new one."
                )
        else:
            logger.info(
                f"User '{current_user.username}' ({current_user.email}) bypassing total story limit check for new story creation.")

    ensure_vertex_ai_initialized()  # Initialize AI after passing story limit checks for new stories

    today_utc_str_for_usage = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    usage_doc_ref = db.collection(USERS_COLLECTION).document(current_user.username) \
        .collection(USAGE_SUBCOLLECTION).document(today_utc_str_for_usage)
    try:
        await check_and_update_usage(db.transaction(), usage_doc_ref, current_user)
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.exception(
            f"Firestore error during daily usage check/update for user '{current_user.username}' on {today_utc_str_for_usage}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Could not verify daily usage limit.")

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
        previous_story_text=storyteller_only_previous_text, user_idea=payload.inputText
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
        if not full_story_context_for_image: full_story_context_for_image = "The story is just beginning."
        try:
            storyteller_turn_image_url = await generate_and_upload_image(
                image_gen_model=image_gen_model, gcs_client=gcs_client,
                gcs_bucket_name=GCS_BUCKET_NAME_CONFIG,
                current_story_segment=current_segment_for_image,
                full_story_context=full_story_context_for_image,
                username=current_user.username
            )
        except Exception as img_gen_exc:
            logger.exception(f"Error in image gen for user '{current_user.username}': {img_gen_exc}")

    updated_chat_turns.append(
        CampfireChatTurn(
            sender='Storyteller', text=gemini_story_data.storyContinuation,
            imageUrl=storyteller_turn_image_url, promptForUser=gemini_story_data.nextUserPrompt
        )
    )

    response_data_model = CampfirePostResponse(
        storyContent=full_story_content,
        chatTurns=updated_chat_turns,
        storyTitle=story_title_to_save
    )

    campfire_doc_ref = None
    existing_doc_id = None
    created_at_timestamp = None

    if existing_doc_list:
        existing_doc = existing_doc_list[0]
        existing_doc_id = existing_doc.id
        campfire_doc_ref = campfires_collection_ref.document(existing_doc_id)
        logger.info(f"Updating existing story with title '{story_title_to_save}', doc ID: {existing_doc_id}.")
        existing_data = existing_doc.to_dict()
        if existing_data and 'createdAt' in existing_data:
            created_at_timestamp = existing_data['createdAt']
        else:  # If createdAt missing in an old doc, set it now.
            created_at_timestamp = firestore.SERVER_TIMESTAMP
    else:
        new_doc_id = str(uuid.uuid4())
        campfire_doc_ref = campfires_collection_ref.document(new_doc_id)
        created_at_timestamp = firestore.SERVER_TIMESTAMP
        logger.info(f"Creating new story with title '{story_title_to_save}', new doc ID: {new_doc_id}.")

    try:
        data_to_save = response_data_model.model_dump(mode='json')
        data_to_save['storyTitle'] = story_title_to_save  # Ensure this is part of the saved data
        data_to_save['savedAt'] = firestore.SERVER_TIMESTAMP
        data_to_save['createdAt'] = created_at_timestamp

        await campfire_doc_ref.set(data_to_save, merge=True if existing_doc_id else False)
        logger.info(
            f"Saved/Updated campfire response for user '{current_user.username}' with storyTitle '{story_title_to_save}' (Doc ID: {campfire_doc_ref.id}).")
    except Exception as e:
        logger.exception(
            f"Failed to save campfire response for user '{current_user.username}' with storyTitle '{story_title_to_save}': {e}")

    return response_data_model