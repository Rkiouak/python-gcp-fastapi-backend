import uuid
import logging
from typing import List, Optional, Union, Annotated
from datetime import datetime, timezone
import json
import asyncio # Added for running synchronous IO in threads

from fastapi import APIRouter, HTTPException, Body, Depends, Request, status
from pydantic import BaseModel, Field, ValidationError
from google.cloud import firestore
from google.cloud.firestore import AsyncClient
from google.cloud import storage # Added for GCS client type hint

import vertexai
# Updated to include ImageGenerationModel and Image (if needed for other operations, though not directly used in this snippet for image bytes)
from vertexai.generative_models import GenerativeModel, Part as GeminiPart, GenerationConfig, SafetySetting, HarmCategory # Renamed Part to GeminiPart to avoid conflict if vertexai.vision_models.Image also has Part
from vertexai.preview.vision_models import ImageGenerationModel # For Imagen
# If you use vertexai.vision_models.Image, you might need:
# from vertexai.vision_models import Image as VisionImage

from google.auth import exceptions as google_auth_exceptions

import AuthAndUser as auth

logger = logging.getLogger('uvicorn.error')

# ... (rest of your existing constants like GCP_PROJECT_ID, GCP_LOCATION, GEMINI_MODEL_NAME)
IMAGE_GEN_MODEL_NAME = "imagen-3.0-generate-002" # Model for image generation
GCS_BUCKET_NAME = "musings-mr.net" # Define GCS bucket name

GCP_PROJECT_ID = "clojure-gen-blog"
GCP_LOCATION = "us-central1"
GEMINI_MODEL_NAME = "gemini-2.0-flash"

_vertex_ai_initialized = False

def ensure_vertex_ai_initialized():
    """Initializes Vertex AI if not already done."""
    global _vertex_ai_initialized
    if not _vertex_ai_initialized:
        try:
            logger.info(f"Initializing Vertex AI with Project ID: {GCP_PROJECT_ID} and Location: {GCP_LOCATION}")
            vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
            _vertex_ai_initialized = True
            logger.info("Vertex AI initialized successfully.")
        except google_auth_exceptions.DefaultCredentialsError as e:
            logger.error(f"Vertex AI DefaultCredentialsError: {e}. Ensure ADC is configured or service account is set up.")
            raise HTTPException(status_code=500, detail="Vertex AI authentication failed. Check server configuration.")
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI: {e}")
            raise HTTPException(status_code=500, detail="Vertex AI initialization error.")


async def get_firestore_client(request: Request) -> AsyncClient:
    """Dependency function to get the Firestore AsyncClient."""
    if not hasattr(request.app.state, 'db') or not request.app.state.db:
        logger.error("Firestore client not initialized or unavailable.")
        raise HTTPException(status_code=503, detail="Database service unavailable")
    if not isinstance(request.app.state.db, AsyncClient):
         logger.error("Firestore client is not an AsyncClient.")
         raise HTTPException(status_code=503, detail="Database service misconfigured")
    return request.app.state.db

router = APIRouter(
    prefix="/experiments",
    tags=["experiments"],
    dependencies=[Depends(auth.get_current_active_user)],
)

# --- Constants ---
CAMPFIRE_DAILY_LIMIT = 5
USERS_COLLECTION = "users"
USAGE_SUBCOLLECTION = "usage"
CAMPFIRES_SUBCOLLECTION = "campfires"
RATE_LIMIT_BYPASS_EMAIL = "mrkiouak@gmail.com"

# --- Pydantic Models ---
class CampfireChatTurn(BaseModel):
    """Represents a single turn in the campfire chat."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender: str
    text: str

class GeminiStructuredResponse(BaseModel):
    """Expected structure for the JSON response from Gemini."""
    storyContinuation: str
    nextUserPrompt: str

class CampfireGetResponse(BaseModel):
    """Response model for the initial GET request."""
    storyContent: str
    prompt: str
    chatTurns: List[CampfireChatTurn]
    newImageUrl: Optional[str] = Field(default=None, description="URL of an image relevant to the initial story state, if available.")
    hasActiveSessionToday: bool # New field

class CampfirePostRequest(BaseModel):
    """Request model for posting a new turn."""
    previousContent: str # The cumulative story text so far
    prompt: str # The prompt the user was responding to
    inputText: str
    # Client should send the current chat history to be appended to
    chatTurns: List[CampfireChatTurn] = Field(default_factory=list)

class CampfirePostResponse(BaseModel):
    """Response model after processing a user's turn."""
    storyContent: str # The updated cumulative story text
    prompt: str # The next prompt for the user
    chatTurns: List[CampfireChatTurn] # The updated list of chat turns
    newImageUrl: Optional[str] = Field(default=None, description="URL of a newly generated image relevant to the story turn, if available.")


async def generate_image_for_story(
        prompt_text: str,
        request: Request,
        current_user: auth.User
) -> Optional[str]:
    """
    Generates an image based on the prompt_text using Vertex AI's Imagen model
    and uploads it to Google Cloud Storage.
    Returns the public URL of the image, or None if an error occurs.
    """
    ensure_vertex_ai_initialized()
    logger.info(
        f"User '{current_user.username}' attempting to generate image for story with prompt: '{prompt_text[:100]}...'")

    try:
        image_model = ImageGenerationModel.from_pretrained(IMAGE_GEN_MODEL_NAME)

        # generate_images is synchronous, run in a thread pool
        generated_image_response = await asyncio.to_thread(
            image_model.generate_images,
            prompt=prompt_text,
            number_of_images=1,
            # You can add other parameters like aspect_ratio="1:1", safety_settings, etc.
            # E.g., aspect_ratio="16:9"
        )

        if not generated_image_response or not generated_image_response.images:
            logger.warning(f"Image generation failed or returned no images for user '{current_user.username}'.")
            return None

        image_obj = generated_image_response.images[0]
        # Check if _image_bytes attribute exists and is populated
        if hasattr(image_obj, '_image_bytes') and image_obj._image_bytes:
            image_bytes = image_obj._image_bytes
        else:
            # Fallback: save to a temporary in-memory buffer or temp file if _image_bytes is not available
            logger.info("'_image_bytes' not directly available. Saving image to temporary file to get bytes.")
            # This part would require careful handling of temp files or buffers.
            # For this example, let's log a warning and return if bytes aren't easily accessible.
            # A more robust solution would implement the save-to-temp-file-and-read pattern here.
            temp_filename_for_bytes = f"/tmp/temp_image_{uuid.uuid4()}.png"  # Ensure /tmp is writable
            try:
                image_obj.save(location=temp_filename_for_bytes, include_watermark=False)  # Or True if desired
                with open(temp_filename_for_bytes, "rb") as f:
                    image_bytes = f.read()
                import os
                os.remove(temp_filename_for_bytes)
                logger.info(f"Successfully read image bytes from temporary file for user '{current_user.username}'.")
            except Exception as e_save:
                logger.error(
                    f"Failed to save image to temp file or read bytes for user '{current_user.username}': {e_save}")
                return None

        if not image_bytes:
            logger.warning(f"Image bytes are empty after generation for user '{current_user.username}'.")
            return None

        gcs_client: storage.Client = request.app.state.gcs_client
        if not gcs_client:
            logger.error(f"GCS client not available in app state for user '{current_user.username}'.")
            return None

        image_gcs_filename = f"{uuid.uuid4()}.png"  # Assuming PNG format
        blob_name = f"campfire_images/{current_user.username}/{image_gcs_filename}"
        bucket = gcs_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(blob_name)

        # Uploading to GCS (synchronous call, run in thread)
        await asyncio.to_thread(
            blob.upload_from_string,
            image_bytes,
            content_type="image/png"  # Adjust if the format is different
        )

        logger.info(f"Campfire image '{blob_name}' uploaded to GCS for user '{current_user.username}'.")
        return blob.public_url

    except google_auth_exceptions.DefaultCredentialsError as e:
        logger.error(f"Vertex AI Image Gen DefaultCredentialsError for user '{current_user.username}': {e}.")
        return None
    except Exception as e:
        logger.exception(f"Error generating or uploading image for story for user '{current_user.username}': {e}")
        return None

# --- Gemini Helper Function ---
async def generate_story_elements_with_gemini(
    previous_story_text: str,
    user_idea: str
) -> GeminiStructuredResponse:
    """
    Generates story continuation and next user prompt using Gemini, expecting JSON output.
    """
    ensure_vertex_ai_initialized() # Ensure Vertex AI is ready

    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: SafetySetting.HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: SafetySetting.HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: SafetySetting.HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: SafetySetting.HarmBlockThreshold.BLOCK_NONE,
    }
    # Configure generation parameters
    generation_config = GenerationConfig(
        temperature=0.8, # Controls randomness (creativity)
        top_p=0.95,      # Controls nucleus sampling
        max_output_tokens=1000, # Max length of response
        response_mime_type="application/json" # Explicitly request JSON
    )

    # Initialize the Gemini model
    model = GenerativeModel(GEMINI_MODEL_NAME)

    # Construct the prompt with few-shot examples and JSON instructions
    prompt = f"""You are a creative storyteller helping to write a collaborative children's fairy tale or an elementary school-aged myth.
Your task is to continue the story based on the "Previous Story" and the "User's Idea".
You must also suggest a "nextUserPrompt" to ask the user what should happen next.
Your response MUST be a valid JSON object with two keys: "storyContinuation" and "nextUserPrompt".
The "storyContinuation" should be a few engaging sentences (2-4 sentences) in a style suitable for young children, continuing the narrative.
The "nextUserPrompt" should be a question to the user, guiding them to provide the next story beat.

Example 1:
Previous Story:
The little squirrel, Squeaky, had found a shiny, mysterious nut. It was glowing with a faint blue light!
User's Idea:
Squeaky should try to open the nut.
Your JSON Response:
```json
{{
  "storyContinuation": "Squeaky tapped the glowing nut gently with a tiny pebble. Crack! A tiny wisp of sparkling dust puffed out, and the nut opened to reveal a miniature map, showing a path to the legendary Whispering Waterfall! \\"Wow!\\" chirped Squeaky, his eyes wide with wonder.",
  "nextUserPrompt": "What does Squeaky decide to do with the map to the Whispering Waterfall?"
}}
```

Example 2:
Previous Story:
A brave knight named Sir Reginald was riding through a dark forest. He heard a strange sound.
User's Idea:
Sir Reginald should investigate the sound, which is a dragon snoring.
Your JSON Response:
```json
{{
  "storyContinuation": "Sir Reginald, always brave, tiptoed towards the rumbling sound. Peeking behind a giant oak tree, he saw a HUGE green dragon, fast asleep and snoring so loudly that the leaves on the trees trembled! Sir Reginald gulped, then decided this dragon looked more sleepy than scary.",
  "nextUserPrompt": "Does Sir Reginald try to talk to the sleepy dragon, or sneak past it?"
}}
```

--- Current Story ---
Previous Story:
{previous_story_text}

User's Idea:
{user_idea}

Your JSON Response:
"""
    try:
        logger.debug(f"Sending prompt to Gemini for JSON response: {prompt[:500]}...")
        # Generate content asynchronously
        response = await model.generate_content_async(
            prompt,
            generation_config=generation_config,
            safety_settings=safety_settings
        )

        # Process the response
        if response.candidates and response.candidates[0].content.parts:
            json_text = response.candidates[0].content.parts[0].text
            logger.info("Received JSON response from Gemini.")
            logger.debug(f"Gemini JSON text: {json_text}")
            try:
                # Clean potential markdown code block formatting
                if json_text.strip().startswith("```json"):
                    json_text = json_text.strip()[7:]
                    if json_text.endswith("```"):
                        json_text = json_text[:-3]

                # Parse and validate the JSON response using the Pydantic model
                data = json.loads(json_text)
                return GeminiStructuredResponse(**data)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from Gemini: {e}. Response text: {json_text}")
                raise HTTPException(status_code=500, detail="Story generator returned an invalid format.")
            except ValidationError as e: # Catch Pydantic validation errors
                logger.error(f"Failed to validate Gemini response structure: {e}. Data: {json_text}")
                raise HTTPException(status_code=500, detail="Story generator response structure mismatch.")
        else:
            # Handle cases where the response is empty or blocked
            logger.warning("Gemini response was empty or malformed for structured JSON.")
            if response.candidates and response.candidates[0].finish_reason:
                 logger.warning(f"Gemini finish reason: {response.candidates[0].finish_reason.name}")
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                logger.warning(f"Gemini prompt blocked. Reason: {response.prompt_feedback.block_reason.name}")
                raise HTTPException(status_code=400, detail=f"Story generation blocked by content policy: {response.prompt_feedback.block_reason.name}")
            raise HTTPException(status_code=500, detail="Story generator failed to provide a response.")

    except HTTPException as http_exc: # Re-raise specific HTTP exceptions
        raise http_exc
    except Exception as e:
        logger.exception(f"Error calling Gemini API for structured response: {e}")
        raise HTTPException(status_code=500, detail="Failed to communicate with the story generator.")

# --- Campfire Endpoint ---
@router.get("/campfire", response_model=CampfireGetResponse)
async def get_campfire_start(
    current_user: Annotated[auth.User, Depends(auth.get_current_active_user)],
    db: AsyncClient = Depends(get_firestore_client) # Added Firestore client dependency
):
    """
    Provides the initial state for the campfire story.
    If a story for today already exists for the user, it returns that story and sets hasActiveSessionToday to True.
    Otherwise, returns default state with hasActiveSessionToday set to False.
    Requires authentication.
    """
    logger.info(f"User '{current_user.username}' accessed GET /campfire")

    today_utc_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    # Path to the user's campfire log for today
    campfire_log_ref = db.collection(USERS_COLLECTION).document(current_user.username) \
                         .collection(CAMPFIRES_SUBCOLLECTION).document(today_utc_str)

    has_active_session = False # Initialize flag

    try:
        # Attempt to get today's campfire log
        doc = await campfire_log_ref.get()
        if doc.exists:
            logger.info(f"Found existing campfire log for user '{current_user.username}' for date {today_utc_str}.")
            saved_data = doc.to_dict()
            has_active_session = True # Set flag to True as session exists
            try:
                # Spread saved_data and add/overwrite hasActiveSessionToday
                return CampfireGetResponse(**saved_data, hasActiveSessionToday=has_active_session)
            except ValidationError as e:
                logger.error(f"Validation error when creating CampfireGetResponse from saved data for user {current_user.username}: {e}. Data: {saved_data}")
                # Fallback to default if saved data is malformed, but still indicate no *valid* active session was loaded
                has_active_session = False # Reset if validation fails for loaded data
        else:
            logger.info(f"No existing campfire log for user '{current_user.username}' for date {today_utc_str}. Returning default.")
            # has_active_session remains False
    except Exception as e:
        logger.exception(f"Error fetching campfire log for user '{current_user.username}' for date {today_utc_str}: {e}. Returning default.")
        # has_active_session remains False

    # Return the default starting point of the story if no log found or error occurred
    return CampfireGetResponse(
        storyContent="The campfire crackles, waiting for a tale...",
        prompt="How does the story begin?",
        chatTurns=[
            CampfireChatTurn(id='start', sender='Storyteller', text='The adventure begins...')
        ],
        newImageUrl="https://storage.googleapis.com/example-bucket/campfire_start.jpg", # Placeholder image
        hasActiveSessionToday=has_active_session # Use the determined flag
    )

@firestore.async_transactional
async def check_and_update_usage(transaction, usage_ref, current_user: auth.User):
    """
    Firestore transaction to check usage limit (with bypass) and update the count.
    """
    usage_snapshot = await usage_ref.get(transaction=transaction)
    current_count = 0
    if usage_snapshot.exists:
        current_count = usage_snapshot.get("campfireCalls") or 0

    logger.debug(f"User '{current_user.username}' - Today's campfire usage count before update: {current_count}")

    # Check if rate limit should be applied (bypass for specific email)
    apply_limit_check = True
    if current_user.email == RATE_LIMIT_BYPASS_EMAIL:
        apply_limit_check = False
        logger.info(f"User '{current_user.username}' ({current_user.email}) bypassing rate limit check.")

    # If limit applies and is exceeded, raise 429 error
    if apply_limit_check and current_count >= CAMPFIRE_DAILY_LIMIT:
        logger.warning(f"User '{current_user.username}' exceeded daily limit of {CAMPFIRE_DAILY_LIMIT} for /campfire POST.")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Daily limit of {CAMPFIRE_DAILY_LIMIT} campfire interactions exceeded. Please try again tomorrow."
        )

    # Increment usage count atomically for all users (even bypassed ones)
    transaction.set(usage_ref, {
        "campfireCalls": firestore.Increment(1),
        "lastUpdated": firestore.SERVER_TIMESTAMP # Record update time
    }, merge=True) # Use merge=True to create or update
    logger.debug(f"User '{current_user.username}' - Incrementing campfire usage count.")
    return current_count # Return count *before* increment

@router.post("/campfire", response_model=CampfirePostResponse)
async def post_campfire_turn(
    current_user: Annotated[auth.User, Depends(auth.get_current_active_user)],
    payload: CampfirePostRequest = Body(...),
    request: Request = Request,  # Added Request to access app.state.gcs_client
    db: AsyncClient = Depends(get_firestore_client)
):
    """
    Processes a user's turn in the story, generates a new image based on the story,
    saves the response, and returns the next state.
    Requires authentication and enforces a daily usage limit (with bypass).
    Manages chat turn history by appending new turns.
    """
    logger.info(f"User '{current_user.username}' ({current_user.email}) attempting POST /campfire")

    today_utc_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    usage_doc_ref = db.collection(USERS_COLLECTION).document(current_user.username) \
                      .collection(USAGE_SUBCOLLECTION).document(today_utc_str)
    try:
        await check_and_update_usage(db.transaction(), usage_doc_ref, current_user) # check_and_update_usage is defined elsewhere in your file
        logger.info(f"User '{current_user.username}' usage count updated for {today_utc_str}.")
    except HTTPException as http_exc:
        raise http_exc # Re-raise HTTP exceptions like 429
    except Exception as e:
        logger.exception(f"Firestore error during usage check/update for user '{current_user.username}' on {today_utc_str}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not verify usage limit.")

    logger.debug(f"User '{current_user.username}' passed rate limit check. Generating story elements with Gemini.")

    gemini_response: GeminiStructuredResponse
    try:
        # First call to Gemini for story text and next prompt
        gemini_response = await generate_story_elements_with_gemini( # generate_story_elements_with_gemini is defined elsewhere
            payload.previousContent,
            payload.inputText
        )
    except HTTPException as e: # Re-raise specific HTTP exceptions from Gemini helper
        raise e
    except Exception as e: # Catch any other unexpected errors
        logger.exception(f"Unexpected error during Gemini story text generation for user '{current_user.username}': {e}")
        raise HTTPException(status_code=500, detail="Failed to generate story elements.")

    full_story_content = f"{payload.previousContent}\n\n{gemini_response.storyContinuation}"
    next_user_prompt = gemini_response.nextUserPrompt

    updated_chat_turns = list(payload.chatTurns)
    updated_chat_turns.append(
        CampfireChatTurn(sender='User', text=payload.inputText)
    )
    updated_chat_turns.append(
        CampfireChatTurn(sender='Storyteller', text=gemini_response.storyContinuation)
    )

    # --- Second call to Vertex AI to generate an image ---
    new_image_url: Optional[str] = None
    if gemini_response.storyContinuation: # Only generate image if there's new story content
        try:
            logger.info(f"Requesting image generation for story continuation for user '{current_user.username}'.")
            new_image_url = await generate_image_for_story(
                prompt_text=gemini_response.storyContinuation,
                request=request, # Pass the FastAPI request object
                current_user=current_user
            )
            if new_image_url:
                logger.info(f"Successfully generated image URL: {new_image_url} for user '{current_user.username}'.")
            else:
                logger.warning(f"Image generation did not return a URL for user '{current_user.username}'.")
        except Exception as img_gen_exc:
            logger.exception(f"Error during image generation or upload for user '{current_user.username}': {img_gen_exc}")
            # Decide if this should be a fatal error or if the flow can continue without an image.
            # For now, it continues, and new_image_url will remain None.

    response_data = CampfirePostResponse(
        storyContent=full_story_content,
        prompt=next_user_prompt,
        chatTurns=updated_chat_turns,
        newImageUrl=new_image_url  # Use the dynamically generated image URL
    )

    # Save the complete response (including the new image URL) to Firestore
    try:
        campfire_log_ref = db.collection(USERS_COLLECTION).document(current_user.username) \
                             .collection(CAMPFIRES_SUBCOLLECTION).document(today_utc_str)
        data_to_save = response_data.model_dump(mode='json') # Get dict from Pydantic model
        data_to_save['savedAt'] = firestore.SERVER_TIMESTAMP # Add a server timestamp
        await campfire_log_ref.set(data_to_save)
        logger.info(f"Saved campfire response (with image URL if any) for user '{current_user.username}' for date {today_utc_str}.")
    except Exception as e:
        logger.exception(f"Failed to save campfire response for user '{current_user.username}' for date {today_utc_str}: {e}")
        # Not raising an HTTP error here, as the core operation (story gen) succeeded.

    return response_data

