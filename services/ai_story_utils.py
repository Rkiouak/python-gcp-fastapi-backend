import logging
import json
import uuid
import asyncio
from typing import Optional, Dict

from pydantic import BaseModel, ValidationError
from fastapi import HTTPException

from google.cloud import storage
from vertexai.generative_models import GenerativeModel, GenerationConfig, SafetySetting, HarmCategory, GenerationResponse
from vertexai.preview.vision_models import ImageGenerationModel
from google.auth import exceptions as google_auth_exceptions


logger = logging.getLogger('uvicorn.error')

# --- Pydantic Models ---
class GeminiStructuredResponse(BaseModel):
    storyContinuation: str
    nextUserPrompt: str


# --- Prompt Building ---
def build_story_generation_prompt(previous_story_text: str, user_idea: str) -> str:
    start_json_block = chr(96) * 3 + "json"
    end_json_block = chr(96) * 3
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
{start_json_block}
{{
  "storyContinuation": "Squeaky tapped the glowing nut gently with a tiny pebble. Crack! A tiny wisp of sparkling dust puffed out, and the nut opened to reveal a miniature map, showing a path to the legendary Whispering Waterfall! \\"Wow!\\" chirped Squeaky, his eyes wide with wonder.",
  "nextUserPrompt": "What does Squeaky decide to do with the map to the Whispering Waterfall?"
}}
{end_json_block}

Example 2:
Previous Story:
A brave knight named Sir Reginald was riding through a dark forest. He heard a strange sound.
User's Idea:
Sir Reginald should investigate the sound, which is a dragon snoring.
Your JSON Response:
{start_json_block}
{{
  "storyContinuation": "Sir Reginald, always brave, tiptoed towards the rumbling sound. Peeking behind a giant oak tree, he saw a HUGE green dragon, fast asleep and snoring so loudly that the leaves on the trees trembled! Sir Reginald gulped, then decided this dragon looked more sleepy than scary.",
  "nextUserPrompt": "Does Sir Reginald try to talk to the sleepy dragon, or sneak past it?"
}}
{end_json_block}

--- Current Story ---
Previous Story:
{previous_story_text}

User's Idea:
{user_idea}

Your JSON Response:
"""
    return prompt


# --- Gemini Text Generation ---
async def _parse_gemini_story_response(
    response: GenerationResponse
) -> GeminiStructuredResponse:
    if response.candidates and response.candidates[0].content.parts:
        json_text = response.candidates[0].content.parts[0].text
        logger.info("Received JSON response from Gemini for story generation.")
        logger.debug(f"Gemini JSON text (story generation): {json_text}")
        try:
            if json_text.strip().startswith("```json"):
                json_text = json_text.strip()[len("```json"):]
                if json_text.startswith("\n"): json_text = json_text[1:]
                if json_text.endswith("```"): json_text = json_text[:-len("```")]
            elif json_text.strip().startswith("```"):
                json_text = json_text.strip()[len("```"):]
                if json_text.startswith("\n"): json_text = json_text[1:]
                if json_text.endswith("```"): json_text = json_text[:-len("```")]
            json_text = json_text.strip()
            data = json.loads(json_text)
            return GeminiStructuredResponse(**data)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from Gemini (story generation): {e}. Response text: '{json_text}'")
            raise HTTPException(status_code=500, detail="Story generator returned an invalid format.")
        except ValidationError as e:
            logger.error(f"Failed to validate Gemini response structure (story generation): {e}. Data: '{json_text}'")
            raise HTTPException(status_code=500, detail="Story generator response structure mismatch.")
    else:
        logger.warning("Gemini response (story generation) was empty or malformed.")
        if response.candidates and response.candidates[0].finish_reason:
            logger.warning(f"Gemini finish reason (story generation): {response.candidates[0].finish_reason.name}")
        if response.prompt_feedback and response.prompt_feedback.block_reason:
            logger.warning(f"Gemini prompt (story generation) blocked. Reason: {response.prompt_feedback.block_reason.name}")
            raise HTTPException(status_code=400, detail=f"Story generation blocked by content policy: {response.prompt_feedback.block_reason.name}")
        raise HTTPException(status_code=500, detail="Story generator failed to provide a response.")

async def generate_story_from_prompt(
    gemini_text_model: GenerativeModel,
    prompt: str,
    generation_config: GenerationConfig,
    safety_settings: Dict[HarmCategory, SafetySetting.HarmBlockThreshold]
) -> GeminiStructuredResponse:
    try:
        logger.debug(f"Sending prompt to Gemini for story elements (via utility): {prompt[:500]}...")
        response = await gemini_text_model.generate_content_async(
            prompt,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        return await _parse_gemini_story_response(response)
    except HTTPException as http_exc:
        raise http_exc
    except google_auth_exceptions.GoogleAuthError as e:
        logger.error(f"Google Auth Error during Gemini text generation: {e}")
        raise HTTPException(status_code=500, detail="Authentication error with AI service.")
    except Exception as e:
        logger.exception(f"Error calling Gemini API for story elements (via utility): {e}")
        raise HTTPException(status_code=500, detail="Failed to communicate with the story generator or process its response.")


# --- Image Generation and Upload ---
async def generate_and_upload_image(
    image_gen_model: ImageGenerationModel,
    gcs_client: storage.Client,
    gcs_bucket_name: str,
    current_story_segment: str, # MODIFIED: Renamed from image_prompt
    full_story_context: str,    # MODIFIED: Added new parameter
    username: str
) -> Optional[str]:
    """
    Generates an image for the current_story_segment, using full_story_context for consistency,
    and uploads it to Google Cloud Storage.
    Returns the public URL of the image, or None if an error occurs.
    """

    # MODIFIED: Create a new detailed prompt for Imagen
    imagen_prompt = (
        f"Illustrate the following scene from a children's fairy tale or myth: \"{current_story_segment}\"\n\n"
        f"CONTEXT from the story so far: \"{full_story_context}\"\n\n"
        f"The image should be in a whimsical, vibrant, and engaging style suitable for a children's storybook. "
        f"Focus on the characters, actions, and atmosphere described in the LATEST scene. "
        f"Ensure visual consistency with the described context if possible (e.g., character appearance, setting details mentioned previously)."
        f"If specific characters were described (e.g. 'a squirrel named Squeaky with a blue-striped tail'), try to incorporate those details."
    )
    # Limit prompt length if necessary, though Imagen handles reasonably long prompts.
    # A very long full_story_context might be truncated by the model or API.
    # Consider summarizing full_story_context if it becomes excessively long.
    max_prompt_length = 8000 # Example limit, check Imagen documentation for actual limits
    if len(imagen_prompt) > max_prompt_length:
        # Basic truncation strategy, could be more sophisticated
        truncate_by = len(imagen_prompt) - max_prompt_length
        # Prioritize truncating the context rather than the current segment
        if len(full_story_context) > truncate_by + 50: # Ensure some context remains
             truncated_context = full_story_context[:-(truncate_by + 3)] + "..." # +3 for "..."
             imagen_prompt = (
                f"Illustrate the following scene from a children's fairy tale or myth: \"{current_story_segment}\"\n\n"
                f"CONTEXT from the story so far (truncated): \"{truncated_context}\"\n\n"
                f"The image should be in a whimsical, vibrant, and engaging style suitable for a children's storybook. "
                f"Focus on the characters, actions, and atmosphere described in the LATEST scene. "
                f"Ensure visual consistency with the described context if possible."
             )
        else: # If context is too short to truncate meaningfully, truncate the whole prompt
            imagen_prompt = imagen_prompt[:max_prompt_length-3] + "..."
        logger.warning(f"Imagen prompt was truncated to {len(imagen_prompt)} characters for user '{username}'.")


    logger.info(
        f"User '{username}' attempting to generate image (via utility) with effective prompt for segment: '{current_story_segment[:100]}...'")
    logger.debug(f"Full Imagen prompt for user '{username}': {imagen_prompt[:500]}...")


    try:
        generated_image_response = await asyncio.to_thread(
            image_gen_model.generate_images,
            prompt=imagen_prompt, # MODIFIED: Use the new detailed prompt
            number_of_images=1,
            # aspect_ratio="16:9", # Example: common aspect ratio
            # negative_prompt="text, words, letters, watermark, signature", # Example
        )

        if not generated_image_response or not generated_image_response.images:
            logger.warning(f"Image generation failed or returned no images for user '{username}' with segment '{current_story_segment[:50]}...'.")
            return None

        image_obj = generated_image_response.images[0]
        image_bytes = None
        if hasattr(image_obj, '_image_bytes') and image_obj._image_bytes:
            image_bytes = image_obj._image_bytes
        else:
            logger.info("'_image_bytes' not directly available for image gen. Saving image to temporary file to get bytes.")
            temp_filename_for_bytes = f"/tmp/temp_image_gen_util_{uuid.uuid4()}.png"
            try:
                await asyncio.to_thread(image_obj.save, location=temp_filename_for_bytes, include_watermark=False)
                with open(temp_filename_for_bytes, "rb") as f:
                    image_bytes = f.read()
                await asyncio.to_thread(os.remove, temp_filename_for_bytes)
                logger.info(f"Successfully read image bytes from temporary file for image gen for user '{username}'.")
            except Exception as e_save:
                logger.error(
                    f"Failed to save image gen to temp file or read bytes for user '{username}': {e_save}")
                return None

        if not image_bytes:
            logger.warning(f"Image bytes are empty after image generation for user '{username}'.")
            return None

        if not gcs_client:
            logger.error(f"GCS client not available for image upload for user '{username}'.")
            return None # Should not happen if router passes it

        image_gcs_filename = f"img_ctx_{uuid.uuid4()}.png" # img_ctx for context-aware
        blob_name = f"campfire_images/{username}/{image_gcs_filename}"
        bucket = gcs_client.bucket(gcs_bucket_name)
        blob = bucket.blob(blob_name)

        await asyncio.to_thread(
            blob.upload_from_string,
            image_bytes,
            content_type="image/png"
        )

        logger.info(f"Campfire image '{blob_name}' uploaded to GCS (via utility) for user '{username}'.")
        return blob.public_url

    except google_auth_exceptions.GoogleAuthError as e:
        logger.error(f"Vertex AI Image Gen Auth Error for user '{username}': {e}.")
        return None
    except Exception as e:
        logger.exception(f"Error generating or uploading image (via utility) for user '{username}': {e}")
        return None