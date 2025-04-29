# In posts.py

import logging
from fastapi import (
    APIRouter, HTTPException, Request, Depends, status,
    File, UploadFile, Form
)
from google.cloud import firestore
from google.cloud import storage
from pydantic import BaseModel, Field
from typing import List, Optional, Annotated
import datetime
import uuid

import AuthAndUser as auth

logger = logging.getLogger('uvicorn.error')

# --- Constants ---
GCS_BUCKET_NAME = "musings-mr.net"
MAX_IMAGE_SIZE_KB = 600
MAX_IMAGE_SIZE_BYTES = MAX_IMAGE_SIZE_KB * 1024
MAX_TEXT_FIELD_SIZE_KB = 100 # <-- New constant for text fields
MAX_TEXT_FIELD_SIZE_BYTES = MAX_TEXT_FIELD_SIZE_KB * 1024 # <-- New constant

router = APIRouter(
    prefix="/posts",
    tags=["posts"]
)

class Post(BaseModel):
    id: str
    title: str
    author: str
    date: str | datetime.date
    imageUrl: Optional[str] = Field(default=None)
    snippet: str
    content: str
    class Config:
         from_attributes = True

class PostSnippet(BaseModel):
    id: str
    title: str
    author: str
    date: str | datetime.date
    snippet: str
    # Also uses imageUrl if snippets should show images
    imageUrl: Optional[str] = Field(default=None)
    class Config:
         from_attributes = True


async def get_firestore_client(request: Request) -> firestore.Client:
    if not hasattr(request.app.state, 'db') or not request.app.state.db:
        logger.error("Firestore client not initialized or unavailable.")
        raise HTTPException(status_code=503, detail="Database service unavailable")
    return request.app.state.db

async def get_gcs_client(request: Request) -> storage.Client:
    if not hasattr(request.app.state, 'gcs_client') or not request.app.state.gcs_client:
        logger.error("GCS client not initialized or unavailable.")
        raise HTTPException(status_code=503, detail="GCS service unavailable")
    return request.app.state.gcs_client


async def upload_to_gcs(
    gcs_client: storage.Client,
    image_bytes: bytes,
    filename: str,
    content_type: str,
    username: str
) -> Optional[str]:
    # ... (implementation as before)
    if not image_bytes or not filename:
        return None
    try:
        bucket = gcs_client.bucket(GCS_BUCKET_NAME)
        safe_filename = f"{uuid.uuid4()}_{filename.replace(' ', '_')}"
        blob_name = f"post_images/{username}/{safe_filename}"
        blob = bucket.blob(blob_name)
        blob.upload_from_string(image_bytes, content_type=content_type)
        logger.info(f"File {filename} uploaded to gs://{GCS_BUCKET_NAME}/{blob_name}")
        return blob.public_url
    except Exception as e:
        logger.exception(f"Failed to upload {filename} to GCS for user {username}: {e}")
        return None

# --- API Routes ---

@router.get("/", response_model=List[PostSnippet])
async def get_all_post_snippets(
    current_user: Annotated[auth.User, Depends(auth.get_current_active_user)],
    db: firestore.Client = Depends(get_firestore_client)
):
    """Retrieve summaries (including image URL) for all blog posts."""
    posts_collection = db.collection('posts')
    try:
        all_post_snippets = []
        docs = posts_collection.stream()
        for doc in docs:
            post_data = doc.to_dict()
            post_data['id'] = doc.id
            try:
                snippet = PostSnippet(**post_data)
                all_post_snippets.append(snippet)
            except Exception as validation_error:
                 logger.error(f"Data validation error for snippet doc {doc.id}: {validation_error}. Data: {post_data}")
                 continue
        return all_post_snippets
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.exception(f"Error retrieving all post snippets: {e}")
        raise HTTPException(status_code=500, detail="Internal server error while fetching post snippets")


@router.get("/{post_id}", response_model=Post)
async def get_post_by_id(
    post_id: str,
    current_user: Annotated[auth.User, Depends(auth.get_current_active_user)],
    db: firestore.Client = Depends(get_firestore_client)
):
    """Retrieve a specific blog post (including image URL) by its ID."""
    posts_collection = db.collection('posts')
    try:
        post_ref = posts_collection.document(post_id)
        post_doc = post_ref.get()
        if not post_doc.exists:
            logger.warning(f"Post document with ID {post_id} not found.")
            raise HTTPException(status_code=404, detail=f"Post with id {post_id} not found")
        post_data = post_doc.to_dict()
        post_data['id'] = post_doc.id
        return Post(**post_data)
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.exception(f"Error retrieving post {post_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error while fetching post")


@router.post("/", response_model=Post, status_code=status.HTTP_201_CREATED)
async def create_post(
    current_user: Annotated[auth.User, Depends(auth.get_current_active_user)],
    title: str = Form(...),
    snippet: str = Form(...),
    content: str = Form(...),
    date: Optional[str] = Form(default=None),
    image_file: Optional[UploadFile] = File(default=None, description=f"Optional image file (max {MAX_IMAGE_SIZE_KB} KB)"),
    db: firestore.Client = Depends(get_firestore_client),
    gcs: storage.Client = Depends(get_gcs_client)
):
    """
    Create a new blog post. Requires authentication.
    Accepts form data (title, snippet, content <= 100KB each) and
    an optional image file upload (max 600 KB).
    Image is uploaded to GCS; URL is stored.
    """
    # --- Validate Text Field Sizes ---
    # Check title size
    if len(title.encode('utf-8')) > MAX_TEXT_FIELD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Title exceeds the maximum size of {MAX_TEXT_FIELD_SIZE_KB} KB."
        )
    # Check snippet size
    if len(snippet.encode('utf-8')) > MAX_TEXT_FIELD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Snippet exceeds the maximum size of {MAX_TEXT_FIELD_SIZE_KB} KB."
        )
    # Check content size
    if len(content.encode('utf-8')) > MAX_TEXT_FIELD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Content exceeds the maximum size of {MAX_TEXT_FIELD_SIZE_KB} KB."
        )

    # --- Continue with image processing and other logic if text fields are valid ---
    posts_collection = db.collection('posts')
    image_public_url: Optional[str] = None
    processed_date: str

    # --- Process and Validate Image File ---
    if image_file and image_file.filename:
        image_bytes = await image_file.read()
        if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
            logger.warning(f"User '{current_user.username}' attempted to upload oversized image: {image_file.filename} ({len(image_bytes)} bytes)")
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Image file size exceeds the limit of {MAX_IMAGE_SIZE_KB} KB."
            )
        image_public_url = await upload_to_gcs(
            gcs_client=gcs,
            image_bytes=image_bytes,
            filename=image_file.filename,
            content_type=image_file.content_type or 'application/octet-stream',
            username=current_user.username
        )
        if image_public_url is None:
             logger.error(f"GCS image upload failed for {image_file.filename}, proceeding without image URL.")

    # --- Process Date ---
    if date:
        try:
            # Handle potential timezone info if present in ISO string
            parsed_date = datetime.datetime.fromisoformat(date.replace('Z', '+00:00')).date() if isinstance(date, str) else date
            processed_date = parsed_date.isoformat()
        except ValueError:
             raise HTTPException(status_code=400, detail=f"Invalid date format: '{date}'. Use YYYY-MM-DD.")
    else:
        processed_date = datetime.date.today().isoformat()

    new_post_data = {
        "title": title,
        "snippet": snippet,
        "content": content,
        "author": current_user.username,
        "date": processed_date,
        "imageUrl": image_public_url
    }

    try:
        update_time, doc_ref = posts_collection.add(new_post_data)
        logger.info(f"User '{current_user.username}' created post {doc_ref.id} with image URL {image_public_url} at {update_time}")
        response_data = new_post_data
        response_data['id'] = doc_ref.id
        return Post(**response_data)
    except Exception as e:
        logger.exception(f"Error creating Firestore post for user '{current_user.username}': {e}")
        raise HTTPException(status_code=500, detail="Internal server error while creating post")
