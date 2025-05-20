# In routers/posts.py

import logging
from fastapi import (
    APIRouter, HTTPException, Request, Depends, status,
    File, UploadFile, Form
)
from google.cloud import firestore # Firestore client
from google.cloud.firestore import AsyncClient # Specifically import AsyncClient for type hinting
from google.cloud import storage
from pydantic import BaseModel, Field
from typing import List, Optional, Annotated
import datetime
import uuid
import re # Import the 're' module for regular expressions

import AuthAndUser as auth

logger = logging.getLogger('uvicorn.error')

# --- Constants ---
GCS_BUCKET_NAME = "musings-mr.net"
MAX_IMAGE_SIZE_KB = 600
MAX_IMAGE_SIZE_BYTES = MAX_IMAGE_SIZE_KB * 1024
MAX_TEXT_FIELD_SIZE_KB = 500
MAX_TEXT_FIELD_SIZE_BYTES = MAX_TEXT_FIELD_SIZE_KB * 1024
MAX_SLUG_LENGTH = 200 # Define a max length for generated slugs

router = APIRouter(
    prefix="/posts",
    tags=["posts", "comments"]
)

# --- Pydantic Models (as before) ---
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
    imageUrl: Optional[str] = Field(default=None)
    class Config:
         from_attributes = True

class CommentBase(BaseModel):
    content: str
    parent_comment_id: Optional[str] = None

class Comment(CommentBase):
    id: str = Field(default_factory=lambda: f"comment-{uuid.uuid4().hex}")
    post_id: str
    author: str
    date: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))
    class Config:
        from_attributes = True

# --- Helper Functions (get_firestore_client, get_gcs_client, upload_to_gcs as before) ---
async def get_firestore_client(request: Request) -> AsyncClient:
    if not hasattr(request.app.state, 'db') or not request.app.state.db:
        logger.error("Firestore client not initialized or unavailable.")
        raise HTTPException(status_code=503, detail="Database service unavailable")
    if not isinstance(request.app.state.db, AsyncClient):
         logger.error("Firestore client is not an AsyncClient in posts router.")
         raise HTTPException(status_code=503, detail="Database service misconfigured for posts")
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

def generate_post_slug(title: str) -> str:
    """
    Generates a Firestore-friendly document ID (slug) from a title.
    """
    if not title or not title.strip():
        # Fallback for empty or whitespace-only titles
        return f"untitled-post-{uuid.uuid4().hex[:12]}"

    # Convert to lowercase and strip leading/trailing whitespace
    slug = title.lower().strip()

    # Replace non-alphanumeric characters (excluding hyphens) with an empty string.
    # This step handles many special characters.
    slug = re.sub(r'[^\w\s-]', '', slug)

    # Replace whitespace and sequences of hyphens with a single hyphen
    slug = re.sub(r'[-\s]+', '-', slug)

    # Remove leading/trailing hyphens that might have formed
    slug = slug.strip('-')

    # If slug becomes empty after processing (e.g., title was "!!! ???"), generate a unique one
    if not slug:
        return f"untitled-post-{uuid.uuid4().hex[:12]}"

    # Firestore ID constraints checks (simplified for common cases)
    if slug == "." or slug == "..":
        slug = f"post-{slug}-{uuid.uuid4().hex[:8]}" # Append uuid to make it valid

    # Ensure slug doesn't start and end with double underscores (Firestore reserved)
    if slug.startswith("__") and slug.endswith("__"):
         # A simple way to break the pattern, e.g., by prefixing
        slug = f"post-{slug}"

    # Truncate to a maximum length to avoid overly long IDs
    if len(slug) > MAX_SLUG_LENGTH:
        slug = slug[:MAX_SLUG_LENGTH].rsplit('-', 1)[0] # Truncate and try to cut at a hyphen
        slug = slug.strip('-') # Clean up if truncation left a hyphen

    # Final check if slug became empty after truncation
    if not slug:
        return f"final-fallback-post-{uuid.uuid4().hex[:12]}"

    return slug


# --- Post API Routes (get_all_post_snippets, get_post_by_id as before) ---
@router.get("/", response_model=List[PostSnippet])
async def get_all_post_snippets(
    db: AsyncClient = Depends(get_firestore_client)
):
    # ... (implementation as before) ...
    posts_collection = db.collection('posts')
    try:
        all_post_snippets = []
        async for doc in posts_collection.order_by("date", direction=firestore.Query.DESCENDING).stream():
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
    db: AsyncClient = Depends(get_firestore_client)
):
    # ... (implementation as before) ...
    posts_collection = db.collection('posts')
    try:
        post_ref = posts_collection.document(post_id)
        post_doc = await post_ref.get()
        if not post_doc.exists:
            logger.warning(f"Post document with ID {post_id} not found.")
            raise HTTPException(status_code=404, detail=f"Post with id {post_id} not found")
        post_data = post_doc.to_dict()
        post_data['id'] = post_doc.id # Ensure 'id' is part of the data for model validation
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
    db: AsyncClient = Depends(get_firestore_client),
    gcs: storage.Client = Depends(get_gcs_client)
):
    if current_user.username != "mrkiouak@gmail.com":
        raise HTTPException(status_code=403, detail="You are not authorized to post.")

    if len(title.encode('utf-8')) > MAX_TEXT_FIELD_SIZE_BYTES or \
       len(snippet.encode('utf-8')) > MAX_TEXT_FIELD_SIZE_BYTES or \
       len(content.encode('utf-8')) > MAX_TEXT_FIELD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"One or more text fields exceed the maximum size of {MAX_TEXT_FIELD_SIZE_KB} KB."
        )

    # Generate the document ID (slug) from the title
    try:
        post_id_slug = generate_post_slug(title)
        if not post_id_slug: # Should be handled by generate_post_slug, but as a safeguard
            raise ValueError("Generated post ID slug is empty.")
    except ValueError as ve:
        logger.error(f"Error generating post ID slug from title '{title}': {ve}")
        raise HTTPException(status_code=400, detail=f"Could not generate a valid ID from the title: {ve}")

    logger.info(f"Generated post ID slug: '{post_id_slug}' from title: '{title}'")

    posts_collection = db.collection('posts')
    post_ref = posts_collection.document(post_id_slug)

    # Check if a document with this generated ID already exists
    try:
        existing_doc = await post_ref.get()
        if existing_doc.exists:
            logger.warning(f"Post with generated ID '{post_id_slug}' (from title '{title}') already exists.")
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"A post with a title that generates the ID '{post_id_slug}' already exists. Please choose a different title."
            )
    except Exception as e:
        logger.exception(f"Error checking for existing post with ID '{post_id_slug}': {e}")
        raise HTTPException(status_code=500, detail="Error checking for existing post.")


    image_public_url: Optional[str] = None
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

    processed_date_str: str
    if date:
        try:
            parsed_date = datetime.datetime.fromisoformat(date.replace('Z', '+00:00')).date()
            processed_date_str = parsed_date.isoformat()
        except ValueError:
             raise HTTPException(status_code=400, detail=f"Invalid date format: '{date}'. Use YYYY-MM-DD.")
    else:
        processed_date_str = datetime.date.today().isoformat()

    new_post_data_dict = {
        "title": title, # Original title is stored in the document body
        "snippet": snippet,
        "content": content,
        "author": current_user.username,
        "date": processed_date_str,
        "imageUrl": image_public_url,
        # Do NOT include 'id' here, it's the document key
    }

    try:
        # Use .set() with the generated post_id_slug
        await post_ref.set(new_post_data_dict)
        # The timestamp of write is not directly returned by .set() like .add()
        # If you need it, you'd typically add a server_timestamp field in new_post_data_dict
        # For logging, we can log the successful write.
        logger.info(f"User '{current_user.username}' created post with ID '{post_id_slug}' and image URL {image_public_url}")

        # Construct the response model, explicitly passing the generated ID
        response_data = Post(id=post_id_slug, **new_post_data_dict)
        return response_data
    except Exception as e:
        logger.exception(f"Error creating Firestore post with ID '{post_id_slug}' for user '{current_user.username}': {e}")
        raise HTTPException(status_code=500, detail="Internal server error while creating post")


# --- Comment API Routes (as before) ---
# ... (rest of the Comment API routes: create_comment, get_comments_for_post, get_comment_by_id)
COMMENTS_SUBCOLLECTION = "comments"

@router.post("/{post_id}/comments/", response_model=Comment, status_code=status.HTTP_201_CREATED)
async def create_comment(
    post_id: str,
    comment_in: CommentBase,
    db: AsyncClient = Depends(get_firestore_client),
    current_user: auth.User = Depends(auth.get_current_active_user)
):
    # ... (implementation as before) ...
    posts_collection = db.collection('posts')
    post_ref = posts_collection.document(post_id)

    post_doc = await post_ref.get()
    if not post_doc.exists:
        logger.warning(f"Attempt to comment on non-existent post {post_id} by user {current_user.username}")
        raise HTTPException(status_code=404, detail=f"Post with id {post_id} not found.")

    if len(comment_in.content.encode('utf-8')) > MAX_TEXT_FIELD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Comment content exceeds the maximum size of {MAX_TEXT_FIELD_SIZE_KB} KB."
        )
    new_comment_obj = Comment(
        post_id=post_id,
        author=current_user.username,
        content=comment_in.content,
        parent_comment_id=comment_in.parent_comment_id,
    )
    try:
        comment_doc_ref = post_ref.collection(COMMENTS_SUBCOLLECTION).document(new_comment_obj.id)
        await comment_doc_ref.set(new_comment_obj.model_dump())
        logger.info(f"User '{current_user.username}' created comment '{new_comment_obj.id}' on post '{post_id}'")
        return new_comment_obj
    except Exception as e:
        logger.exception(f"Error creating comment for post '{post_id}' by user '{current_user.username}': {e}")
        raise HTTPException(status_code=500, detail="Internal server error while creating comment.")

@router.get("/{post_id}/comments/", response_model=List[Comment])
async def get_comments_for_post(
    post_id: str,
    db: AsyncClient = Depends(get_firestore_client)
):
    # ... (implementation as before) ...
    posts_collection = db.collection('posts')
    post_ref = posts_collection.document(post_id)
    post_doc = await post_ref.get()
    if not post_doc.exists:
        logger.warning(f"Attempt to get comments for non-existent post {post_id}")
        raise HTTPException(status_code=404, detail=f"Post with id {post_id} not found.")
    try:
        comments_query = post_ref.collection(COMMENTS_SUBCOLLECTION).order_by("date", direction=firestore.Query.ASCENDING)
        all_comments = []
        async for doc in comments_query.stream():
            comment_data = doc.to_dict()
            comment_data['id'] = doc.id
            try:
                all_comments.append(Comment(**comment_data))
            except Exception as validation_error:
                logger.error(f"Data validation error for comment {doc.id} in post {post_id}: {validation_error}. Data: {comment_data}")
                continue
        return all_comments
    except Exception as e:
        logger.exception(f"Error retrieving comments for post {post_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error while fetching comments.")


@router.get("/{post_id}/comments/{comment_id}", response_model=Comment)
async def get_comment_by_id(
    post_id: str,
    comment_id: str,
    db: AsyncClient = Depends(get_firestore_client)
):
    # ... (implementation as before) ...
    posts_collection = db.collection('posts')
    post_ref = posts_collection.document(post_id)
    post_doc = await post_ref.get()
    if not post_doc.exists:
        logger.warning(f"Attempt to get specific comment from non-existent post {post_id}")
        raise HTTPException(status_code=404, detail=f"Post with id {post_id} not found.")
    try:
        comment_doc_ref = post_ref.collection(COMMENTS_SUBCOLLECTION).document(comment_id)
        comment_doc = await comment_doc_ref.get()
        if not comment_doc.exists:
            logger.warning(f"Comment {comment_id} not found in post {post_id}")
            raise HTTPException(status_code=404, detail=f"Comment with id {comment_id} not found in post {post_id}.")
        comment_data = comment_doc.to_dict()
        comment_data['id'] = comment_doc.id
        return Comment(**comment_data)
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.exception(f"Error retrieving comment {comment_id} for post {post_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error while fetching specific comment.")