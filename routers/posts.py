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

import AuthAndUser as auth

logger = logging.getLogger('uvicorn.error')

# --- Constants ---
GCS_BUCKET_NAME = "musings-mr.net"
MAX_IMAGE_SIZE_KB = 600
MAX_IMAGE_SIZE_BYTES = MAX_IMAGE_SIZE_KB * 1024
MAX_TEXT_FIELD_SIZE_KB = 500
MAX_TEXT_FIELD_SIZE_BYTES = MAX_TEXT_FIELD_SIZE_KB * 1024

router = APIRouter(
    prefix="/posts",
    tags=["posts", "comments"]
)

# --- Pydantic Models ---

class Post(BaseModel):
    id: str
    title: str
    author: str
    date: str | datetime.date # Keep as is, Firestore stores dates as timestamps or strings
    imageUrl: Optional[str] = Field(default=None)
    snippet: str
    content: str
    class Config:
         from_attributes = True

class PostSnippet(BaseModel):
    id: str
    title: str
    author: str
    date: str | datetime.date # Keep as is
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
        # Ensure Pydantic can handle datetime objects correctly when creating from attributes
        # and when serializing to JSON (though Firestore handles datetime objects natively)


# --- Helper Functions ---

async def get_firestore_client(request: Request) -> AsyncClient: # Type hint with AsyncClient
    if not hasattr(request.app.state, 'db') or not request.app.state.db:
        logger.error("Firestore client not initialized or unavailable.")
        raise HTTPException(status_code=503, detail="Database service unavailable")
    if not isinstance(request.app.state.db, AsyncClient): # Check instance type
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
        # Note: blob.upload_from_string is synchronous.
        # For a fully async GCS upload, you'd typically use aiohttp or similar
        # with GCS's resumable upload API, or run this in a thread pool.
        # For simplicity with the official library, this remains synchronous here.
        # If this becomes a bottleneck, consider libraries like gcloud-aio-storage.
        blob.upload_from_string(image_bytes, content_type=content_type)
        logger.info(f"File {filename} uploaded to gs://{GCS_BUCKET_NAME}/{blob_name}")
        return blob.public_url
    except Exception as e:
        logger.exception(f"Failed to upload {filename} to GCS for user {username}: {e}")
        return None

# --- Post API Routes ---

@router.get("/", response_model=List[PostSnippet])
async def get_all_post_snippets(
    db: AsyncClient = Depends(get_firestore_client) # Use AsyncClient
):
    posts_collection = db.collection('posts')
    try:
        all_post_snippets = []
        # Use "async for" with .stream() for AsyncClient
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
    db: AsyncClient = Depends(get_firestore_client) # Use AsyncClient
):
    posts_collection = db.collection('posts')
    try:
        post_ref = posts_collection.document(post_id)
        post_doc = await post_ref.get() # Use await for .get()
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
    date: Optional[str] = Form(default=None), # Dates are tricky with forms, ensure frontend sends ISO
    image_file: Optional[UploadFile] = File(default=None, description=f"Optional image file (max {MAX_IMAGE_SIZE_KB} KB)"),
    db: AsyncClient = Depends(get_firestore_client), # Use AsyncClient
    gcs: storage.Client = Depends(get_gcs_client)
):
    if current_user.username != "mrkiouak@gmail.com": # Example authorization
        raise HTTPException(status_code=403, detail="You are not authorized to post.")

    if len(title.encode('utf-8')) > MAX_TEXT_FIELD_SIZE_BYTES or \
       len(snippet.encode('utf-8')) > MAX_TEXT_FIELD_SIZE_BYTES or \
       len(content.encode('utf-8')) > MAX_TEXT_FIELD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"One or more text fields exceed the maximum size of {MAX_TEXT_FIELD_SIZE_KB} KB."
        )

    image_public_url: Optional[str] = None
    if image_file and image_file.filename:
        image_bytes = await image_file.read()
        if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
            logger.warning(f"User '{current_user.username}' attempted to upload oversized image: {image_file.filename} ({len(image_bytes)} bytes)")
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Image file size exceeds the limit of {MAX_IMAGE_SIZE_KB} KB."
            )
        # upload_to_gcs is currently synchronous. Consider making it async if it's a bottleneck.
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
            # Attempt to parse the date string. Assumes YYYY-MM-DD format from form.
            # For more robust date handling, consider a date parsing library or stricter validation.
            parsed_date = datetime.datetime.fromisoformat(date.replace('Z', '+00:00')).date()
            processed_date_str = parsed_date.isoformat()
        except ValueError:
             raise HTTPException(status_code=400, detail=f"Invalid date format: '{date}'. Use YYYY-MM-DD.")
    else:
        processed_date_str = datetime.date.today().isoformat()

    new_post_data_dict = {
        "title": title,
        "snippet": snippet,
        "content": content,
        "author": current_user.username,
        "date": processed_date_str, # Store as ISO string
        "imageUrl": image_public_url
        # Firestore will store the date string. If you need it as a Firestore Timestamp,
        # you might convert it: firestore.SERVER_TIMESTAMP or datetime.datetime.strptime(...)
        # For consistency with your model (str | datetime.date), string is fine.
    }

    try:
        posts_collection = db.collection('posts')
        # .add() returns a tuple (timestamp, document_reference)
        # For AsyncClient, .add() is a coroutine
        timestamp, doc_ref = await posts_collection.add(new_post_data_dict)
        logger.info(f"User '{current_user.username}' created post {doc_ref.id} with image URL {image_public_url} at {timestamp}")

        # Construct the response model
        response_data = Post(id=doc_ref.id, **new_post_data_dict)
        return response_data
    except Exception as e:
        logger.exception(f"Error creating Firestore post for user '{current_user.username}': {e}")
        raise HTTPException(status_code=500, detail="Internal server error while creating post")

# --- Comment API Routes ---

COMMENTS_SUBCOLLECTION = "comments"

@router.post("/{post_id}/comments/", response_model=Comment, status_code=status.HTTP_201_CREATED)
async def create_comment(
    post_id: str,
    comment_in: CommentBase,
    db: AsyncClient = Depends(get_firestore_client), # Use AsyncClient
    current_user: auth.User = Depends(auth.get_current_active_user)
):
    posts_collection = db.collection('posts')
    post_ref = posts_collection.document(post_id)

    post_doc = await post_ref.get() # Use await
    if not post_doc.exists:
        logger.warning(f"Attempt to comment on non-existent post {post_id} by user {current_user.username}")
        raise HTTPException(status_code=404, detail=f"Post with id {post_id} not found.")

    if len(comment_in.content.encode('utf-8')) > MAX_TEXT_FIELD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Comment content exceeds the maximum size of {MAX_TEXT_FIELD_SIZE_KB} KB."
        )

    # Create the full Comment object for storage and response
    new_comment_obj = Comment(
        # id and date are set by default_factory in the model
        post_id=post_id,
        author=current_user.username,
        content=comment_in.content,
        parent_comment_id=comment_in.parent_comment_id,
    )

    try:
        comment_doc_ref = post_ref.collection(COMMENTS_SUBCOLLECTION).document(new_comment_obj.id)
        await comment_doc_ref.set(new_comment_obj.model_dump()) # Use await for .set()

        logger.info(f"User '{current_user.username}' created comment '{new_comment_obj.id}' on post '{post_id}'")
        return new_comment_obj # Return the Pydantic model instance
    except Exception as e:
        logger.exception(f"Error creating comment for post '{post_id}' by user '{current_user.username}': {e}")
        raise HTTPException(status_code=500, detail="Internal server error while creating comment.")

@router.get("/{post_id}/comments/", response_model=List[Comment])
async def get_comments_for_post(
    post_id: str,
    db: AsyncClient = Depends(get_firestore_client) # Use AsyncClient
):
    posts_collection = db.collection('posts')
    post_ref = posts_collection.document(post_id)

    post_doc = await post_ref.get() # Use await
    if not post_doc.exists:
        logger.warning(f"Attempt to get comments for non-existent post {post_id}")
        raise HTTPException(status_code=404, detail=f"Post with id {post_id} not found.")

    try:
        comments_query = post_ref.collection(COMMENTS_SUBCOLLECTION).order_by("date", direction=firestore.Query.ASCENDING)
        all_comments = []
        async for doc in comments_query.stream(): # Use "async for"
            comment_data = doc.to_dict()
            # The ID is already in comment_data if stored correctly from model_dump,
            # but good to ensure it's explicitly set from doc.id for the response model
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
    db: AsyncClient = Depends(get_firestore_client) # Use AsyncClient
):
    posts_collection = db.collection('posts')
    post_ref = posts_collection.document(post_id)

    post_doc = await post_ref.get() # Use await
    if not post_doc.exists:
        logger.warning(f"Attempt to get specific comment from non-existent post {post_id}")
        raise HTTPException(status_code=404, detail=f"Post with id {post_id} not found.")

    try:
        comment_doc_ref = post_ref.collection(COMMENTS_SUBCOLLECTION).document(comment_id)
        comment_doc = await comment_doc_ref.get() # Use await

        if not comment_doc.exists:
            logger.warning(f"Comment {comment_id} not found in post {post_id}")
            raise HTTPException(status_code=404, detail=f"Comment with id {comment_id} not found in post {post_id}.")

        comment_data = comment_doc.to_dict()
        comment_data['id'] = comment_doc.id
        return Comment(**comment_data)
    except HTTPException as http_exc: # Re-raise HTTP exceptions
        raise http_exc
    except Exception as e:
        logger.exception(f"Error retrieving comment {comment_id} for post {post_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error while fetching specific comment.")
