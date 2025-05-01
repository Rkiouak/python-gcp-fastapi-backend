# In main.py

from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, status, Request # Added Request
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette_authlib.middleware import AuthlibMiddleware as SessionMiddleware

import logging
import AuthAndUser as auth
from cryptography.fernet import Fernet
import secretmanager
from contextlib import asynccontextmanager
import base64

from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os

from google.cloud import firestore
from google.cloud import storage # <-- Import GCS client

# Import routers
from routers import users, posts, conversations
ACCESS_TOKEN_EXPIRE_MINUTES = 150

logger = logging.getLogger('uvicorn.error')


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup: Initializing resources...")
    try:
        app.state.fernet = Fernet(
            base64.b64decode(secretmanager.get_secret("projects/4042672389/secrets/fernet_asymmetric_key/versions/1"))
        )
        logger.info("Fernet client initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize Fernet: {e}")
        app.state.fernet = None

    try:
        app.state.db = firestore.AsyncClient()
        logger.info("Firestore Async client initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize Firestore Async client: {e}")
        app.state.db = None

    try:
        app.state.gcs_client = storage.Client()
        logger.info("Google Cloud Storage client initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize GCS client: {e}")
        app.state.gcs_client = None

    yield
    logger.info("Application shutdown: Cleaning up resources...")
    if hasattr(app.state, 'db') and app.state.db:
        try:
            await app.state.db.close() # Close the async client
            logger.info("Firestore Async client closed.")
        except Exception as e:
             logger.error(f"Error closing Firestore client: {e}")
    pass


app = FastAPI(lifespan=lifespan)
app.include_router(users.router)
app.include_router(posts.router)
app.include_router(conversations.router)

app.add_middleware(
    TrustedHostMiddleware, allowed_hosts=["musings-mr.net", "*.musings-mr.net", "localhost", "127.0.0.1"]
)

@app.post("/token")
async def login_for_access_token(
        form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
        request: Request # Access request state if needed
) -> auth.Token:
    # You might want to check if app.state.db is available here if auth depends on it
    if not request.app.state.db:
         raise HTTPException(status_code=503, detail="Database service unavailable")

    user = auth.authenticate_user(form_data.username, form_data.password) # Assuming authenticate_user handles db connection if needed
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth.create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return auth.Token(access_token=access_token, token_type="bearer")