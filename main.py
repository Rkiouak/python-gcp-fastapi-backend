from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
import logging
import AuthAndUser as auth
from cryptography.fernet import Fernet
import secretmanager
from contextlib import asynccontextmanager
import base64

from routers import users

ACCESS_TOKEN_EXPIRE_MINUTES = 30

logger = logging.getLogger('uvicorn.error')

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info('Starting lifespan')
    app.state.fernet = Fernet(
    base64.b64decode(secretmanager.get_secret("projects/4042672389/secrets/fernet_asymmetric_key/versions/1")))
    logger.info('fernet set on app')
    yield

app = FastAPI(lifespan=lifespan)
app.include_router(users.router)

@app.post("/token")
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
) -> auth.Token:
    user = auth.authenticate_user(form_data.username, form_data.password)
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