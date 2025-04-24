from contextlib import asynccontextmanager

from fastapi import APIRouter
import AuthAndUser as auth
from typing import Annotated, Optional
from fastapi import Depends, FastAPI, HTTPException, status
import logging
from google.cloud import firestore
from cryptography.fernet import Fernet
import secretmanager
import base64
import pickle
import sendgridemail
from domain.user import SignUpUser

logger = logging.getLogger('uvicorn.error')

router = APIRouter()
app = FastAPI()


fernet = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global fernet
    fernet = Fernet(
    base64.b64decode(secretmanager.get_secret("projects/4042672389/secrets/fernet_asymmetric_key/versions/1")))
    yield

@router.get("/users/me/", response_model=auth.User, tags=["users"])
async def read_users_me(
    current_user: Annotated[auth.User, Depends(auth.get_current_active_user)],
):
    return current_user

@router.post("/users/")
async def create_user(challenge: str,  tags=["users"]):
    global fernet
    decoded_challenge = base64.b64decode(challenge)
    decrypted_challenge = fernet.decrypt(decoded_challenge)
    user = pickle.loads(decrypted_challenge)
    logger.info(f"Creating user {user}")
    print(user)
    db = firestore.Client()
    users_ref = db.collection("users")
    users_ref.document(user.username).set({
        "username": user.username,
        "given_name": user.given_name,
        "family_name": user.family_name,
        "email": user.email,
        "hashed_password": auth.get_password_hash(user.password)
    })
    return user

@router.post("/challenge/")
async def create_challenge(user: SignUpUser):
    global fernet
    challenge_bytes = fernet.encrypt(pickle.dumps(user))
    challenge = base64.b64encode(challenge_bytes)
    sendgridemail.send_email(user, challenge)
