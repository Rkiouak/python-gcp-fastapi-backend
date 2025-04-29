from fastapi import Request, APIRouter
import AuthAndUser as auth
from typing import Annotated
from fastapi import Depends
import logging
from google.cloud import firestore
import base64
import pickle
import sendgridemail
from domain.user import SignUpUser, Challenge

logger = logging.getLogger('uvicorn.error')

router = APIRouter()

@router.get("/users/me/", response_model=auth.User, tags=["users"])
async def read_users_me(
    current_user: Annotated[auth.User, Depends(auth.get_current_active_user)],
):
    return current_user

@router.post("/users/")
async def create_user(challenge: Challenge, request: Request,  tags=["users"]):
    decoded_challenge = base64.b64decode(challenge.challenge)
    decrypted_challenge = request.app.state.fernet.decrypt(decoded_challenge)
    user = pickle.loads(decrypted_challenge)
    logger.info(f"Creating user {user}")
    print(user)
    db = firestore.Client()
    users_ref = db.collection("users")
    existing_users = users_ref.where("email", "==", user.email).stream()
    result = next(existing_users, None)
    if result is not None:
        raise Exception(f"User with email: {user.email} already exists. Please log in.")
    existing_users = users_ref.where("username", "==", user.username).stream()
    result = next(existing_users, None)
    if result is not None:
        raise Exception(f"User with username: {user.username} already exists. Please log in.")
    users_ref.document(user.username).set({
        "username": user.username,
        "given_name": user.given_name,
        "family_name": user.family_name,
        "email": user.email,
        "hashed_password": auth.get_password_hash(user.password)
    })
    return user

@router.post("/challenge/")
async def create_challenge(user: SignUpUser, request: Request):
    challenge_bytes = request.app.state.fernet.encrypt(pickle.dumps(user))
    challenge = base64.b64encode(challenge_bytes).decode('utf-8')
    sendgridemail.send_email(user, challenge)
    return {"message":f"Validate your signup using the email sent to {user.email}"}
