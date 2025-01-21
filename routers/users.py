from fastapi import APIRouter
import AuthAndUser as auth
from typing import Annotated, Optional
from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel
import logging
from google.cloud import firestore

logger = logging.getLogger('uvicorn.error')

router = APIRouter()

class SignUpUser(BaseModel):
    username: str
    email: str | None = None
    given_name: str | None = None
    family_name: str | None = None
    password: str | None = None
    disabled: bool | None = None

@router.get("/users/me/", response_model=auth.User, tags=["users"])
async def read_users_me(
    current_user: Annotated[auth.User, Depends(auth.get_current_active_user)],
):
    return current_user

@router.post("/users/")
async def create_user(user: SignUpUser,  tags=["users"]):
    logger.info(f"Creating user {user}")
    print(user)
    db = firestore.Client()
    users_ref = db.collection("users")
    users_ref.document(user.username).set({
        "username": user.username,
        "given_name": user.given_name,
        "family_name": user.family_name,
        "email": user.email,
        "hashed_password": auth.pwd_context.hash(user.password)
    })
    return user