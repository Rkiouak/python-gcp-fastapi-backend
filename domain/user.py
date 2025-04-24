from pydantic import BaseModel

class SignUpUser(BaseModel):
    username: str
    email: str | None = None
    given_name: str | None = None
    family_name: str | None = None
    password: str | None = None
    disabled: bool | None = None
