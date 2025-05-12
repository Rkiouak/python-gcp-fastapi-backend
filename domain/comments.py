# In a new file, perhaps domain/comments.py or within routers/posts.py if you prefer
from pydantic import BaseModel, Field
from typing import Optional
import datetime
import uuid

class Comment(BaseModel):
    id: str = Field(default_factory=lambda: f"comment-{uuid.uuid4().hex}")
    post_id: str
    author: str # Username of the commenter
    content: str
    date: datetime.datetime = Field(default_factory=datetime.datetime.now)
    parent_comment_id: Optional[str] = None # For threaded comments

    class Config:
        from_attributes = True