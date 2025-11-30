from unittest.mock import Base
from pydantic import BaseModel
from typing import Optional




class UserBase(BaseModel):
    email: str
    name: str
    Gender: Optional[str]

class BlogBase(BaseModel):
    title: str
    content: str
    author_id: int

class UpdateBlog(BaseModel):
    title: Optional[str]
    content: Optional[str]

class UpdateUser(BaseModel):
    email: Optional[str]
    name: Optional[str]
    Gender: Optional[str]

