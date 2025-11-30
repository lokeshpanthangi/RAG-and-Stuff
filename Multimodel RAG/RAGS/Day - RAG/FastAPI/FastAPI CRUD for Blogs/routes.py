from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from schema import BlogBase, UpdateBlog, UpdateUser, UserBase
from database import get_db
from sqlalchemy.orm import Session
from models import User, Blog

router = APIRouter()





@router.get("/")
def get_index():
    return FileResponse("index.html")  

@router.post("/createuser")
def create_user(user: UserBase, db: Session = Depends(get_db)):
    db_user = User(
        email=user.email,
        name=user.name,
        gender=user.Gender
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    db.close()
    return db_user

@router.post("/createblog")
def create_blog(blog: BlogBase, db: Session = Depends(get_db)):
    db_blog = Blog(
        title=blog.title,
        content=blog.content,
        author_id=blog.author_id
    )
    db.add(db_blog)
    db.commit()
    db.refresh(db_blog)
    db.close()
    return db_blog

@router.get("/getuser/{user_id}")
def get_user(user_id: int, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user

@router.get("/getblog/{blog_id}")
def get_blog(blog_id: int, db: Session = Depends(get_db)):
    db_blog = db.query(Blog).filter(Blog.id == blog_id).first()
    if not db_blog:
        raise HTTPException(status_code=404, detail="Blog not found")
    return db_blog

@router.get("/getallusers")
def get_all_users(db: Session = Depends(get_db)):
    db_users = db.query(User).all()
    return db_users

@router.get("/getallblogs")
def get_all_blogs(db: Session = Depends(get_db)):
    db_blogs = db.query(Blog).all()
    return db_blogs

@router.put("/updateblog/{blog_id}")
def update_blog(blog_id: int, blog: UpdateBlog, db: Session = Depends(get_db)):
    db_blog = db.query(Blog).filter(Blog.id == blog_id).first()
    if not db_blog:
        raise HTTPException(status_code=404, detail="Blog not found")
    if blog.title is not None:
        db_blog.title = blog.title
    if blog.content is not None:
        db_blog.content = blog.content
    db.commit()
    db.refresh(db_blog)
    db.close()
    return db_blog


@router.put("/updateuser/{user_id}")
def update_user(user_id: int, user: UpdateUser, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    if user.email is not None:
        db_user.email = user.email
    if user.name is not None:
        db_user.name = user.name
    if user.Gender is not None:
        db_user.gender = user.Gender
    db.commit()
    db.refresh(db_user)
    db.close()
    return db_user

@router.post("/delete_user/{user_id}")
def delete_user(user_id: int, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    db.delete(db_user)
    db.commit()
    db.close()
    return {"detail": "User deleted successfully"}

@router.post("/delete_blog/{blog_id}")
def delete_blog(blog_id: int, db: Session = Depends(get_db)):
    db_blog = db.query(Blog).filter(Blog.id == blog_id).first()
    if not db_blog:
        raise HTTPException(status_code=404, detail="Blog not found")
    db.delete(db_blog)
    db.commit()
    db.close()
    return {"detail": "Blog deleted successfully"}
