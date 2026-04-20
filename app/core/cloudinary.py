# app/utils/cloudinary.py
import cloudinary
import cloudinary.uploader
from fastapi import UploadFile, HTTPException
from app.core.config import settings
# Cấu hình (Nên lấy từ biến môi trường .env)
cloudinary.config( 
  cloud_name = settings.CLOUDINARY_CLOUD_NAME, 
  api_key = settings.CLOUDINARY_API_KEY, 
  api_secret = settings.CLOUDINARY_API_SECRET,
  secure = True
)

def upload_image_file(file: UploadFile) -> str:
    """
    Upload file lên Cloudinary và trả về URL
    """
    try:
        # Cloudinary có thể đọc trực tiếp file-like object từ FastAPI
        response = cloudinary.uploader.upload(file.file)
        return response.get("secure_url")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image upload failed: {str(e)}")