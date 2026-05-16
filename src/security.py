from fastapi import Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from dotenv import load_dotenv
import os

load_dotenv()

# --- AUTH SETUP ---
security = HTTPBasic()

APP_USER = os.getenv("APP_USER")
APP_PASS = os.getenv("APP_PASS")

def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username != APP_USER and credentials.password != APP_PASS:
        raise HTTPException(
            status_code=400,
            detail="Unauthorized",
            headers = {"WW-Authenticate": "Basic"}
        )
    return credentials.username