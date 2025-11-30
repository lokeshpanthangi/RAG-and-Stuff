from datetime import datetime,timedelta
from re import A
from grpc import access_token_call_credentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException,status
from fastapi.security import HTTPAuthorizationCredentials,HTTPBearer

secret_key = "Q2hszywkvb@"
algorithm = "HS256"
access_token_expiration = 30


