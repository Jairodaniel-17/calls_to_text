# Project created by:  Jairo Daniel Mendoza Torres
from pydantic import BaseModel, validator


class DataBaseCredentials(BaseModel):
    """
    Model class for storing database credentials.
    """

    host: str
    user: str
    password: str
    database: str
    port: int
