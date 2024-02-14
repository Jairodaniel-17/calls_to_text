from pydantic import BaseModel


class CredentialsHF(BaseModel):
    token: str
    model_name: str
