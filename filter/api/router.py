from fastapi import APIRouter

from pydantic import BaseModel


class CheckMailRequest(BaseModel):
    email_from: str
    email_to: str
    subject: str
    body: str


router = APIRouter()


@router.get("mail/checked")
def check(mail: CheckMailRequest):
