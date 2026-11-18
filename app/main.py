from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.rag.generator import answer
from app.rag.retriever import ROLE_LEVEL, OFF_TOPIC

app = FastAPI(title="Workplace Copilot")


class AskRequest(BaseModel):
    question: str
    role: str  # TEMPORARY: the next step replaces this with the role from a login token


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post(
    "/ask",
    responses={
        400: {"description": "Unknown role"},
        404: {"description": "The question is not about company policies"},
    },
)
def ask(req: AskRequest):
    if req.role not in ROLE_LEVEL:
        raise HTTPException(status_code=400, detail="Unknown role")

    result = answer(req.question, req.role)

    # Off-topic question (nothing like it in any policy): 404 with the custom message.
    # Restricted questions are NOT handled here on purpose: they keep returning 200
    # with the normal "couldn't find" text, so nobody can tell a hidden section exists.
    if result["answer"] == OFF_TOPIC:
        raise HTTPException(status_code=404, detail=OFF_TOPIC)

    return result
