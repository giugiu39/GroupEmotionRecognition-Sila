from pydantic import BaseModel


class Message(BaseModel):
    """A single message in a conversation."""

    role: str
    content: str


class AskAgentRequest(BaseModel):
    """Request body for the chatbot endpoint."""

    messages: list[Message]
    foto: str | None = None


class AskAgentResponse(BaseModel):
    """Response from the chatbot endpoint."""

    response: str
