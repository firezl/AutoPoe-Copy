import fastapi_poe as fp
import asyncio
from fastapi import FastAPI, WebSocket, HTTPException, Response, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union, Literal
import time
import os

app = FastAPI()

api_key = os.environ.get("POE_API_KEY")


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "owner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []


class FunctionCallResponse(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system", "function"]
    content: str = None
    name: Optional[str] = None
    function_call: Optional[FunctionCallResponse] = None


class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None
    function_call: Optional[FunctionCallResponse] = None


## for Embedding
class EmbeddingRequest(BaseModel):
    input: List[str]
    model: str


class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    data: list
    model: str
    object: str
    usage: CompletionUsage


# for ChatCompletionRequest


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.8
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    tools: Optional[Union[dict, List[dict]]] = None
    repetition_penalty: Optional[float] = 1.1


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "function_call"]


class ChatCompletionResponseStreamChoice(BaseModel):
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length", "function_call"]]
    index: int


class ChatCompletionResponse(BaseModel):
    model: str
    id: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[
        Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]
    ]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    usage: Optional[UsageInfo] = None


EventSourceResponse.DEFAULT_PING_INTERVAL = 100

app = FastAPI()
llms = ModelList(
    data=[
        ModelCard(id="Assistant"),
        ModelCard(id="Claude-instant"),
        ModelCard(id="Claude-instant-100k"),
        ModelCard(id="Gemini-Pro"),
        ModelCard(id="Web-Search"),
        ModelCard(id="ChatGPT"),
        ModelCard(id="Claude-3-Sonnet"),
        ModelCard(id="Claude-3-Sonnet-200k"),
        ModelCard(id="Claude-3-Haiku"),
        ModelCard(id="Claude-3-Haiku-200k"),
        ModelCard(id="Claude-3-Opus"),
        ModelCard(id="Claude-3-Opus-200k"),
        ModelCard(id="GPT-4"),
        ModelCard(id="GPT-3.5-Turbo"),
    ]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    return llms


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    if request.model not in [llm.id for llm in llms.data]:
        raise HTTPException(status_code=404, detail="model not found")
    if len(request.messages) < 1 or request.messages[-1].role == "assistant":
        raise HTTPException(status_code=400, detail="Invalid request")
    messages = []
    for message in request.messages:
        if message.role == "assistant":
            message.role = "bot"
        messages.append(
            fp.types.ProtocolMessage(
                role=message.role, content=message.content, name=message.name
            )
        )

    if request.stream:

        async def stream_gen():
            async for response in fp.client.get_bot_response(
                messages=messages,
                bot_name=request.model,
                api_key=api_key,
                temperature=request.temperature,
                tools=request.tools,
            ):
                text = response.text
                message = DeltaMessage(
                    content=text if text is not None else "",
                    role="assistant",
                    function_call=None,
                )
                choice_data = ChatCompletionResponseStreamChoice(
                    index=0,
                    delta=message,
                    finish_reason=None if text is not None else "stop",
                )
                chunk = ChatCompletionResponse(
                    model=request.model,
                    id="",
                    choices=[choice_data],
                    created=int(time.time()),
                    object="chat.completion.chunk",
                )
                yield "{}".format(chunk.model_dump_json(exclude_unset=True))

                if text is None:
                    yield "[DONE]"
                    break

        return EventSourceResponse(stream_gen(), media_type="text/event-stream")
    else:
        data = ""
        for text in await fp.client.get_bot_response(
            messages=messages,
            bot_name=request.model,
            api_key="",
            temperature=request.temperature,
            tools=request.tools,
        ):
            data = text
        message = DeltaMessage(
            content=data if data is not None else "",
            role="assistant",
            function_call=None,
        )
        choice_data = ChatCompletionResponseChoice(
            index=0, message=message, finish_reason=None if data is not None else "stop"
        )
        return ChatCompletionResponse(
            model=request.model,
            id="",
            choices=[choice_data],
            created=int(time.time()),
            object="chat.completion",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="100.64.0.24", port=8000, workers=1)
