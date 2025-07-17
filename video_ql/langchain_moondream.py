"""
Integration with Moondream API for VideoQL.
"""

import json
import os
from typing import Any, Dict, List, Optional

import requests
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field, model_validator


class ChatMoondream(BaseChatModel):
    """Chat model for Moondream API."""

    api_key: Optional[str] = Field(default=None)
    base_url: str = Field(default="https://api.moondream.ai/v1")
    streaming: bool = Field(default=False)
    timeout: Optional[float] = Field(default=None)

    model_name: str = Field(default="moondream-v1")
    temperature: float = Field(default=0.7)

    # For LangChain compatibility
    client: Optional[Any] = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def validate_environment(self) -> "ChatMoondream":
        """Validate that api key exists in environment."""
        if self.api_key is None:
            api_key = os.getenv("MOONDREAM_API_KEY")
            if api_key is None:
                raise ValueError(
                    "Moondream API key must be provided as an argument or "
                    "set as environment variable 'MOONDREAM_API_KEY'"
                )
            self.api_key = api_key
        return self

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_name": self.model_name,
            "base_url": self.base_url,
            "streaming": self.streaming,
        }

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "moondream"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a chat response."""
        if not messages:
            raise ValueError("No messages provided.")

        # Extract image from messages if present
        image_base64 = None
        question = None

        for message in messages:
            if isinstance(message, HumanMessage):
                if isinstance(message.content, str):
                    question = message.content
                elif isinstance(message.content, list):
                    for content_item in message.content:
                        if isinstance(content_item, dict):
                            if content_item.get("type") == "text":
                                question = content_item.get("text", "")
                            elif content_item.get("type") == "image_url":
                                image_url = content_item.get(
                                    "image_url", {}
                                ).get("url", "")
                                if image_url.startswith("data:image"):
                                    # Image is already base64 encoded
                                    image_base64 = image_url
                                else:
                                    # This is a file path or URL,
                                    # we need to handle it
                                    raise ValueError(
                                        "Non-base64 image URLs not supported yet. "  # noqa
                                        "Please use base64 encoded images."
                                    )

        if not image_base64 or not question:
            raise ValueError(
                "Moondream requires both an image and a question in the message."  # noqa
            )

        # Prepare API request
        headers = {
            "X-Moondream-Auth": self.api_key,
            "Content-Type": "application/json",
        }

        data = {
            "image_url": image_base64,
            "question": question,
            "stream": self.streaming,
        }

        url = f"{self.base_url}/query"

        # Send request to the API
        try:
            response = requests.post(
                url, headers=headers, json=data, timeout=self.timeout
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error calling Moondream API: {e}")

        if self.streaming:
            # Handle streaming response
            text = ""
            if run_manager:
                for line in response.iter_lines():
                    if line:
                        line_text = line.decode("utf-8")
                        if line_text.startswith("data:"):
                            json_str = line_text[5:].strip()
                            try:
                                chunk_data = json.loads(json_str)
                                chunk = chunk_data.get("chunk", "")
                                text += chunk
                                if run_manager and chunk:
                                    run_manager.on_llm_new_token(chunk)
                            except json.JSONDecodeError:
                                continue
        else:
            # Handle regular response
            json_response = response.json()
            text = json_response.get("answer", "")
            request_id = json_response.get("request_id", "")

        # Create ChatGeneration and ChatResult
        message = AIMessage(content=text)
        generation = ChatGeneration(message=message)

        # Include some metadata about the API call
        llm_output = {
            "token_usage": {
                "prompt_tokens": 0,  # Moondream doesn't provide token info
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "model_name": self.model_name,
            "request_id": request_id if "request_id" in locals() else None,
        }

        return ChatResult(generations=[generation], llm_output=llm_output)
