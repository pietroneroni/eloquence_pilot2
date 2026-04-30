# llm_online.py
from __future__ import annotations

import json
from typing import Any, Iterator, List, Optional

import requests
from pydantic import Field
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pathlib import Path

class CustomOnlineLLM(BaseChatModel):
    """
    Custom LLM wrapper for an HTTP endpoint that returns JSON with a "text" field.
    Designed for LangChain's BaseChatModel interface.
    """

    url: str = Field(..., description="URL endpoint of the online model")
    llm_name: str = Field("GPT-3.5", description="Model name")
    task_config: str = Field("Basic", description="Task configuration")
    docs_k: int = Field(5, description="Number of documents")
    temp: float = Field(0.7, description="Temperature")
    top_p: float = Field(0.9, description="Top P")
    max_tokens: int = Field(100, description="Maximum tokens")
    index_name: str = Field("Eloquence", description="Index name")
    retriever_address: str = Field("public", description="Retriever address")
    default_system_prompt: str = Field(
        "You are a helpful assistant.",
        description="Default system prompt",
    )
    timeout_s: float = Field(60.0, description="HTTP timeout (seconds)")

    @property
    def _llm_type(self) -> str:
        return "custom_online_llm"

    def _convert_messages_to_history(self, messages: List[BaseMessage]) -> List[List[str]]:
        """
        Convert LangChain messages to the [ [human, ai], ... ] history format expected by the API.
        System messages are handled separately.
        """
        history: List[List[str]] = []
        pending_human: Optional[str] = None

        for msg in messages:
            if isinstance(msg, HumanMessage):
                pending_human = msg.content
            elif isinstance(msg, AIMessage):
                if pending_human is not None:
                    history.append([pending_human, msg.content])
                    pending_human = None
            else:
                # System message or other types ignored here
                pass

        return history

    def _get_system_prompt(self, messages: List[BaseMessage]) -> str:
        for msg in messages:
            if isinstance(msg, SystemMessage):
                return msg.content
        return self.default_system_prompt

    def _get_last_human_message(self, messages: List[BaseMessage]) -> str:
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                return msg.content
        return ""

    def _post(self, body_data: dict) -> dict:
        """
        Execute the HTTP request and return the parsed JSON response.
        Raises a clear exception with status code and response text on failure.
        """
        post_data = {"body": json.dumps(body_data)}
        try:
            resp = requests.post(self.url, data=post_data, timeout=self.timeout_s)
            resp.raise_for_status()
        except requests.RequestException as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            text = getattr(getattr(e, "response", None), "text", "")
            raise RuntimeError(f"Error calling online LLM (status={status}): {e}. Response: {text[:500]}") from e

        try:
            return resp.json()
        except ValueError as e:
            raise RuntimeError(f"Online LLM did not return valid JSON. First 500 chars: {resp.text[:500]}") from e

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        history = self._convert_messages_to_history(messages)
        system_prompt = self._get_system_prompt(messages)
        input_text = self._get_last_human_message(messages)

        body_data = {
            "history": history,
            "input_text": input_text,
            "llm_name": self.llm_name,
            "task_config": self.task_config,
            "docs_k": self.docs_k,
            "temp": self.temp,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "index_name": self.index_name,
            "retriever_address": self.retriever_address,
            "system_prompt": system_prompt,
        }

        response_json = self._post(body_data)
        generated_text = response_json.get("text", "") or ""

        generation = ChatGeneration(message=AIMessage(content=generated_text))
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatResult]:
        # The upstream endpoint is not streamed; we expose a single-chunk iterator.
        yield self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def configure_online_llm(filepath: str | None = None) -> CustomOnlineLLM:
    if filepath is None:
        project_root = Path(__file__).resolve().parents[1]
        filepath = str(project_root / "configuration_data" / "config_online_llm.json")

    print(f"ONLINE MODEL CONFIGURATION from {filepath}...")

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            config_params = json.load(f)
        online_llm = CustomOnlineLLM(**config_params)
        print("Online model configured successfully.")
        return online_llm
    except FileNotFoundError:
        print(f"ERROR: Configuration file not found: {filepath}")
        raise
    except json.JSONDecodeError as e:
        print(f"ERROR: JSON decoding error in file: {filepath}. Details: {e}")
        raise
