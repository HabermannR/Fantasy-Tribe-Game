import os
import time
from typing import Dict, Any, TypeVar, Type, List
from pydantic import BaseModel
from enum import Enum

import anthropic
from openai import OpenAI

T = TypeVar('T', bound=BaseModel)

class LLMProvider(Enum):
    LOCAL = "local"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class Config:
    LLM_PROVIDER: LLMProvider = LLMProvider.ANTHROPIC
    OPENAI_MODEL: str = "gpt-4o-mini"
    LOCAL_URL: str = "http://127.0.0.1:1234/v1"
    ANTHROPIC_MODEL: str = "claude-3-5-sonnet-20240620"

    @classmethod
    def from_env(cls):
        config = cls()
        config.ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
        config.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        return config

def create_json_schema(model: Type[BaseModel]) -> Dict[str, Any]:
    schema = model.model_json_schema()
    return {"type": "json_schema", "json_schema": {"schema": schema}}

class LLMStrategy:
    def make_api_call(self, messages: List[Dict[str, str]], response_model: Type[T], max_tokens: int = 500) -> T:
        raise NotImplementedError


class AnthropicStrategy(LLMStrategy):
    def __init__(self, config: Config):
        self.client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        self.model = config.ANTHROPIC_MODEL

    def make_api_call(self, messages: List[Dict[str, str]], response_model: Type[T], max_tokens: int = 1500) \
            -> Type[T] | None:
        try:
            tool_schema = create_json_schema(response_model)
            start_time = time.time()
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=messages[0]['content'] + " do not use the word Nexus",
                # Use the content of the first message as the system message
                messages=messages[1:],  # Use the remaining messages, excluding the first one
                tools=[{
                    "name": "Tribe_Game",
                    "description": "Fill in game structures using well-structured JSON.",
                    "input_schema": tool_schema['json_schema']['schema']
                }],
                tool_choice={"type": "tool", "name": "Tribe_Game"}
            )
            end_time = time.time()
            print("Time in Anthropic: ", end_time - start_time)
            print("Input tokens: ", response.usage.input_tokens, "Output tokens: ", response.usage.output_tokens)
            return response_model.model_validate(response.content[0].input)

        except Exception as e:
            raise RuntimeError(f"Anthropic API call failed: {str(e)}")


class OpenAIStrategy(LLMStrategy):
    def __init__(self, config: Config):
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.model = config.OPENAI_MODEL

    def make_api_call(self, messages: List[Dict[str, str]], response_model: Type[T],
                      max_tokens: int = 1500) -> Any | None:
        try:
            start_time = time.time()
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=response_model,
            )
            end_time = time.time()
            print("Time in OpenAI: ", end_time - start_time)
            print("Input tokens: ", completion.usage.input_tokens, "Output tokens: ", completion.usage.output_tokens)
            return completion.choices[0].message.parsed
        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {str(e)}")


class LocalModelStrategy(LLMStrategy):
    def __init__(self, config: Config):
        self.client = OpenAI(base_url=config.LOCAL_URL, api_key="not-needed")
        self.model = "local-model"

    def make_api_call(self, messages: List[Dict[str, str]], response_model: Type[T], max_tokens: int = 1500) -> None:
        try:
            start_time = time.time()
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format=create_json_schema(response_model),
                max_tokens=max_tokens
            )
            end_time = time.time()
            print("Time in local LLM: ", end_time - start_time)
            #print("Input tokens: ", completion.usage.input_tokens, "Output tokens: ", completion.usage.output_tokens)
            return response_model.model_validate_json(completion.choices[0].message.content)
        except Exception as e:
            raise RuntimeError(f"Local model API call failed: {str(e)}")


class LLMContext:
    def __init__(self, config: Config):
        self.config = config
        self._strategy = self._get_llm_strategy()

    def update(self, config: Config):
        self.config = config
        self._strategy = self._get_llm_strategy()

    def _get_llm_strategy(self) -> LLMStrategy:
        if self.config.LLM_PROVIDER == LLMProvider.OPENAI:
            return OpenAIStrategy(self.config)
        elif self.config.LLM_PROVIDER == LLMProvider.ANTHROPIC:
            return AnthropicStrategy(self.config)
        else:
            return LocalModelStrategy(self.config)

    def make_api_call(self, messages: List[Dict[str, str]], response_model: Type[T], max_tokens: int = 1500) -> T:
        return self._strategy.make_api_call(messages, response_model, max_tokens)