import os
import time
from typing import Dict, Any, TypeVar, Type, List, Optional, Union
from pydantic import BaseModel
from enum import Enum

from openai import OpenAI
import anthropic

T = TypeVar('T', bound=BaseModel)


class SummaryModel(BaseModel):
    summary: str


class LLMProvider(Enum):
    LOCAL = "local"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class ModelType(Enum):
    STORY = "story"
    SUMMARY = "summary"


class SummaryMode(Enum):
    JSON = "json"
    RAW = "raw"


class ModelConfig:
    def __init__(self,
                 provider: LLMProvider,
                 model_name: str,
                 local_url: Optional[str] = None):
        self.provider = provider
        self.model_name = model_name
        self.local_url = local_url


class SummaryModelConfig(ModelConfig):
    def __init__(self,
                 provider: LLMProvider,
                 model_name: str,
                 mode: SummaryMode = SummaryMode.JSON,
                 local_url: Optional[str] = None):
        super().__init__(provider, model_name, local_url)
        self.mode = mode


class Config:
    def __init__(self,
                 story_config: ModelConfig,
                 summary_config: SummaryModelConfig):
        self.story_config = story_config
        self.summary_config = summary_config
        self.ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

    @classmethod
    def default_config(cls):
        return cls(
            story_config=ModelConfig(
                provider=LLMProvider.ANTHROPIC,
                model_name="claude-3-5-sonnet-20240620"
            ),
            summary_config=SummaryModelConfig(
                provider=LLMProvider.OPENAI,
                model_name="gpt-4o-mini",
                mode=SummaryMode.JSON
            )
        )

    @classmethod
    def local_config(cls, summary_mode: SummaryMode = SummaryMode.JSON):
        return cls(
            story_config=ModelConfig(
                provider=LLMProvider.LOCAL,
                model_name="Qwen2.5-14B-Instruct:latest",
                local_url="http://127.0.0.1:1234/v1"
            ),
            summary_config=SummaryModelConfig(
                provider=LLMProvider.LOCAL,
                model_name="Qwen1.5-7B-Chat:latest",
                mode=summary_mode,
                local_url="http://127.0.0.1:1235/v1"
            )
        )


def create_json_schema(model: Type[BaseModel]) -> Dict[str, Any]:
    schema = model.model_json_schema()
    return {"type": "json_schema", "json_schema": {"schema": schema}}


class LLMStrategy:
    def __init__(self, model_config: ModelConfig, api_key: str = ""):
        self.model_config = model_config
        self.api_key = api_key

    def make_api_call(self, messages: List[Dict[str, str]], response_model: Optional[Type[T]] = None, max_tokens: Optional[int] = None) -> Union[T, str]:
        raise NotImplementedError


class AnthropicStrategy(LLMStrategy):
    def __init__(self, model_config: ModelConfig, api_key: str):
        super().__init__(model_config, api_key)
        self.client = anthropic.Anthropic(api_key=api_key)

    def make_api_call(self, messages: List[Dict[str, str]], response_model: Optional[Type[T]] = None,
                      max_tokens: Optional[int] = 1500) -> Union[T, str]:
        try:
            start_time = time.time()
            if isinstance(self.model_config, SummaryModelConfig) and self.model_config.mode == SummaryMode.RAW:
                response = self.client.messages.create(
                    model=self.model_config.model_name,
                    max_tokens=max_tokens,
                    system=messages[0]['content'],
                    messages=messages[1:]
                )
                result = response.content[0].text
            else:
                tool_schema = create_json_schema(response_model)
                response = self.client.messages.create(
                    model=self.model_config.model_name,
                    max_tokens=max_tokens,
                    system=messages[0]['content'],
                    messages=messages[1:],
                    tools=[{
                        "name": "Tribe_Game",
                        "description": "Fill in game structures using well-structured JSON.",
                        "input_schema": tool_schema['json_schema']['schema']
                    }],
                    tool_choice={"type": "tool", "name": "Tribe_Game"}
                )
                result = response_model.model_validate(response.content[0].input)

            end_time = time.time()
            print(f"Time in Anthropic ({self.model_config.model_name}): ", end_time - start_time)
            print("Input tokens: ", response.usage.input_tokens, "Output tokens: ", response.usage.output_tokens)
            return result

        except Exception as e:
            raise RuntimeError(f"Anthropic API call failed: {str(e)}")


class OpenAIStrategy(LLMStrategy):
    def __init__(self, model_config: ModelConfig, api_key: str):
        super().__init__(model_config, api_key)
        self.client = OpenAI(api_key=api_key)

    def make_api_call(self, messages: List[Dict[str, str]], response_model: Optional[Type[T]] = None,
                      max_tokens: Optional[int] = 1500) -> Union[T, str]:
        try:
            start_time = time.time()
            if isinstance(self.model_config, SummaryModelConfig) and self.model_config.mode == SummaryMode.RAW:
                completion = self.client.chat.completions.create(
                    model=self.model_config.model_name,
                    messages=messages,
                    max_tokens=max_tokens
                )
                result = completion.choices[0].message.content
            else:
                completion = self.client.beta.chat.completions.parse(
                    model=self.model_config.model_name,
                    messages=messages,
                    response_format=response_model,
                    max_tokens=max_tokens
                )
                result = completion.choices[0].message.parsed

            end_time = time.time()
            print(f"Time in OpenAI ({self.model_config.model_name}): ", end_time - start_time)
            print("Input tokens: ", completion.usage.prompt_tokens, "Output tokens: ",
                  completion.usage.completion_tokens)
            return result
        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {str(e)}")


class LocalModelStrategy(LLMStrategy):
    def __init__(self, model_config: ModelConfig, api_key: str = "not-needed"):
        super().__init__(model_config, api_key)
        self.client = OpenAI(base_url=model_config.local_url, api_key=api_key)

    def make_api_call(self, messages: List[Dict[str, str]], response_model: Optional[Type[T]] = None,
                      max_tokens: Optional[int] = 1500) -> Union[T, str]:
        try:
            start_time = time.time()
            if isinstance(self.model_config, SummaryModelConfig) and self.model_config.mode == SummaryMode.RAW:
                completion = self.client.chat.completions.create(
                    model=self.model_config.model_name,
                    messages=messages,
                    max_tokens=max_tokens
                )
                result = completion.choices[0].message.content
            else:
                completion = self.client.chat.completions.create(
                    model=self.model_config.model_name,
                    messages=messages,
                    response_format=create_json_schema(response_model),
                    max_tokens=max_tokens
                )
                result = response_model.model_validate_json(completion.choices[0].message.content)

            end_time = time.time()
            print(f"Time in local LLM ({self.model_config.model_name}): ", end_time - start_time)
            return result
        except Exception as e:
            raise RuntimeError(f"Local model API call failed: {str(e)}")


class LLMContext:
    def __init__(self, config: Config):
        self.config = config
        self._story_strategy = self._create_strategy(config.story_config)
        self._summary_strategy = self._create_strategy(config.summary_config)

    def update(self, config: Config):
        self.config = config
        self._story_strategy = self._create_strategy(config.story_config)
        self._summary_strategy = self._create_strategy(config.summary_config)

    def _create_strategy(self, model_config: ModelConfig) -> LLMStrategy:
        if model_config.provider == LLMProvider.ANTHROPIC:
            return AnthropicStrategy(model_config, self.config.ANTHROPIC_API_KEY)
        elif model_config.provider == LLMProvider.OPENAI:
            return OpenAIStrategy(model_config, self.config.OPENAI_API_KEY)
        else:  # LOCAL
            return LocalModelStrategy(model_config)

    def make_api_call(self, messages: List[Dict[str, str]], response_model: Optional[Type[T]] = None,
                      max_tokens: Optional[int] = None) -> Union[T, str]:
        if response_model == SummaryModel:
            strategy = self._summary_strategy
        else:
            strategy = self._story_strategy

        return strategy.make_api_call(messages, response_model, max_tokens)

    def make_story_call(self, messages: List[Dict[str, str]], response_model: Type[T],
                        max_tokens: Optional[int] = 1500) -> T:
        return self._story_strategy.make_api_call(messages, response_model, max_tokens)

    def make_summary_call(self, messages: List[Dict[str, str]], response_model: Optional[Type[T]] = None,
                          max_tokens: Optional[int] = 1500) -> Union[T, str]:
        return self._summary_strategy.make_api_call(messages, response_model, max_tokens)

    def get_summary(self, context: str,
                    max_tokens: Optional[int] = 1500) -> str:
        """
        Get a summary of the provided context using the summary model configuration.
        Returns either a string (RAW mode) or SummaryModel (JSON mode)
        """
        messages = [
            {
                "role": "system",
                "content": "Take this text from a fantasy setting and provide a summary of its content"
            },
            {
                "role": "user",
                "content": f"Summarize this content: {context}"
            }
        ]

        if isinstance(self.config.summary_config, SummaryModelConfig) and \
                self.config.summary_config.mode == SummaryMode.JSON:
            return self.make_summary_call(messages, SummaryModel, max_tokens).summary
        else:
            return self.make_summary_call(messages, max_tokens)