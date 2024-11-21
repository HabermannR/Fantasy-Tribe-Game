"""
Game Name: Fantasy Tribe Game
Description: Play a fantasy tribe from humble beginnings to earth shattering godlike power thanks to LLMs

GitHub: https://github.com/HabermannR/Fantasy-Tribe-Game

License: MIT License

Copyright (c) 2024 HabermannR

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import time
from typing import Dict, Any, TypeVar, Type, List, Optional, Union, Generic
from pydantic import BaseModel
from typing_extensions import TypedDict
from enum import Enum
from Language import Language

from openai import OpenAI
import anthropic
import json


T = TypeVar('T')

class SummaryModel(BaseModel):
    summary: str

class SummaryModelDict(TypedDict):
    summary: str

class LLMProvider(Enum):
    LOCAL = "local"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


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
                 summary_config: SummaryModelConfig,
                 language: Language = "english"):
        self.story_config = story_config
        self.summary_config = summary_config
        self.language = language
        self.ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        self.GEMINI_API_KEY = os.getenv("GEMINI_KEY", "")

    @classmethod
    def default_config(cls, language: Language = "english"):
        return cls(
            story_config=ModelConfig(
                provider=LLMProvider.ANTHROPIC,
                model_name="claude-3-5-sonnet-20241022"
            ),
            summary_config=SummaryModelConfig(
                provider=LLMProvider.OPENAI,
                model_name="gpt-4o-mini",
                mode=SummaryMode.JSON
            ),
            language=language
        )

    @classmethod
    def local_config(cls,
                     summary_mode: SummaryMode = SummaryMode.JSON,
                     language: Language = "english"):
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
            ),
            language=language
        )


def create_json_schema(model: Type[BaseModel]) -> Dict[str, Any]:
    schema = model.model_json_schema()

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "test_schema",
            "strict": True,
            "schema": schema,
        },
    }


class SystemPrompts:
    ENGLISH_PROMPT = """"""

    GERMAN_PROMPT = """. When generating JSON responses, keep the field names in English but provide the content/values in German.
For example:
{
    "summary": "Der Hauptcharakter beginnt seine Reise...",
    "mainCharacters": ["König Arthur", "Merlin", "Guinevere"],
    "setting": "Mittelalterliches England"
}"""

    @staticmethod
    def get_prompt(language: Language) -> str:
        if language == "english":
            return SystemPrompts.ENGLISH_PROMPT
        elif language == "german":
            return SystemPrompts.GERMAN_PROMPT
        else:
            raise ValueError(f"Unsupported language: {language}")

    @staticmethod
    def inject_system_prompt(messages: List[Dict[str, str]], language: Language) -> List[Dict[str, str]]:
        system_prompt = SystemPrompts.get_prompt(language)

        # If there is a system message, replace it while maintaining order
        return [
            {"role": "system", "content": msg["content"] + system_prompt} if msg["role"] == "system"
            else msg
            for msg in messages
        ]


class LLMStrategy:
    def __init__(self, model_config: ModelConfig, api_key: str = ""):
        self.model_config = model_config
        self.api_key = api_key

    def make_api_call(
        self,
        messages: List[Dict[str, str]],
        response_types: T = None,
        max_tokens: Optional[int] = None
    ) -> Union[T, str]:
        raise NotImplementedError


class AnthropicStrategy(LLMStrategy):
    def __init__(self, model_config: ModelConfig, api_key: str):
        super().__init__(model_config, api_key)
        self.client = anthropic.Anthropic(api_key=api_key)

    def make_api_call(
            self,
            messages: List[Dict[str, str]],
            response_types: T = None,
            max_tokens: Optional[int] = None
    ) -> Union[T, str]:
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
                tool_schema = create_json_schema(response_types)
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
                result = response_types.model_validate(response.content[0].input)

            end_time = time.time()
            print(f"Time in Anthropic ({self.model_config.model_name}): ", end_time - start_time)
            print("Input tokens: ", response.usage.input_tokens, "Output tokens: ", response.usage.output_tokens)
            return result

        except Exception as e:
            raise RuntimeError(f"Anthropic API call failed: {str(e)}")

class GeminiStrategy(LLMStrategy):
    def __init__(self, model_config: ModelConfig, api_key: str):
        super().__init__(model_config, api_key)
        #genai.configure(api_key=api_key)  # alternative API key configuration
        self.client = OpenAI(
            api_key=os.environ["GEMINI_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/"
        )

    def make_api_call(
            self,
            messages: List[Dict[str, str]],
            response_types: T = None,
            max_tokens: Optional[int] = None
    ) -> Union[T, str]:
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
                tool_schema = create_json_schema(response_types)
                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": "Tribe_Game",
                            "description": "Create game structures",
                            "parameters": tool_schema['json_schema']['schema']
                        }
                    }
                ]
                response = self.client.chat.completions.create(
                    model=self.model_config.model_name,
                    messages=messages[1:],
                    tools=tools,
                    tool_choice={"type": "tool", "name": "Tribe_Game"}
                )
                result = response_types.model_validate(
                    json.loads(response.choices[0].message.tool_calls[0].function.arguments))

            end_time = time.time()
            print(f"Time in Gemini ({self.model_config.model_name}): ", end_time - start_time)
            #print("Input tokens: ", response.usage.input_tokens, "Output tokens: ", response.usage.output_tokens)
            return result

        except Exception as e:
            raise RuntimeError(f"Gemini API call failed: {str(e)}")


class OpenAIStrategy(LLMStrategy):
    def __init__(self, model_config: ModelConfig, api_key: str):
        super().__init__(model_config, api_key)
        self.client = OpenAI(api_key=api_key)

    def make_api_call(
            self,
            messages: List[Dict[str, str]],
            response_types: T = None,
            max_tokens: Optional[int] = None
    ) -> Union[T, str]:
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
                    response_format=response_types,
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

    def make_api_call(
            self,
            messages: List[Dict[str, str]],
            response_types: T = None,
            max_tokens: Optional[int] = None
    ) -> Union[T, str]:
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
                    response_format=create_json_schema(response_types),
                    max_tokens=max_tokens
                )
                result = response_types.model_validate_json(completion.choices[0].message.content)
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
        elif model_config.provider == LLMProvider.GEMINI:
            return GeminiStrategy(model_config, self.config.GEMINI_API_KEY)
        else:  # LOCAL
            return LocalModelStrategy(model_config)

    def make_api_call(
        self,
        messages: List[Dict[str, str]],
        response_types: T = None,
        max_tokens: Optional[int] = None
    ) -> Union[T, str]:
        if response_types and isinstance(response_types, type(SummaryModel)):
            strategy = self._summary_strategy
        else:
            strategy = self._story_strategy

        return strategy.make_api_call(messages, response_types, max_tokens)

    def make_story_call(
        self,
        messages: List[Dict[str, str]],
        response_types: T = None,
        max_tokens: Optional[int] = 1500
    ) -> T:
        messages = SystemPrompts.inject_system_prompt(messages, self.config.language)
        return self._story_strategy.make_api_call(messages, response_types, max_tokens)

    def make_summary_call(self,
        messages: List[Dict[str, str]],
        response_types: T,
        max_tokens: Optional[int] = 1500
    ) -> T:
        return self._summary_strategy.make_api_call(messages, response_types, max_tokens)

    def get_summary(self, context: str,
                    max_tokens: Optional[int] = 1500) -> str:
        """
        Get a summary of the provided context using the summary model configuration.
        Returns either a string (RAW mode) or SummaryModel (JSON mode)
        """
        systemprompt = {
            "english": "Take this text from a fantasy setting and rephrase its content more fluidly, while shortening it a bit",
            "german": "Take this text from a fantasy setting and rephrase its content more fluidly in German, while shortening it a bit"
        }.get(self.config.language)
        messages = [
            {
                "role": "system",
                "content": systemprompt
            },
            {
                "role": "user",
                "content": f"Rephrase this content: {context}"
            }
        ]

        if isinstance(self.config.summary_config, SummaryModelConfig) and \
                self.config.summary_config.mode == SummaryMode.JSON:
            return self.make_summary_call(messages, SummaryModel, max_tokens).summary
        else:
            return self.make_summary_call(messages, max_tokens)
