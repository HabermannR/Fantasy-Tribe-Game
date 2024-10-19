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

import json
import os
import random
import textwrap
import copy
from collections import defaultdict
from difflib import SequenceMatcher
from datetime import datetime
from enum import Enum
from typing import Dict, Any, TypeVar, Type, List, Tuple, Optional

import anthropic
import gradio as gr
from openai import OpenAI
from pydantic import BaseModel, Field

try:
    anthropicKey = os.environ["ANTHROPIC_API_KEY"]
except KeyError:
    print("ANTHROPIC_API_KEY not found in environment variables")
    anthropicKey = ""
try:
    openAIKey = os.environ.get("OPENAI_API_KEY")
except KeyError:
    print("OPENAI_API_KEY not found in environment variables")
    openAIKey = ""

T = TypeVar('T', bound=BaseModel)


class LLMProvider(Enum):
    LOCAL = "local"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class Config:
    LLM_PROVIDER: LLMProvider = LLMProvider.LOCAL
    openai_model: str = "gpt-4o-mini"
    local_url: str = "http://127.0.0.1:1234/v1"


class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return {"__enum__": str(obj)}
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        return json.JSONEncoder.default(self, obj)


def as_enum(d):
    if "__enum__" in d:
        name, member = d["__enum__"].split(".")
        return getattr(globals()[name], member)
    return d


def convert_enums(data):
    if isinstance(data, dict):
        if "__enum__" in data:
            enum_str = data["__enum__"]
            enum_class, enum_value = enum_str.split(".")
            return getattr(globals()[enum_class], enum_value)
        return {k: convert_enums(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_enums(item) for item in data]
    return data


def create_json_schema(model: Type[BaseModel]) -> Dict[str, Any]:
    """Create a JSON schema from a Pydantic model."""
    schema = model.model_json_schema()
    return {
        "type": "json_schema",
        "json_schema": {
            "schema": schema
        }
    }


class LLMStrategy:
    def make_api_call(self, messages: List[Dict[str, str]], response_model: Type[T], max_tokens: int = 500) -> T:
        raise NotImplementedError


class AnthropicStrategy(LLMStrategy):
    def __init__(self):
        self.client = anthropic.Anthropic(
            api_key=anthropicKey)
        self.model = "claude-3-5-sonnet-20240620"

    def make_api_call(self, messages: List[Dict[str, str]], response_model: Type[T], max_tokens: int = 1500) \
            -> Type[T] | None:
        try:
            tool_schema = create_json_schema(response_model)
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
            return response_model.model_validate(response.content[0].input)
        except Exception as e:
            raise RuntimeError(f"Anthropic API call failed: {str(e)}")


class OpenAIStrategy(LLMStrategy):
    def __init__(self):
        self.client = OpenAI(api_key=openAIKey)
        self.model = config.openai_model

    def make_api_call(self, messages: List[Dict[str, str]], response_model: Type[T],
                      max_tokens: int = 1500) -> Any | None:
        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=response_model,
            )
            return completion.choices[0].message.parsed
        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {str(e)}")


class LocalModelStrategy(LLMStrategy):
    def __init__(self):
        self.client = OpenAI(
            base_url=config.local_url,
            api_key="not-needed"
        )
        self.model = "local-model"

    def make_api_call(self, messages: List[Dict[str, str]], response_model: Type[T], max_tokens: int = 1500) -> None:
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format=create_json_schema(response_model),
                max_tokens=max_tokens
            )
            return response_model.model_validate_json(completion.choices[0].message.content)
        except Exception as e:
            raise RuntimeError(f"Local model API call failed: {str(e)}")


class LLMContext:
    def __init__(self, strategy: LLMStrategy):
        self._strategy = strategy

    def make_api_call(self, messages: List[Dict[str, str]], response_model: Type[T], max_tokens: int = 1500) -> T:
        return self._strategy.make_api_call(messages, response_model, max_tokens)


def get_llm_strategy(provider: LLMProvider) -> LLMStrategy:
    if provider == LLMProvider.OPENAI:
        return OpenAIStrategy()
    elif provider == LLMProvider.ANTHROPIC:
        return AnthropicStrategy()
    else:
        return LocalModelStrategy()


config = Config()
# Use the function to initialize the LLMContext
llm_context = LLMContext(get_llm_strategy(config.LLM_PROVIDER))


def make_api_call(messages: List[Dict[str, str]], response_model: Type[T], max_tokens: int = 1500) -> T:
    return llm_context.make_api_call(messages, response_model, max_tokens)


def get_string_similarity(a, b):
    """Calculate similarity ratio between two strings."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


class OutcomeType(Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class ActionChoice(BaseModel):
    caption: str
    description: str
    probability: float


class NextChoices(BaseModel):
    choices: List[ActionChoice]


class NextReactionChoices(BaseModel):
    choices: List[ActionChoice]
    situation: str


class Relationship(BaseModel):
    target: str
    type: str
    opinion: int  # -100 to 100


class Character(BaseModel):
    name: str
    title: str
    relationships: Optional[List[Relationship]]


class ForeignCharacter(Character):
    tribe: str


class DiplomaticStance(Enum):
    PEACEFUL = "peaceful"
    NEUTRAL = "neutral"
    AGGRESSIVE = "aggressive"


class DevelopmentType(Enum):
    MAGICAL = "magical"
    HYBRID = "hybrid"
    PRACTICAL = "practical"


class DiplomaticStatus(Enum):
    ALLY = "ally"
    NEUTRAL = "neutral"
    ENEMY = "enemy"


class TribeType(BaseModel):
    name: str
    description: str
    development: DevelopmentType
    stance: DiplomaticStance

class ForeignTribeType(TribeType):
    diplomatic_status: DiplomaticStatus
    leaders: List[Character]

class InitialChoices(BaseModel):
    choices: List[TribeType]


class GameStateBase(BaseModel):
    tribe: TribeType
    leaders: List[Character]
    foreign_tribes: List[ForeignTribeType]
    situation: str
    event_result: str
    event_result_short: str


class GameState(GameStateBase):
    previous_situation: Optional[str] = None
    current_action_choices: Optional[NextChoices] = None
    last_action_sets: List[List[str]] = Field(default_factory=list)
    chosen_action: ActionChoice = None
    last_outcome: Optional[OutcomeType] = None
    turn: int = 1

    class Config:
        arbitrary_types_allowed = True

    def add_action_set(self, actions: List[str]):
        self.last_action_sets.append(actions)
        if len(self.last_action_sets) > 3:  # Keep only the last 3 sets
            self.last_action_sets.pop(0)

    def to_context_string(self) -> str:
        def get_sentiment(opinion: int) -> str:
            if opinion >= 90:
                return "devoted"
            elif opinion >= 70:
                return "strongly positive"
            elif opinion >= 40:
                return "favorable"
            elif opinion >= 10:
                return "somewhat positive"
            elif opinion > -10:
                return "neutral"
            elif opinion >= -40:
                return "somewhat negative"
            elif opinion >= -70:
                return "unfavorable"
            elif opinion >= -90:
                return "strongly negative"
            else:
                return "hostile"

        # Get tribe orientation
        tribe_type = get_tribe_orientation(self.tribe.development, self.tribe.stance)

        # Build main tribe section
        text = f"""{self.tribe.name}
{tribe_type}

{self.tribe.description}

Leaders:"""

        # Format all leaders with consistent formatting and their relationships
        for leader in self.leaders:
            text += f"\n  • {leader.title}: {leader.name}"
            if leader.relationships:
                text += "\n    Relationships:"
                for rel in leader.relationships:
                    sentiment = get_sentiment(rel.opinion)
                    text += f"\n      - {rel.type} relationship with {rel.target} ({sentiment}, {rel.opinion})"

        # Add foreign tribes section if there are any
        if self.foreign_tribes:
            text += "\n\nForeign Tribes:"

            # Create a dictionary to group tribes by diplomatic status
            status_order = [DiplomaticStatus.ALLY, DiplomaticStatus.NEUTRAL, DiplomaticStatus.ENEMY]
            tribes_by_status = {status: [] for status in status_order}

            # Group tribes by their diplomatic status
            for tribe in self.foreign_tribes:
                tribes_by_status[tribe.diplomatic_status].append(tribe)

            # Add tribes in order of diplomatic status
            for status in status_order:
                if tribes_by_status[status]:
                    # Add status header
                    text += f"\n\n{status.value.title()}:"

                    # Add each tribe in this status group
                    for foreign_tribe in tribes_by_status[status]:
                        foreign_type = get_tribe_orientation(
                            foreign_tribe.development,
                            foreign_tribe.stance
                        )

                        text += f"\n\n{foreign_tribe.name}"
                        text += f"\n  {foreign_type}"
                        text += f"\n{foreign_tribe.description}"

                        # Add foreign tribe leaders and their relationships
                        if foreign_tribe.leaders:
                            text += "\nLeaders:"
                            for leader in foreign_tribe.leaders:
                                text += f"\n  • {leader.title}: {leader.name}"
                                if leader.relationships:
                                    text += "\n    Relationships:"
                                    for rel in leader.relationships:
                                        sentiment = get_sentiment(rel.opinion)
                                        text += f"\n      - {rel.type} relationship with {rel.target} ({sentiment}, {rel.opinion})"

        return text

    @staticmethod
    def get_recent_history() -> str:
        global history
        formatted_events = []
        if len(history) > 0:
            for event in reversed(history):
                turn_header = f"=== Turn {event.turn} ==="
                formatted_events.extend([turn_header, event.event_result])
        return "\n\n".join(formatted_events)

    @staticmethod
    def get_recent_short_history(num_events: int = 5) -> str:
        global history
        # Calculate the starting index from the end
        start_idx = 1 + num_events

        # Get the slice of events we want, in reverse order
        recent_events = history[-start_idx:-1][-num_events:]
        return "\n\n".join([
                f"- {event.event_result_short } (Outcome: {OutcomeType(event.last_outcome).name if event.last_outcome else 'N/A'})"
                for event in recent_events])


    @classmethod
    def initialize(cls, tribe: TribeType, leader: Character) -> 'GameState':
        return cls(
            tribe=tribe,
            leaders=[leader, ],
            foreign_tribes=[],
            situation="",
            event_result="",
            event_result_short=""
        )

    def update(self, new_state: GameStateBase):
        for field in GameStateBase.model_fields:
            setattr(self, field, getattr(new_state, field))


current_game_state: Optional[GameState] = None
global history
history: List[GameState] = []


def generate_tribe_choices() -> InitialChoices:
    # Generate 3 unique combinations of development type and diplomatic stance
    # combinations: List[Tuple[DevelopmentType, DiplomaticStance]] = []
    possible_combinations = [
        (dev, stance)
        for dev in DevelopmentType
        for stance in DiplomaticStance
    ]

    selected_combinations = random.sample(possible_combinations, 3)

    # Get orientation descriptions for each combination
    orientations = [
        get_tribe_orientation(dev_type, stance)
        for dev_type, stance in selected_combinations
    ]

    # Construct the prompt with the specific requirements for each tribe
    messages = [
        {
            "role": "system",
            "content": "You are the game master for a text based strategic game, where the player rules over a tribe in a fantasy based setting. Output your answers as JSON"
        },
        {
            "role": "user",
            "content": f"""Create three unique tribes with the following specifications:

Tribe 1: 
- Must be {selected_combinations[0][0].value} in development and {selected_combinations[0][1].value} in diplomacy
- Should fit the description: {orientations[0]}

Tribe 2:
- Must be {selected_combinations[1][0].value} in development and {selected_combinations[1][1].value} in diplomacy
- Should fit the description: {orientations[1]}

Tribe 3:
- Must be {selected_combinations[2][0].value} in development and {selected_combinations[2][1].value} in diplomacy
- Should fit the description: {orientations[2]}

For each tribe provide:
1. A unique tribe name (do not use 'Sylvan')
2. A description of the tribe (without mentioning the tribe name, and do not quote the given description directly)
"""
        }
    ]

    choices = make_api_call(messages, InitialChoices, max_tokens=2000)

    if choices:
        print("\n=== Available Tribe Choices ===")
        for number, (choice, (dev_type, stance)) in enumerate(zip(choices.choices, selected_combinations)):
            tribe_type = get_tribe_orientation(dev_type, stance)
            print(f"\n{number}. {choice.name}, {tribe_type}")
            print(textwrap.fill(f"Description: {choice.description}", width=80, initial_indent="   ",
                                subsequent_indent="   "))

    return choices

def get_leader(chosen_tribe: TribeType) -> Character:
    messages = [
        {
            "role": "system",
            "content": "You are the game master for a text based strategic game, where the player rules over a tribe in a fantasy based setting. Output your answers as JSON"
        },
        {
            "role": "user",
            "content": f"""Give me the name and title for the leader of {chosen_tribe.name}. Their description is:
{chosen_tribe.description}"""
        }
    ]

    leader = make_api_call(messages, Character, max_tokens=2000)

    if leader:
        print(textwrap.fill(f"\nLeader: {leader}", width=80, initial_indent="   ",
                            subsequent_indent="   "))

    return leader

def get_tribe_orientation(development: DevelopmentType, stance: DiplomaticStance) -> str:
    """
    Get the orientation-specific description for a tribe based on their development path and diplomatic stance.

    Args:
        development (Development): The tribe's development focus (magical, hybrid, or practical)
        stance (DiplomaticStance): The tribe's diplomatic stance
    """
    if development == DevelopmentType.MAGICAL:
        if stance == DiplomaticStance.PEACEFUL:
            return "Mystic sages who commune with natural forces"
        elif stance == DiplomaticStance.NEUTRAL:
            return "Pragmatic mages who balance power and wisdom"
        else:  # AGGRESSIVE
            return "Battle-mages who harness destructive magic"
    elif development == DevelopmentType.HYBRID:
        if stance == DiplomaticStance.PEACEFUL:
            return "Technomancers who blend science and spirituality"
        elif stance == DiplomaticStance.NEUTRAL:
            return "Arcane engineers who merge magic with technology"
        else:  # AGGRESSIVE
            return "Magitech warriors who combine spell and steel"
    else:  # PRACTICAL
        if stance == DiplomaticStance.PEACEFUL:
            return "Builders who focus on technological advancement"
        elif stance == DiplomaticStance.NEUTRAL:
            return "Innovators who balance progress with strength"
        else:  # AGGRESSIVE
            return "Warriors who excel in military innovation"




def generate_choices(game_state: GameState, choice_type: str) -> NextReactionChoices:
    recent_history = game_state.get_recent_short_history(num_events=5)
    probability_adjustment = get_probability_adjustment(game_state.last_outcome)

    if game_state.chosen_action:
        action_context = f"""The player chose the following action: {game_state.chosen_action.caption}
{game_state.chosen_action.description}

The outcome of this action was:
{game_state.event_result}"""
    else:
        action_context = ""
    action_instruction = ""
    additional_instructions = ""

    if choice_type == "single_event":
        action_instruction = "create a new event. Add the event description to:"
    elif choice_type == "reaction":
        action_instruction = "create a reaction to this last outcome. Add this reaction to:"
        additional_instructions = "The reaction should include one of the foreign characters of the foreign tribe."
    elif choice_type == "followup":
        action_instruction = "create a follow up to this last outcome. Add this follow up to:"
        additional_instructions = "Keep these choices thematically close to the last action."
    else:
        raise ValueError(f"Invalid choice type: {choice_type}")

    messages = [
        {
            "role": "system",
            "content": "You are the game master for a text based strategic game, where the player rules over a tribe in a fantasy based setting. Output your answers as JSON"
        },
        {
            "role": "user",
            "content": f"""Current game state: 
            Player tribe: {game_state.to_context_string()}

Recent History:
{recent_history}

Last situation: 
{game_state.previous_situation} 

{action_context}

Based on this information, {action_instruction}
{game_state.situation} 
while slightly summarizing it and save it in "situation".

Then present three possible actions the player can take next, possibly utilizing one or more of the tribes characters. 
{additional_instructions}
Include potential consequences and strategic considerations for each action.
{probability_adjustment}
Give one of the choices a quite high probability, and make at least one choice quite aggressive, 
but do not automatically give the aggressive options a low probability. Instead, base the probabilities on the given information about the tribe. """
        }
    ]

    next_choices = make_api_call(messages, NextReactionChoices, max_tokens=2000)
    if next_choices:
        print("\n=== Available Actions ===")
        for i, choice in enumerate(next_choices.choices, 1):
            print(f"\n{i}. {choice.caption}")
            print(textwrap.fill(f"Description: {choice.description}", width=80, initial_indent="   ",
                                subsequent_indent="   "))
            print(f"   Probability of Success: {choice.probability:.2f}")
    return next_choices


def get_probability_adjustment(last_outcome: OutcomeType) -> str:
    if last_outcome in [OutcomeType.NEUTRAL, OutcomeType.NEGATIVE]:
        return "Due to the recent neutral or negative outcome, provide higher probabilities of success for the next choices (between 0.6 and 0.9)."
    return "Provide balanced probabilities of success for the next choices (between 0.5 and 0.8)."


def generate_new_game_state(game_state: GameState) -> GameStateBase:
    recent_history = game_state.get_recent_short_history(num_events=5)
    tribes_prompt = ""
    if game_state.turn > 5 and len(game_state.enemies) < 1:
        tribes_prompt += "Add one or two enemy factions"
    if game_state.turn > 3 and len(game_state.neutrals) < 2:
        tribes_prompt += "Add one or two neutral factions"

    messages = [
        {
            "role": "system",
            "content": "You are the game master for a text-based strategic game, where the player rules over a tribe in a fantasy setting. Your task is to update the game state based on the player's actions and their outcomes."
        },
        {
            "role": "user",
            "content": f"""Current game state: 
            Player tribe: {game_state.to_context_string()}
The tribes development is {game_state.tribe.development.value}, its diplomatic stance is {game_state.tribe.stance.value}.
Change this only for major upheavals!

Recent History:
{recent_history}

Previous Situation: 
{game_state.previous_situation}

Player's Chosen Action: {game_state.chosen_action.caption}
{game_state.chosen_action.description}

Action Outcome: {game_state.last_outcome.name}

Please provide an updated game state given the chosen action and its outcome.
Remember that events typically span multiple turns - this outcome doesn't need to conclude the event.

Required Updates:

1. Tribe and Leadership Changes:
   - Update the tribe's name and description if major events warrant it
   - Modify leader names, titles, add/remove leaders as appropriate and add, update or delete their realtionships to each othera dn to other tribe's leaders.
   - Consider adding special leaders (heroes, generals, mages, shamans)
   - Maximum of 5 leaders total
   - Changes should reflect significant events (alliances, marriages, wars, etc.)
   - Tribe name changes should be rare and tied to momentous events

2. Current Situation:
   - Provide an updated situation field reflecting recent developments

3. Foreign Relations:
   - Update foreign tribes (ForeignTribeType) with:
     * Diplomatic status (ALLY/NEUTRAL/ENEMY)
     * Name and description (do not put the tribe name in the description)
     * Development type and stance
     * Leaders (maximum 2 per tribe) and their relationships
   - Remember that diplomatic changes take time:
     * Multiple turns to move from NEUTRAL to ALLY
     * Even longer to form formal alliances
     * Consider the impact of development types and stances

4. Event Results:
   - Provide a narrative description in event_result
   - Tell the story of what happened in 2-3 paragraphs
   - Focus on narrative impact while maintaining consistency
   - Remember the event may continue in future turns
   - Include a short summary in event_result_short

Be creative but ensure all updates maintain consistency with:
- The current game state
- The chosen action and its outcome
- The tribe's development type and diplomatic stance
- Existing relationships and ongoing storylines
"""
        }
    ]

    new_state = make_api_call(messages, GameStateBase, max_tokens=3000)
    print(textwrap.fill(f"{new_state}", width=80))
    return new_state


def determine_outcome(probability: float) -> Tuple[OutcomeType, float]:
    roll = random.random()
    # roll = 0.0
    # roll = 1.0
    if roll < probability:
        return OutcomeType.POSITIVE, roll
    elif roll < (probability + (1 - probability) / 2):
        return OutcomeType.NEUTRAL, roll
    else:
        return OutcomeType.NEGATIVE, roll


def decide_event_type() -> str:
    return random.choices(["single_event", "reaction", "followup"], weights=[0.2, 0.4, 0.4], k=1)[0]
    # return "followup"


def write_story_history(history: List[GameState], filename: str):
    story = []
    for state in history:
        story.append(f"=== Turn {state.turn} ===")

        story.append(f"Situation: {state.previous_situation}")

        story.append(f"Taken action: {state.chosen_action.description}")

        outcome = OutcomeType(state.last_outcome).name
        story.append(f"The result was: {outcome} and resulted in: {state.event_result}\n")

    with open(filename, 'w') as f:
        f.write("\n".join(story))


def save_all_game_states(filename: str):
    global history
    with open(filename, 'w') as f:
        json.dump(history, f, cls=EnumEncoder)

    # Also write the story version
    write_story_history(history, filename.replace('.json', '_story.txt'))


def load_game_state(filename) -> tuple[List[GameState], GameState]:
    with open(filename, 'r') as f:
        loaded_history = json.load(f)

    # Convert enum dictionaries to actual enum values
    converted_history = convert_enums(loaded_history)

    # Convert each history entry to a GameState object
    history: List[GameState] = [GameState(**state) for state in converted_history]

    # The current game state is the last item in the history
    current_game_state = copy.deepcopy(history[-1])

    return history, current_game_state


def create_gui():
    global current_game_state
    global current_tribe_choices
    global current_action_choices
    global config
    global llm_context

    current_tribe_choices = None
    current_action_choices = None

    def save_settings(provider_str, model, url):
        try:
            # Convert string to enum
            provider = LLMProvider(provider_str.lower())

            # Update config
            config.LLM_PROVIDER = provider

            # Update global llm_context with new strategy
            global llm_context
            llm_context = LLMContext(get_llm_strategy(provider))

            # Store additional settings that might be needed by specific strategies
            if hasattr(config, 'openai_model'):
                config.openai_model = model
            if hasattr(config, 'local_url'):
                config.local_url = url

            return f"Settings saved: Provider: {provider.value}, Model: {model}, URL: {url}"
        except ValueError as e:
            return f"Error saving settings: {str(e)}"

    def save_current_game():
        global history
        save_all_game_states('test.json')
        return "Game saved successfully!"

    def load_saved_game(filename):
        global current_game_state, current_action_choices, history
        try:
            history, current_game_state = load_game_state(filename)
            current_action_choices = current_game_state.current_action_choices

            # Update all necessary GUI elements
            tribe_overview = current_game_state.to_context_string()
            recent_history = current_game_state.get_recent_history()
            current_situation = current_game_state.situation

            action_updates = []
            choices = current_action_choices.choices if current_action_choices else []
            for i in range(3):
                if i < len(choices):
                    choice = choices[i]
                    action_updates.extend([
                        gr.update(visible=True, value=choice.caption),
                        gr.update(visible=True, value=choice.description)
                    ])
                else:
                    action_updates.extend([gr.update(visible=False), gr.update(visible=False)])

            current_game_state.turn += 1
            return (
                "Game loaded successfully!",
                gr.update(visible=False),  # Hide tribe selection group
                gr.update(visible=True, value=tribe_overview),  # Show and update tribe overview
                gr.update(visible=True, value=recent_history),  # Show and update recent history
                gr.update(visible=True, value=current_situation),  # Show and update current situation
                *action_updates
            )
        except FileNotFoundError:
            return (
                "No saved game found.",
                gr.update(visible=True),  # Show tribe selection group
                gr.update(visible=False),  # Keep tribe overview hidden
                gr.update(visible=False),  # Keep recent history hidden
                gr.update(visible=False),  # Keep current situation hidden
                *[gr.update(visible=False)] * 6
            )

    def update_game_display():
        if not current_game_state:
            return (
                gr.update(visible=False, value="No active game"),
                gr.update(visible=False, value="No history"),
                gr.update(visible=False, value="No current situation"),
                *[gr.update(visible=False)] * 6
            )

        tribe_overview = current_game_state.to_context_string()
        recent_history = current_game_state.get_recent_history()
        current_situation = current_game_state.situation

        if current_action_choices is not None:
            choices = current_action_choices.choices
        else:
            choices = []

        updates = []
        for i in range(3):
            if i < len(choices):
                choice = choices[i]
                updates.extend([
                    gr.update(visible=True, value=choice.caption),
                    gr.update(visible=True, value=choice.description)
                ])
            else:
                updates.extend([gr.update(visible=False), gr.update(visible=False)])

        return (
            gr.update(visible=True, value=tribe_overview),
            gr.update(visible=True, value=recent_history),
            gr.update(visible=True, value=current_situation),
            *updates
        )

    def perform_action(action_index):
        global current_game_state, current_action_choices, history

        current_game_state.previous_situation = current_game_state.situation
        chosen_action = current_action_choices.choices[action_index]
        current_game_state.chosen_action = chosen_action

        action_captions = [choice.caption for choice in current_action_choices.choices]
        current_game_state.add_action_set(action_captions)

        outcome, roll = determine_outcome(chosen_action.probability)
        current_game_state.last_outcome = outcome  # Store the last outcome
        print(f"Debug: Roll = {roll:.2f}, Probability = {chosen_action.probability:.2f}")

        new_state = generate_new_game_state(current_game_state)

        current_game_state.update(new_state)

        event_type = decide_event_type()
        print(f"Event type: {event_type}")
        choices = generate_choices(current_game_state, event_type)

        current_game_state.situation = choices.situation
        current_action_choices = NextChoices(choices=choices.choices)

        current_game_state.current_action_choices = current_action_choices
        # Make a deep copy of the current game state before appending
        deep_copy_state = copy.deepcopy(current_game_state)
        # Append the deep copy to the history
        history.append(deep_copy_state)
        print("added game state to history")
        base_filename = "Debug_History"
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Create unique filename
        filename = f"{base_filename}_turn{current_game_state.turn}_{timestamp}.json"
        save_all_game_states(filename)
        print(f"turn{current_game_state.turn}")
        current_game_state.turn += 1
        print(f"turn{current_game_state.turn}")

        return update_game_display()

    def generate_tribe_choices_gui():
        global current_tribe_choices, history
        history = []
        current_tribe_choices = generate_tribe_choices()
        if current_tribe_choices:
            result = ""
            for number, choice in enumerate(current_tribe_choices.choices):
                tribe_type = get_tribe_orientation(choice.development, choice.stance)
                result += f"{number + 1}. {choice.name}, "
                result += f"{tribe_type}\n"
                result += f"{choice.description}\n\n"

            return result, gr.update(visible=True, choices=[1, 2, 3])
        else:
            return "Error generating initial choices. Please try again.", gr.update(visible=False)

    def select_tribe_and_start_game(choice):
        global current_game_state, current_tribe_choices, current_action_choices, history
        history = []
        if current_tribe_choices:
            chosen_tribe = next(
                (tribe for index, tribe in enumerate(current_tribe_choices.choices, 1) if index == choice), None)
            if chosen_tribe:
                leader = get_leader(chosen_tribe)
                current_game_state = GameState.initialize(chosen_tribe, leader)
                current_game_state.previous_situation = "Humble beginnings of the " + chosen_tribe.name

                choices = generate_choices(current_game_state, "single_event")

                current_game_state.situation = choices.situation
                current_action_choices = NextChoices(choices=choices.choices)
                current_game_state.current_action_choices = current_action_choices

                # Unpack the return values from update_game_display
                tribe_overview, recent_history, current_situation, *action_updates = update_game_display()

                return (
                    gr.update(visible=False),  # Hide tribe selection group
                    tribe_overview,  # Now includes visibility in update
                    recent_history,  # Now includes visibility in update
                    current_situation,  # Now includes visibility in update
                    *action_updates
                )
            else:
                return (
                    gr.update(visible=True),
                    gr.update(visible=False, value="Invalid choice. Please select a number between 1 and 3."),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    *[gr.update(visible=False)] * 6
                )
        else:
            return (
                gr.update(visible=True),
                gr.update(visible=False, value="Please generate tribe choices first."),
                gr.update(visible=False),
                gr.update(visible=False),
                *[gr.update(visible=False)] * 6
            )

    with gr.Blocks(title="Fantasy Tribe Game") as app:
        gr.Markdown("# Fantasy Tribe Game")

        # Game Tab
        with gr.Tab("Game"):
            # Create tribe selection group
            with gr.Group() as tribe_selection_group:
                start_button = gr.Button("Generate Tribe Choices")
                tribe_choices = gr.Textbox(label="Available Tribes", lines=10)
                tribe_selection = gr.Radio(choices=[1, 2, 3], label="Select your tribe", visible=False)
                select_tribe_button = gr.Button("Select Tribe")

            with gr.Row():
                with gr.Column(scale=1):
                    # Initialize these elements as hidden
                    tribe_overview = gr.Textbox(label="Tribe Overview", lines=10, visible=False)
                    recent_history = gr.Textbox(label="Recent History", lines=5, visible=False)
                with gr.Column(scale=2):
                    current_situation = gr.Textbox(label="Current Situation", lines=5, visible=False)
                    action_button1 = gr.Button("Action 1", visible=False)
                    action_desc1 = gr.Textbox(label="", lines=3, visible=False)
                    action_button2 = gr.Button("Action 2", visible=False)
                    action_desc2 = gr.Textbox(label="", lines=3, visible=False)
                    action_button3 = gr.Button("Action 3", visible=False)
                    action_desc3 = gr.Textbox(label="", lines=3, visible=False)

            with gr.Row():
                save_button = gr.Button("Save Game")
                load_file = gr.File(label="Select Save File", file_types=[".json"])
                load_button = gr.Button("Load Game")

            message_display = gr.Textbox(label="System Messages", lines=2)

            # Add Settings Tab
        with gr.Tab("Settings"):
            with gr.Group():
                gr.Markdown("## LLM Configuration")
                provider_dropdown = gr.Dropdown(
                    choices=[provider.value for provider in LLMProvider],
                    value=config.LLM_PROVIDER.value,
                    label="LLM Provider"
                )
                model_dropdown = gr.Dropdown(
                    choices=["gpt-4o-mini", "gpt-4o"],
                    value=getattr(config, 'openai_model', "gpt-4o-mini"),
                    label="OpenAI Model",
                )
                local_url_input = gr.Textbox(
                    value=getattr(config, 'local_url', "http://127.0.0.1:1234/v1"),
                    label="Local API URL",
                    placeholder="http://127.0.0.1:1234/v1"
                )
                settings_save_btn = gr.Button("Save Settings")
                settings_message = gr.Textbox(label="Settings Status", interactive=False)

                # Add visibility rules for model and URL inputs
                def update_input_visibility(provider):
                    return {
                        model_dropdown: gr.update(visible=provider.lower() == "openai"),
                        local_url_input: gr.update(visible=provider.lower() == "local")
                    }

                provider_dropdown.change(
                    update_input_visibility,
                    inputs=[provider_dropdown],
                    outputs=[model_dropdown, local_url_input]
                )

        # Connect settings events
        settings_save_btn.click(
            save_settings,
            inputs=[provider_dropdown, model_dropdown, local_url_input],
            outputs=[settings_message]
        )

        # Click event handlers remain the same
        start_button.click(
            generate_tribe_choices_gui,
            outputs=[tribe_choices, tribe_selection]
        )

        select_tribe_button.click(
            select_tribe_and_start_game,
            inputs=[tribe_selection],
            outputs=[
                tribe_selection_group,
                tribe_overview,
                recent_history,
                current_situation,
                action_button1, action_desc1,
                action_button2, action_desc2,
                action_button3, action_desc3
            ]
        )

        action_button1.click(
            lambda: perform_action(0),
            outputs=[
                tribe_overview, recent_history, current_situation,
                action_button1, action_desc1,
                action_button2, action_desc2,
                action_button3, action_desc3
            ]
        )

        action_button2.click(
            lambda: perform_action(1),
            outputs=[
                tribe_overview, recent_history, current_situation,
                action_button1, action_desc1,
                action_button2, action_desc2,
                action_button3, action_desc3
            ]
        )

        action_button3.click(
            lambda: perform_action(2),
            outputs=[
                tribe_overview, recent_history, current_situation,
                action_button1, action_desc1,
                action_button2, action_desc2,
                action_button3, action_desc3
            ]
        )

        save_button.click(
            save_current_game,
            outputs=[message_display]
        )

        load_button.click(
            load_saved_game,
            inputs=[load_file],
            outputs=[
                message_display,
                tribe_selection_group,
                tribe_overview,
                recent_history,
                current_situation,
                action_button1, action_desc1,
                action_button2, action_desc2,
                action_button3, action_desc3
            ]
        )

    return app


if __name__ == "__main__":
    app = create_gui()
    app.launch(share=False,
               debug=True,
               server_name="0.0.0.0")
