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
from enum import Enum
from typing import Dict, Any, TypeVar, Type, List, Tuple, Optional, Literal

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
    LLM_PROVIDER: LLMProvider = LLMProvider.ANTHROPIC
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
                system=messages[0]['content'],  # Use the content of the first message as the system message
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
            print(f"\nError occurred: {str(e)}")
            return None


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
            print(f"\nError occurred: {str(e)}")
            return None


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
            print(f"\nError occurred: {str(e)}")
            return None


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


class Quest(BaseModel):
    name: str
    description: str
    total_parts: int
    current_part: int = 1
    state: str = "active"

    def advance_part(self):
        if self.current_part < self.total_parts:
            self.current_part += 1
        else:
            self.state = "completed"


class QuestInfo(BaseModel):
    name: str
    description: str
    total_parts: int
    part_1_choices: List[ActionChoice]


class Character(BaseModel):
    Name: str
    Title: str

class DiplomaticStance(Enum):
    PEACEFUL = "peaceful"
    NEUTRAL = "neutral"
    AGGRESSIVE = "aggressive"

class DevelopmentType(Enum):
    MAGICAL = "magical"
    HYBRID = "hybrid"
    PRACTICAL = "practical"

class TribeChoice(BaseModel):
    name: str
    description: str
    development: DevelopmentType
    stance: DiplomaticStance

class InitialChoices(BaseModel):
    choices: List[TribeChoice]


class GameStateBase(BaseModel):
    tribe: TribeChoice
    leaders: List[Character]
    gold: int
    allies: List[str]
    enemies: List[str]
    territory_size: int
    power_level: int
    tier: int
    quest_description: str


class GameStateUpdate(GameStateBase):
    event_result: str


class GameState(GameStateBase):
    history: List[Dict[str, Any]] = Field(default_factory=list)
    current_action_choices: Optional[NextChoices] = None
    last_action_sets: List[List[str]] = Field(default_factory=list)
    last_outcome: Optional[OutcomeType] = None
    current_quest: Optional[Quest] = None
    turn: int = 1

    class Config:
        arbitrary_types_allowed = True

    def add_quest(self, quest: Quest):
        self.current_quest = quest

    def add_event(self, event_type: str, description: str, outcome: OutcomeType):
        self.history.append({
            "event_type": event_type,
            "description": description,
            "outcome": outcome.value,
            "turn": self.turn
        })

    def add_action_set(self, actions: List[str]):
        self.last_action_sets.append(actions)
        if len(self.last_action_sets) > 3:  # Keep only the last 3 sets
            self.last_action_sets.pop(0)

    def to_context_string(self) -> str:
        tribe_type = get_tribe_orientation(self.tribe.development, self.tribe.stance)
        text = f"""{self.tribe.name}
{tribe_type}

{self.tribe.description}

Leader:"""
        for leader in self.leaders:
            text += f"""\n    {leader.Title}: {leader.Name}"""
        text += f"""\n
Gold: {self.gold}
Allies: {', '.join(self.allies)}
Enemies: {', '.join(self.enemies)}
Territory Size: {self.territory_size}
Power Level: {self.power_level}
Tribe Tier: {self.tier}"""
        return text

    def get_recent_history(self, num_events: int = 5, full_output: bool = True) -> str:
        recent_events = self.history[-num_events:]
        if full_output:
            return "\n\n".join([f"- {event['description']} (Outcome: {OutcomeType(event['outcome']).name})"
                                for event in recent_events])
        else:
            # Reverse the events and include turn numbers
            recent_events.reverse()
            formatted_events = []
            for event in recent_events:
                turn_header = f"=== Turn {event['turn']} ==="
                formatted_events.extend([turn_header, event['description']])
            return "\n\n".join(formatted_events)

    @classmethod
    def initialize(cls, Tribe: TribeChoice,  leader: Character) -> 'GameState':
        return cls(
            tribe=Tribe,
            leaders=[leader, ],
            gold=100,
            allies=[],
            enemies=[],
            territory_size=10,
            power_level=5,
            tier=1,
            quest_description=""
        )

    def update(self, new_state: GameStateUpdate, outcome: OutcomeType):
        for field in GameStateBase.model_fields:
            setattr(self, field, getattr(new_state, field))
        self.add_event("Action", new_state.event_result, outcome)
        self.turn += 1
        # Update quest progress if there's a current quest
        if self.current_quest:
            if outcome == OutcomeType.POSITIVE:
                self.current_quest.current_part += 1
                if self.current_quest.current_part > self.current_quest.total_parts:
                    self.current_quest.state = "completed"
                    print(f"Quest '{self.current_quest.name}' completed!")
                    self.current_quest = None


current_game_state: Optional[GameState] = None
history: List[dict] = []


def generate_tribe_choices():
    race_array = ["Lizardmen", "Giants", "Vampires", "Humans", "Orcs", "Dragons", "Necromancers", "Wizards",
                  "Halflings"]
    attribute_array = ["magic fire capabilities", "water magic", "strong fighting capabilities",
                       "health regeneration", "immortality", "raise the undead capabilities", "telepathic abilities"]

    race = [f"{random.choice(race_array)} with {random.choice(attribute_array)}" for _ in range(3)]

    messages = [
        {
            "role": "system",
            "content": "You are the game master for a text based strategic game, where I rule over a tribe in a fantasy based setting. Output your answers as JSON"
        },
        {
            "role": "user",
            "content": f"""Present me three choices for my tribe. 
Each choice should include:
1. A unique tribe name (e.g., {race[0]}, {race[1]} or {race[2]}, but create new ones)
2. A description of the tribe (without mentioning the tribe name)
3. The tribe's development type (must be one of: "magical", "hybrid", or "practical")
4. The tribe's diplomatic stance, "peaceful", "neutral" or "aggressive"
"""
        }
    ]

    choices = make_api_call(messages, InitialChoices)

    if choices:
        print("\n=== Available Tribe Choices ===")
        for number, choice in enumerate(choices.choices):
            tribe_type = get_tribe_orientation(choice.development, choice.stance)
            print(f"\n{number}. {choice.name}, {tribe_type}")
            print(textwrap.fill(f"Description: {choice.description}", width=80, initial_indent="   ",
                                subsequent_indent="   "))

    return choices


def get_leader(chosen_tribe):
    messages = [
        {
            "role": "system",
            "content": "You are the game master for a text based strategic game, where I rule over a tribe in a fantasy based setting. Output your answers as JSON"
        },
        {
            "role": "user",
            "content": f"""Give me the name and title for the leader of {chosen_tribe}."""
        }
    ]

    leader = make_api_call(messages, Character)

    if leader:
        print(textwrap.fill(f"\nLeader: {leader}", width=80, initial_indent="   ",
                            subsequent_indent="   "))

    return leader


def get_tribe_orientation(development: DevelopmentType, stance: DiplomaticStance):
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


def get_tier_specific_prompt(tier: int, development: DevelopmentType, stance: DiplomaticStance):
    """
    Generate a prompt based on tier level and tribe orientation.
    """
    orientation = get_tribe_orientation(development, stance)

    if tier == 1:
        base_prompt = """Focus on local establishment and survival. Actions should involve:
- Gathering essential resources, establishing basic infrastructure, and dealing with immediate threats."""

        if development == DevelopmentType.MAGICAL:
            if stance == DiplomaticStance.PEACEFUL:
                specific = """- Discovering latent magical abilities
- Communing with local spirits
- Deciphering ancient runes
- Establishing harmony with natural forces"""
            elif stance == DiplomaticStance.NEUTRAL:
                specific = """- Learning both protective and offensive magic
- Establishing magical research centers
- Balancing spiritual harmony with power
- Creating versatile magical defenses"""
            else:  # AGGRESSIVE
                specific = """- Discovering combat magic fundamentals
- Training battle-mage initiates
- Establishing magical defensive perimeters
- Identifying sources of destructive power"""
        elif development == DevelopmentType.HYBRID:
            if stance == DiplomaticStance.PEACEFUL:
                specific = """- Combining simple machines with minor enchantments
- Studying both natural laws and magical principles
- Creating enhanced farming tools
- Developing basic magitech infrastructure"""
            elif stance == DiplomaticStance.NEUTRAL:
                specific = """- Developing enchanted tools and weapons
- Creating hybrid defense systems
- Establishing basic arcane workshops
- Merging magical and mechanical power"""
            else:  # AGGRESSIVE
                specific = """- Crafting magically enhanced weapons
- Training techno-warrior initiates
- Building hybrid fortifications
- Developing magitech combat systems"""
        else:  # PRACTICAL
            if stance == DiplomaticStance.PEACEFUL:
                specific = """- Developing primitive tools
- Understanding local flora and fauna
- Establishing basic crafting techniques
- Building sustainable settlements"""
            elif stance == DiplomaticStance.NEUTRAL:
                specific = """- Creating multipurpose tools and weapons
- Developing defensive structures
- Establishing trade and security
- Training militia for protection"""
            else:  # AGGRESSIVE
                specific = """- Crafting basic weapons and armor
- Training warrior bands
- Building defensive structures
- Developing combat tactics"""

    elif tier == 2:
        base_prompt = """Focus on expansion and power cultivation. Actions should involve:"""

        if development == DevelopmentType.MAGICAL:
            if stance == DiplomaticStance.PEACEFUL:
                specific = """- Harnessing ancient magics
- Taming mythical beasts
- Attuning to elemental forces
- Creating magical sanctuaries"""
            elif stance == DiplomaticStance.NEUTRAL:
                specific = """- Developing versatile magical applications
- Creating magical barriers and wards
- Establishing diplomatic magical channels
- Training battle-ready peacekeepers"""
            else:  # AGGRESSIVE
                specific = """- Mastering combat spells and magical warfare
- Binding war spirits to weapons
- Creating magical siege equipment
- Establishing magical military academies"""
        elif development == DevelopmentType.HYBRID:
            if stance == DiplomaticStance.PEACEFUL:
                specific = """- Creating self-maintaining enchanted machines
- Developing magical power generators
- Establishing magitech research centers
- Harmonizing technology with natural magic"""
            elif stance == DiplomaticStance.NEUTRAL:
                specific = """- Engineering spell-powered devices
- Creating hybrid defense networks
- Establishing technomantic guilds
- Developing versatile magitech systems"""
            else:  # AGGRESSIVE
                specific = """- Developing magitech weaponry
- Creating enchanted war machines
- Establishing combat technomancy schools
- Engineering magical artillery systems"""
        else:  # PRACTICAL
            if stance == DiplomaticStance.PEACEFUL:
                specific = """- Developing advanced technologies
- Establishing trade networks
- Mastering the land
- Building centers of learning"""
            elif stance == DiplomaticStance.NEUTRAL:
                specific = """- Creating defensive technologies
- Establishing military deterrents
- Building diplomatic infrastructure
- Developing dual-use innovations"""
            else:  # AGGRESSIVE
                specific = """- Developing advanced weapons
- Establishing military alliances
- Creating siege engines
- Training elite military units"""

    elif tier == 3:
        base_prompt = """Focus on realm-shaping endeavors and far-reaching influence. Actions should involve:"""

        if development == DevelopmentType.MAGICAL:
            if stance == DiplomaticStance.PEACEFUL:
                specific = """- Forging pacts with benevolent entities
- Creating magical wonderlands
- Establishing magical healing centers
- Developing reality-harmonizing magic"""
            elif stance == DiplomaticStance.NEUTRAL:
                specific = """- Balancing magical forces
- Creating magical deterrence systems
- Establishing planar diplomatic channels
- Developing reality-stabilizing magic"""
            else:  # AGGRESSIVE
                specific = """- Creating magical weapons of mass destruction
- Summoning armies of magical constructs
- Establishing magical dominion
- Developing reality-warping battle magic"""
        elif development == DevelopmentType.HYBRID:
            if stance == DiplomaticStance.PEACEFUL:
                specific = """- Creating self-evolving magitech ecosystems
- Developing reality-engineering devices
- Establishing technomantic sanctuaries
- Merging consciousness with magitech"""
            elif stance == DiplomaticStance.NEUTRAL:
                specific = """- Engineering planar manipulation devices
- Creating hybrid reality stabilizers
- Establishing multidimensional networks
- Developing synthetic magic systems"""
            else:  # AGGRESSIVE
                specific = """- Creating magitech superweapons
- Engineering spell-powered war titans
- Establishing technomantic armies
- Developing reality-breaking siege engines"""
        else:  # PRACTICAL
            if stance == DiplomaticStance.PEACEFUL:
                specific = """- Creating grand wonders
- Establishing intricate political systems
- Pioneering revolutionary innovations
- Building technological marvels"""
            elif stance == DiplomaticStance.NEUTRAL:
                specific = """- Developing autonomous defense systems
- Creating technological deterrents
- Establishing global influence networks
- Engineering advanced civilization systems"""
            else:  # AGGRESSIVE
                specific = """- Developing advanced military technology
- Creating vast armies and navies
- Establishing military industrial complexes
- Engineering superweapons"""

    elif tier >= 4:
        base_prompt = """Focus on world-altering achievements and cosmic influence. Actions should have far-reaching consequences that reshape reality itself."""

        if development == DevelopmentType.MAGICAL:
            if stance == DiplomaticStance.PEACEFUL:
                specific = """- Ascending to benevolent godhood
- Creating planes of harmony
- Establishing cosmic balance
- Achieving magical transcendence"""
            elif stance == DiplomaticStance.NEUTRAL:
                specific = """- Becoming custodians of reality
- Creating planes of balance
- Establishing cosmic order
- Achieving magical equilibrium"""
            else:  # AGGRESSIVE
                specific = """- Becoming gods of war
- Creating planes of eternal conflict
- Wielding reality-destroying magic
- Commanding armies of cosmic beings"""
        elif development == DevelopmentType.HYBRID:
            if stance == DiplomaticStance.PEACEFUL:
                specific = """- Creating techno-organic realities
- Merging consciousness with the cosmos
- Engineering synthetic divine beings
- Achieving technomantic singularity"""
            elif stance == DiplomaticStance.NEUTRAL:
                specific = """- Becoming synthetic demigods
- Creating hybrid dimensional matrices
- Establishing techno-magical order
- Achieving perfect synthesis"""
            else:  # AGGRESSIVE
                specific = """- Creating magitech death stars
- Engineering spell-powered cosmic weapons
- Establishing technomantic dominion
- Achieving destructive synthesis"""
        else:  # PRACTICAL
            if stance == DiplomaticStance.PEACEFUL:
                specific = """- Achieving technological singularity
- Terraforming entire worlds
- Unlocking the secrets of creation
- Building utopian civilizations"""
            elif stance == DiplomaticStance.NEUTRAL:
                specific = """- Creating self-sustaining systems
- Establishing cosmic balance through technology
- Engineering perfect equilibrium
- Achieving technological transcendence"""
            else:  # AGGRESSIVE
                specific = """- Creating planet-destroying weapons
- Establishing interplanar military dominion
- Engineering perfect warrior races
- Achieving technological supremacy"""

    consequence_prompt = """Consider the implications of your path as {}. Your decisions will shape not only your people's destiny but the very nature of power in this world.""".format(
        orientation)

    return f"{base_prompt}\n{specific}\n\n{consequence_prompt}"


def generate_next_choices(game_state: GameState):
    tier_specific_prompt = get_tier_specific_prompt(game_state.tier, game_state.tribe.development, game_state.tribe.stance)
    last_actions_prompt = ""
    if game_state.last_action_sets:
        last_actions_str = ", ".join([action for action_set in game_state.last_action_sets for action in action_set])
        last_actions_prompt = f"The last choices were:\n{last_actions_str}\nPlease avoid repeating these actions and generate entirely new options."

    recent_history = game_state.get_recent_history(num_events=5)

    if game_state.last_outcome in [OutcomeType.NEUTRAL, OutcomeType.NEGATIVE]:
        probability_adjustment = "Due to the recent neutral or negative outcome, provide higher probabilities of success for the next choices (between 0.6 and 0.9)."
    else:
        probability_adjustment = "Provide balanced probabilities of success for the next choices (between 0.5 and 0.8)."

    messages = [
        {
            "role": "system",
            "content": "You are the game master for a text based strategic game, where I rule over a tribe in a fantasy based setting. Output your answers as JSON"
        },
        {
            "role": "user",
            "content": f"""Here's the current game state:

{game_state.to_context_string()}

Recent History:
{recent_history}

Based on this information, present three possible actions I can take next, possibly utilizing one or more of the tribes characters. {tier_specific_prompt} 
Include potential consequences and strategic considerations for each action.
{probability_adjustment}
Give one of the choices a quite high probability, and make at least one choice quite aggressive, 
but do not automatically give the aggressive options a low probability. Instead, base the probabilities on the given information about the tribe. 
Be creative, do not repeat these old choices:
{last_actions_prompt}"""
        }
    ]

    next_choices = make_api_call(messages, NextChoices, max_tokens=2000)
    if next_choices:
        print("\n=== Available Actions ===")
        for i, choice in enumerate(next_choices.choices, 1):
            print(f"\n{i}. {choice.caption}")
            print(textwrap.fill(f"Description: {choice.description}", width=80, initial_indent="   ",
                                subsequent_indent="   "))
            print(f"   Probability of Success: {choice.probability:.2f}")

    return next_choices


def generate_new_quest(game_state: GameState) -> NextChoices:
    recent_history = game_state.get_recent_history(num_events=5)
    tier_specific_prompt = get_tier_specific_prompt(game_state.tier, game_state.tribe.development, game_state.tribe.stance)

    messages = [
        {
            "role": "system",
            "content": "You are the game master for a text based strategic game. Generate a new multi-part quest for the tribe."
        },
        {
            "role": "user",
            "content": f"""
Current game state:
{game_state.to_context_string()}

Recent History:
{recent_history}

Generate a new multi-part quest for the tribe. The quest should be interesting and relevant to the tribe's current situation. {tier_specific_prompt}

Provide the following information:
1. A quest name
2. A brief overall description of the quest
3. The total number of parts for this quest (between 2 and 5)
4. Three possible approaches to start the quest (part 1), each with:
   a. A brief caption
   b. A detailed description
   c. A probability of success (between 0.5 and 0.8)

Be creative and consider the current game state, recent history, and resources available to the player.

Output your response as a JSON object matching the QuestInfo schema.
"""
        }
    ]

    quest_info = make_api_call(messages, QuestInfo, max_tokens=2000)
    print(textwrap.fill(f"{quest_info}", width=80))
    new_quest = Quest(
        name=quest_info.name,
        description=quest_info.description,
        total_parts=quest_info.total_parts,
        current_part=1
    )
    game_state.add_quest(new_quest)
    game_state.quest_description = quest_info.description
    next_choices = NextChoices(choices=quest_info.part_1_choices)
    if next_choices:
        print("\n=== Available Actions ===")
        for i, choice in enumerate(next_choices.choices, 1):
            print(f"\n{i}. {choice.caption}")
            print(textwrap.fill(f"Description: {choice.description}", width=80, initial_indent="   ",
                                subsequent_indent="   "))
            print(f"   Probability of Success: {choice.probability:.2f}")

    return next_choices


def generate_next_quest_part(game_state: GameState) -> NextChoices:
    current_quest = game_state.current_quest
    recent_history = game_state.get_recent_history(num_events=5)
    tier_specific_prompt = get_tier_specific_prompt(game_state.tier, game_state.tribe.development, game_state.tribe.stance)
    if current_quest.current_part == current_quest.total_parts:
        final_quest = "This is the final part, make the choices epic and exciting"
    else:
        final_quest = ""
    messages = [
        {
            "role": "system",
            "content": "You are the game master for a text based strategic game. Generate the next part of an ongoing quest for the tribe."
        },
        {
            "role": "user",
            "content": f"""
Current game state:
{game_state.to_context_string()}

Recent History:
{recent_history}

Current Quest: {current_quest.name}
Quest Description: {current_quest.description}

Generate the next part of this quest with three possible approaches to complete this part. This should be part {current_quest.current_part} of {current_quest.total_parts} of the ongoing quest. The choices should be consistent with the quest's overall narrative and the tribe's current situation.
{tier_specific_prompt}
For each approach, provide:
1. A brief caption
2. A detailed description
3. A probability of success (between 0.5 and 0.8)

Be creative and consider the current game state, the quest's progress so far, recent history, and resources available to the player. Utilize the tribes leaders.
{final_quest}
"""
        }
    ]

    choices = make_api_call(messages, NextChoices, max_tokens=2000)
    if choices:
        print("\n=== Available Actions ===")
        for i, choice in enumerate(choices.choices, 1):
            print(f"\n{i}. {choice.caption}")
            print(textwrap.fill(f"Description: {choice.description}", width=80, initial_indent="   ",
                                subsequent_indent="   "))
            print(f"   Probability of Success: {choice.probability:.2f}")
    return choices


def generate_new_game_state(game_state: GameState, chosen_action: ActionChoice,
                            outcome: OutcomeType) -> GameStateUpdate:
    llm_specific = ""
    if config.LLM_PROVIDER == LLMProvider.ANTHROPIC:
        llm_specific ="keep your answer short for event_result"
    recent_history = game_state.get_recent_history(num_events=5)
    current_quest = game_state.current_quest
    if current_quest:
        quest_update = "5. Update the quest description in quest_description"
    else:
        quest_update = "5. Update the current situation in quest_description"
    messages = [
        {
            "role": "system",
            "content": "You are the game master for a text-based strategic game, where the player rules over a tribe in a fantasy setting. Your task is to update the game state based on the player's actions and their outcomes."
        },
        {
            "role": "user",
            "content": f"""
Current game state:
{game_state.to_context_string()}
The tribes development is {game_state.tribe.development}, its diplomatic stance is {game_state.tribe.stance}.
Change this only for major upheavals!

Recent History:
{recent_history}

The player chose the following action:
{chosen_action.caption}
{chosen_action.description}

The outcome of this action was {outcome.name}.

Please provide an updated game state, including:
1. Any changes to the tribe's name, description, and its leaders. Alliances, Royal marriages, won or lost wars and so on do change the tribe's name, the leader title, or even the leaders name!
But do not change the tribe's name every turn, only when important event have happened.
You can also add new leaders, for example heroes, generals, mages, shamans, and so on. But keep the number limited to a max of 5. You can also drop leaders not necessary anymore.
2. Updated gold values. Tier 1 should be between 0 and 200 gold, Tier 2 between 100 and 1000, Tier 3 between 1000 and 10000, and in Tier 4 gold should be over 10000.
3. Any new allies or enemies
4. Changes to territory size, power level and the current tier. 
    Tier 1 is local issues and basic survival, power level 1-10, power level may rise 1 or 2 points
    Tier 2 is regional influence and early empire-building, power level 11-30, power level may rise 2-5 points
    Tier 3 is nation-building and complex diplomacy, power level 31-75, power level may rise 2-10 points
    Tier 4 is world-shaping decisions and legendary quests, power level > 75, power level may rise 10+ points
{quest_update}
At last, add a description of what happened and its outcome in your role as game master in event_result.
Do not talk about Power level and Tier in event_result, instead focus on telling the story of the event.
{llm_specific}

Be creative but consistent with the current game state, active quests, and the chosen action.
"""
        }
    ]

    new_state = make_api_call(messages, GameStateUpdate, max_tokens=2000)
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
    return random.choices(["single_event", "new_quest"], weights=[0.7, 0.3], k=1)[0]


def save_game(game_state: GameState, filename: str):
    with open(filename, 'w') as f:
        f.write(json.dumps(game_state.model_dump(), cls=EnumEncoder))


def save_all_game_states(all_game_states: list, filename: str):
    with open(filename, 'w') as f:
        f.write(json.dumps(all_game_states, cls=EnumEncoder))


def load_game(filename: str) -> GameState:
    with open(filename, 'r') as f:
        data = json.loads(f.read(), object_hook=as_enum)
    return GameState.model_validate(data)


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
        if current_game_state:
            save_game(current_game_state, "saved_game.json")
            return "Game saved successfully!"
        return "No game in progress to save."

    def load_saved_game():
        global current_game_state, current_action_choices
        try:
            current_game_state = load_game("saved_game.json")
            current_action_choices = current_game_state.current_action_choices

            # Update all necessary GUI elements
            tribe_overview = current_game_state.to_context_string()
            recent_history = current_game_state.get_recent_history(num_events=1000, full_output=False)
            current_situation = current_game_state.quest_description

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
        recent_history = current_game_state.get_recent_history(num_events=1000, full_output=False)
        current_situation = current_game_state.quest_description

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
        global current_game_state, current_action_choices

        chosen_action = current_action_choices.choices[action_index]
        action_captions = [choice.caption for choice in current_action_choices.choices]
        current_game_state.add_action_set(action_captions)

        outcome, roll = determine_outcome(chosen_action.probability)
        print(f"Debug: Roll = {roll:.2f}, Probability = {chosen_action.probability:.2f}")

        new_state = generate_new_game_state(current_game_state, chosen_action, outcome)

        # Check if new tier
        if new_state.tier > current_game_state.tier:
            print(f"Tier up! From {current_game_state.tier} to {new_state.tier}")

        current_game_state.update(new_state, outcome)
        current_game_state.last_outcome = outcome  # Store the last outcome

        if current_game_state.current_quest and current_game_state.current_quest.current_part <= current_game_state.current_quest.total_parts:
            # Continue the current quest
            current_action_choices = generate_next_quest_part(current_game_state)
        else:
            event_type = decide_event_type()

            if event_type == "new_quest":
                current_action_choices = generate_new_quest(current_game_state)
                print(f"New Quest: {current_game_state.current_quest.name}")
            else:  # single_event
                current_action_choices = generate_next_choices(current_game_state)

        current_game_state.current_action_choices = current_action_choices
        history.append(current_game_state.model_dump())
        save_all_game_states(history, "History.json")

        return update_game_display()

    def generate_tribe_choices_gui():
        global current_tribe_choices
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
        global current_game_state, current_tribe_choices, current_action_choices
        if current_tribe_choices:
            chosen_tribe = next(
                (tribe for index, tribe in enumerate(current_tribe_choices.choices, 1) if index == choice), None)
            if chosen_tribe:
                leader = get_leader(chosen_tribe)
                current_game_state = GameState.initialize(chosen_tribe, leader)
                current_action_choices = generate_next_choices(current_game_state)
                current_game_state.current_action_choices = current_action_choices

                history.append(current_game_state.model_dump())
                save_all_game_states(history, "History.json")

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
