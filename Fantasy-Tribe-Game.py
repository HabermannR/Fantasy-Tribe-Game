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
                system=messages[0]['content'] + " do not use the word Nexus",  # Use the content of the first message as the system message
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

class NextReactionChoices(BaseModel):
    choices: List[ActionChoice]
    situation: str


class Character(BaseModel):
    name: str
    title: str
    relationships: Optional[List[Dict[str, Any]]] = Field(default=None)

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
    neutrals: List[str]
    enemies: List[str]
    foreign_characters: List[ForeignCharacter]
    territory_size: int
    power_level: int
    tier: int
    situation: str
    event_result: str


def format_relationship_strength(strength):
    """Format relationship strength with consistent spacing."""
    return f"+{strength:2d}" if strength >= 0 else f"{strength:3d}"


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
        # Helper function to format relationships consistently
        def format_relationships(relationships):
            if not relationships:
                return ""
            formatted_rels = []
            for rel in relationships:
                if len(rel) > 0:
                    strength = format_relationship_strength(rel["strength"])
                    formatted_rels.append(f"    ◦ {rel['target']}: {rel['description']} [{strength}]")
            return "\n".join(formatted_rels)

        # Group characters by tribe and alignment
        tribe_groups = {
            'Allies': {},
            'Neutrals': {},
            'Enemies': {}
        }

        # Sort characters into their respective tribes and alignment groups
        for char in self.foreign_characters:
            if char.tribe in self.allies:
                alignment = 'Allies'
            elif char.tribe in self.neutrals:
                alignment = 'Neutrals'
            elif char.tribe in self.enemies:
                alignment = 'Enemies'
            else:
                continue  # Skip if tribe alignment is unknown

            if char.tribe not in tribe_groups[alignment]:
                tribe_groups[alignment][char.tribe] = []
            tribe_groups[alignment][char.tribe].append(char)

        # Start building the text output
        tribe_type = get_tribe_orientation(self.tribe.development, self.tribe.stance)
        text = f"""{self.tribe.name}
  {tribe_type}

{self.tribe.description}

Leaders:"""

        # Format all leaders with consistent formatting
        for leader in self.leaders:
            text += f"\n  • {leader.title}: {leader.name}"  # Changed hyphen to colon for consistency
            if leader.relationships:
                text += "\n  Relations:"
                text += "\n" + format_relationships(leader.relationships)

        # Format each alignment section with tribes and their characters
        for alignment in ['Allies', 'Neutrals', 'Enemies']:
            text += f"\n\n{alignment}:"
            # First list all tribes in this alignment category
            tribes_in_category = sorted(set(self.allies if alignment == 'Allies'
                                            else self.neutrals if alignment == 'Neutrals'
            else self.enemies))
            #text += f" {', '.join(tribes_in_category)}"

            # Then list characters grouped by their tribes
            for tribe in sorted(tribe_groups[alignment].keys()):
                text += f"\n  {tribe}:"
                for char in sorted(tribe_groups[alignment][tribe], key=lambda x: x.name):
                    text += f"\n  • {char.title}: {char.name}"
                    if char.relationships:
                        text += "\n  Relations:"
                        text += "\n" + format_relationships(char.relationships)

        # Add basic tribe information
        text += f"""

Gold: {self.gold}
Territory Size: {self.territory_size}
Power Level: {self.power_level}
Tribe Tier: {self.tier}"""

        return text

    @staticmethod
    def get_recent_history(num_events: int = 5, full_output: bool = True,
                           start_event: int = 1) -> str:
        global history
        # Calculate the starting index from the end
        start_idx = start_event + num_events - 1

        # Get the slice of events we want, in reverse order
        recent_events = history[-start_idx:-start_event + 1 if start_event > 1 else None][-num_events:]
        if full_output:
            return "\n\n".join([
                                   f"- {event.event_result} (Outcome: {OutcomeType(event.last_outcome).name if event.last_outcome else 'N/A'})"
                                   for event in recent_events])
        else:
            # Reverse the events and include turn numbers
            recent_events.reverse()
            formatted_events = []
            for event in recent_events:
                turn_header = f"=== Turn {event.turn} ==="
                formatted_events.extend([turn_header, event.event_result])
            return "\n\n".join(formatted_events)

    @classmethod
    def initialize(cls, tribe: TribeChoice,  leader: Character) -> 'GameState':
        return cls(
            tribe=tribe,
            leaders=[leader, ],
            gold=100,
            allies=[],
            neutrals=[],
            enemies=[],
            foreign_characters=[],
            territory_size=10,
            power_level=5,
            tier=1,
            situation="",
            event_result=""
        )

    def update(self, new_state: GameStateBase):
        for field in GameStateBase.model_fields:
            setattr(self, field, getattr(new_state, field))

def load_game_state(filename) -> tuple[List[GameState], GameState]:
    with open(filename, 'r') as f:
        loaded_history = json.load(f)

    # Convert enum dictionaries to actual enum values
    converted_history = convert_enums(loaded_history)

    # Convert each history entry to a GameState object
    history: List[GameState] = [GameState(**state) for state in converted_history]

    # The current game state is the last item in the history
    current_game_state = history[-1]

    return history, current_game_state

current_game_state: Optional[GameState] = None
global history
history: List[GameState] = []


def generate_tribe_choices() -> InitialChoices:
    # Generate 3 unique combinations of development type and diplomatic stance
    #combinations: List[Tuple[DevelopmentType, DiplomaticStance]] = []
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

def get_leader(chosen_tribe: TribeChoice) -> Character:
    messages = [
        {
            "role": "system",
            "content": "You are the game master for a text based strategic game, where the player rules over a tribe in a fantasy based setting. Output your answers as JSON"
        },
        {
            "role": "user",
            "content": f"""Give me the name and title for the leader of {chosen_tribe.name}. Their description is:
{chosen_tribe.description}
You can leave the field relationships empty."""
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


def get_tier_specific_prompt(tier: int, development: DevelopmentType, stance: DiplomaticStance)-> str:
    """
    Generate a prompt based on tier level and tribe orientation.
    """
    orientation = get_tribe_orientation(development, stance)
    specific = ""
    base_prompt = ""
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


def generate_next_choices(game_state: GameState) ->NextChoices:
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

Based on this information, create a new event. Add the even description to:
{game_state.situation} 
while slightly summarizing it and save it in "situation".

Then present three possible actions the player can take next, possibly utilizing one or more of the tribes characters. {tier_specific_prompt} 
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

def generate_choices(game_state: GameState, choice_type: str) -> NextReactionChoices:
    recent_history = game_state.get_recent_history(num_events=5, start_event=2)
    probability_adjustment = get_probability_adjustment(game_state.last_outcome)
    tier_specific_prompt = ""

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
        tier_specific_prompt = get_tier_specific_prompt(game_state.tier, game_state.tribe.development,
                                                        game_state.tribe.stance)
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

Then present three possible actions the player can take next, possibly utilizing one or more of the tribes characters. {tier_specific_prompt}
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
    recent_history = game_state.get_recent_history(num_events=5)
    tribes_prompt = ""
    if game_state.turn > 5 and len(game_state.enemies) < 1:
        tribes_prompt += "Add one or two enemies"
    if game_state.turn > 3 and len(game_state.enemies) < 2:
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

The old situation was: 
{game_state.previous_situation}
The player chose the following action as response: {game_state.chosen_action.caption}
{game_state.chosen_action.description}

The outcome of this action was {game_state.last_outcome.name}.

Please provide an updated game state given the chosen action and its outcome.
But keep in mind that the event does not need to be completely finished with the given outcome.
In fact, most events take multiple turns to complete.
Include the following in the updated game state:
1. Any changes to the tribe's name, description, and its leaders. Alliances, Royal marriages, won or lost wars and so on do change the tribe's name, the leader title, or even the leaders name!
But do not change the tribe's name every turn, only when important event have happened.
You can also add new leaders, for example heroes, generals, mages, shamans, and so on. But keep the number limited to a max of 5. You can also drop leaders not necessary anymore.
2. Updated gold values. Tier 1 should be between 0 and 200 gold, Tier 2 between 100 and 1000, Tier 3 between 1000 and 10000, and in Tier 4 gold should be over 10000.
3. Any new allies, neutrals or enemies, and their leaders and heroes. Keep these foreign characters also at max of 5.
{tribes_prompt}
It takes many turns to move neutral factions to become allies, and even more to form formal alliances! 
4. Changes to territory size, power level and the current tier. 
    Tier 1 is local issues and basic survival, power level 1-15, power level may rise 1 or 2 points
    Tier 2 is regional influence and early empire-building, power level 16-40, power level may rise 2-5 points
    Tier 3 is nation-building and complex diplomacy, power level 41-75, power level may rise 2-10 points
    Tier 4 is world-shaping decisions and legendary quests, power level > 75, power level may rise 10+ points
5. Update the current situation in situation
6. Add a description of what happened and its outcome in your role as game master in event_result.
Do not talk about Power level and Tier in event_result, instead focus on telling the story of the event.
Keep your answer focused for event_result, 2 or 3 paragraphs. Remember that the event does not need to be finished with this action. 
7. Please update the relationships between the leaders based on these events. You can:
- Modify existing relationship descriptions and strengths
- Add new relationships
- Remove relationships that are no longer relevant

Remember:
- Strength ranges from -100 (bitter enemies) to 100 (strongest allies)
- Each relationship should have a "target", "description", and "strength"
- Consider how relationships might be reciprocal

Be creative but consistent with the current game state, and the chosen action.
"""
        }
    ]

    new_state = make_api_call(messages, GameStateBase, max_tokens=2000)
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
    #return "followup"



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

    def load_saved_game():
        global current_game_state, current_action_choices, history
        try:
            history, current_game_state = load_game_state("History.json")
            current_action_choices = current_game_state.current_action_choices

            # Update all necessary GUI elements
            tribe_overview = current_game_state.to_context_string()
            recent_history = current_game_state.get_recent_history(num_events=1000, full_output=False)
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
        recent_history = current_game_state.get_recent_history(num_events=1000, full_output=False)
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

        # Check if new tier
        if new_state.tier > current_game_state.tier:
            print(f"Tier up! From {current_game_state.tier} to {new_state.tier}")

        current_game_state.update(new_state)
        # Make a deep copy of the current game state before appending
        deep_copy_state = copy.deepcopy(current_game_state)
        # Append the deep copy to the history
        history.append(deep_copy_state)
        save_all_game_states("History.json")
        current_game_state.turn += 1

        event_type = decide_event_type()
        print(f"Event type: {event_type}")
        choices = generate_choices(current_game_state, event_type)

        current_game_state.situation = choices.situation
        current_action_choices = NextChoices(choices=choices.choices)


        current_game_state.current_action_choices = current_action_choices


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