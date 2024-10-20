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

import random
import textwrap
import copy
import gradio as gr

from datetime import datetime
from enum import Enum
from typing import List, Tuple, Optional

from pydantic import BaseModel, Field
from LLM_manager import LLMContext, Config, ModelConfig, SummaryModelConfig, LLMProvider, SummaryMode

random.seed(datetime.now().timestamp())

class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return {"__enum__": str(obj)}
        if isinstance(obj, BaseModel):
            return obj.model_dump(exclude_unset=True)
        return json.JSONEncoder.default(self, obj)


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


class OutcomeType(Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    CATASTROPHE = "catastrophe"


class EventType(Enum):
    SINGLE_EVENT = "single_event"
    REACTION = "reaction"
    FOLLOWUP = "followup"


class ActionChoice(BaseModel):
    caption: str
    description: str
    probability: float


class NextChoices(BaseModel):
    choices: List[ActionChoice]


class NextReactionChoices(BaseModel):
    choices: List[ActionChoice]
    situation: str

class SummaryModel(BaseModel):
    summary: str


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


class TextFormatter:
    @staticmethod
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

    @staticmethod
    def format_tribe_description(tribe: TribeType, leaders: List[Character], relation: bool) -> str:
        tribe_type = get_tribe_orientation(tribe.development, tribe.stance)
        text = f"""{tribe.name}
{tribe_type}

{tribe.description}

Leaders:"""

        for leader in leaders:
            text += TextFormatter._format_leader(leader, relation)
        return text

    @staticmethod
    def _format_leader(leader: Character, relation: bool) -> str:
        text = f"\n  • {leader.title}: {leader.name}"
        if leader.relationships and relation:
            text += "\n    Relationships:"
            for rel in leader.relationships:
                sentiment = TextFormatter.get_sentiment(rel.opinion)
                text += f"\n      - {rel.type} relationship with {rel.target} ({sentiment}, {rel.opinion})"
        return text

    @staticmethod
    def format_foreign_tribes(foreign_tribes: List[ForeignTribeType], relation: bool) -> str:
        if not foreign_tribes:
            return ""

        text = "\n\nForeign Tribes:"
        status_order = [DiplomaticStatus.ALLY, DiplomaticStatus.NEUTRAL, DiplomaticStatus.ENEMY]
        tribes_by_status = {status: [] for status in status_order}

        for tribe in foreign_tribes:
            tribes_by_status[tribe.diplomatic_status].append(tribe)

        for status in status_order:
            if tribes_by_status[status]:
                text += f"\n\n{status.value.title()}:"
                for tribe in tribes_by_status[status]:
                    text += TextFormatter._format_foreign_tribe(tribe, relation)
        return text

    @staticmethod
    def _format_foreign_tribe(tribe: ForeignTribeType, relation: bool) -> str:
        foreign_type = get_tribe_orientation(tribe.development, tribe.stance)
        text = f"\n\n{tribe.name}\n  {foreign_type}\n{tribe.description}"

        if tribe.leaders:
            text += "\nLeaders:"
            for leader in tribe.leaders:
                text += TextFormatter._format_leader(leader, relation)
        return text

    @staticmethod
    def export_tribe_only(tribe: TribeType, leaders: List[Character], foreign_tribes: List[ForeignTribeType]) -> str:
        text = TextFormatter.format_tribe_description(tribe, leaders, False)
        text += TextFormatter.format_foreign_tribes(foreign_tribes, False)
        return text

    @staticmethod
    def export_relationships_only(leaders: List[Character],
                                  foreign_tribes: List[ForeignTribeType]) -> str:
        text = ""

        # Player tribe relationships
        for leader in leaders:
            if leader.relationships:
                text += f"\n{leader.name} ({leader.title}):\n"
                for rel in leader.relationships:
                    sentiment = TextFormatter.get_sentiment(rel.opinion)
                    text += f"  - {rel.type} relationship with {rel.target} ({sentiment}, {rel.opinion})\n"

        # Foreign tribe relationships
        for foreign_tribe in foreign_tribes:
            if foreign_tribe.leaders:
                for leader in foreign_tribe.leaders:
                    if leader.relationships:
                        text += f"\n{leader.name} ({leader.title}) from {foreign_tribe.name}:\n"
                        for rel in leader.relationships:
                            sentiment = TextFormatter.get_sentiment(rel.opinion)
                            text += f"  - {rel.type} relationship with {rel.target} ({sentiment}, {rel.opinion})\n"

        return text


# Class for managing game history
class GameHistory:
    def __init__(self):
        self.history: List['GameState'] = []

    def add_state(self, state: 'GameState'):
        self.history.append(copy.deepcopy(state))

    def get_recent_history(self) -> str:
        formatted_events = []
        if self.history:
            for event in reversed(self.history):
                turn_header = f"=== Turn {event.turn} ==="
                formatted_events.extend([turn_header, event.event_result])
        return "\n\n".join(formatted_events)

    def get_recent_short_history(self, num_events: int = 5) -> str:
        recent_events = self.history[-num_events - 1:-1][-num_events:]
        return "\n\n".join([
            f"- {event.event_result} (Outcome: {OutcomeType(event.last_outcome).name if event.last_outcome else 'N/A'})"
            for event in recent_events
        ])


# Main GameState class
class GameStateBase(BaseModel):
    tribe: TribeType
    leaders: List[Character]
    foreign_tribes: List[ForeignTribeType]
    situation: str
    event_result: str


class GameState(GameStateBase):
    previous_situation: Optional[str] = None
    current_action_choices: Optional['NextChoices'] = None
    chosen_action: Optional['ActionChoice'] = None
    last_outcome: Optional[OutcomeType] = None
    turn: int = 1
    streak_count: int = 0
    tier: int = 1
    power: int = 5

    # Composition instead of inheritance
    text_formatter: TextFormatter = Field(default_factory=lambda: TextFormatter(), exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def add_action_set(self, actions: List[str]):
        self.action_history.add_action_set(actions)

    def to_context_string(self) -> str:
        tribe_text = self.text_formatter.format_tribe_description(self.tribe, self.leaders, True)
        foreign_tribes_text = self.text_formatter.format_foreign_tribes(self.foreign_tribes, True)
        return tribe_text + foreign_tribes_text

    def tribe_string(self) -> str:
        text = self.text_formatter.export_tribe_only(self.tribe, self.leaders, self.foreign_tribes)
        return text

    def relationship_string(self) -> str:
        text = self.text_formatter.export_relationships_only(self.leaders, self.foreign_tribes)
        return text

    @classmethod
    def initialize(cls, tribe: TribeType, leader: Character) -> 'GameState':
        return cls(
            tribe=tribe,
            leaders=[leader],
            foreign_tribes=[],
            situation="",
            event_result=""
        )

    def update(self, new_state: GameStateBase):
        for field in GameStateBase.model_fields:
            setattr(self, field, getattr(new_state, field))

    def update_streak(self, outcome: OutcomeType) -> None:
        """Updates the streak counter based on the outcome."""
        if outcome == OutcomeType.POSITIVE:
            self.streak_count += 1
        else:
            self.streak_count = 0

    def adjust_power(self, outcome: OutcomeType) -> None:
        """Adjusts power level based on the outcome type."""
        power_adjustments = {
            OutcomeType.POSITIVE: 2,
            OutcomeType.NEUTRAL: 0,
            OutcomeType.NEGATIVE: -1,
            OutcomeType.CATASTROPHE: -3
        }
        self.power += power_adjustments[outcome]

    def update_tier(self) -> None:
        """Updates the tier based on the current power level."""
        self.tier = min(self.power // 10, 4) + 1

    def check_streak_catastrophe(self) -> bool:
        """Checks if streak has reached catastrophe level."""
        if self.streak_count >= 4:
            self.streak_count = 0
            return True
        return False


class GameStateManager:
    def __init__(self, llm_context: LLMContext):
        self.current_game_state: Optional[GameState] = None
        self.game_history = GameHistory()
        self.llm_context = llm_context
        self.current_tribe_choices = None

    def reset(self):
        self.current_game_state = None
        self.game_history = GameHistory()

    def initialize(self, tribe: TribeType, leader: Character):
        self.current_game_state = GameState.initialize(tribe, leader)
        self.game_history = GameHistory()

    def perform_action(self, index: int):
        action = self.current_game_state.current_action_choices.choices[index]
        self.current_game_state.chosen_action = action
        self.current_game_state.previous_situation = self.current_game_state.situation
        outcome, roll = self.determine_outcome(action.probability)
        print(f"Debug: Roll = {roll:.2f}, Probability = {action.probability:.2f}, Outcome = {outcome.name}")
        # Handle streak catastrophe
        if self.current_game_state.check_streak_catastrophe():
            outcome = OutcomeType.CATASTROPHE

        # Update game state
        self.current_game_state.update_streak(outcome)
        self.current_game_state.adjust_power(outcome)
        self.current_game_state.update_tier()
        self.current_game_state.last_outcome = outcome
        self.update_game_state()
        event_type = self.decide_event_type()
        print(f"Event type: {event_type}")
        self.generate_choices(event_type)
        self.game_history.add_state(self.current_game_state)
        base_filename = "Debug_History"
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Create unique filename
        filename = f"{base_filename}_turn{self.current_game_state.turn}_{timestamp}.json"
        self.save_all_game_states(filename)
        self.current_game_state.turn += 1

    def save_all_game_states(self, filename: str):
        with open(filename, 'w') as f:
            json.dump(self.game_history.history, f, cls=EnumEncoder)

        # Also write the story version
        filename = filename.replace('.json', '_story.txt')
        story = []
        for state in self.game_history.history:
            story.append(f"=== Turn {state.turn} ===")
            story.append(f"Situation: {state.previous_situation}")
            story.append(f"Taken action: {state.chosen_action.description}")
            outcome = OutcomeType(state.last_outcome).name
            story.append(f"The result was: {outcome} and resulted in: {state.event_result}\n")

        with open(filename, 'w') as f:
            f.write("\n".join(story))

    def load_game_state(self, filename: str):
        with open(filename, 'r') as f:
            loaded_history = json.load(f)

        # Convert enum dictionaries to actual enum values
        converted_history = convert_enums(loaded_history)
        # Convert each history entry to a GameState object
        history: List[GameState] = [GameState(**state) for state in converted_history]
        # Set the current game state to the last item in the history
        self.current_game_state = copy.deepcopy(history[-1])

        # Add all states to the history
        for state in history:
            self.game_history.add_state(state)
        self.current_game_state.turn += 1

    def generate_tribe_choices(self, external_choice:str = "") -> InitialChoices:
        # Generate 3 unique combinations of development type and diplomatic stance

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
        tribe1 = ""
        if external_choice != "":
            tribe1 = f"- Must be {external_choice}. Mention the race in the description."
        else:
            tribe1 = f"""- Must be {selected_combinations[0][0].value} in development and {selected_combinations[0][1].value} in diplomacy
        - Should fit the description: {orientations[0]}"""

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
    {tribe1}

    Tribe 2:
    - Must be {selected_combinations[1][0].value} in development and {selected_combinations[1][1].value} in diplomacy
    - Should fit the description: {orientations[1]}

    Tribe 3:
    - Must be {selected_combinations[2][0].value} in development and {selected_combinations[2][1].value} in diplomacy
    - Should fit the description: {orientations[2]}

    One tribe should consist of {external_choice}.
    For each tribe provide:
    1. A unique tribe name (do not use 'Sylvan')
    2. A description of the tribe without mentioning the tribe name, and do not quote the given description directly, but name the race
    3. Its DevelopmentType
    4. Its DiplomaticStance"""
            }
        ]

        choices = self.llm_context.make_story_call(messages, InitialChoices, max_tokens=2000)

        if choices:
            self.current_tribe_choices = choices
            print("\n=== Available Tribe Choices ===")
            for number, (choice, (dev_type, stance)) in enumerate(zip(choices.choices, selected_combinations)):
                tribe_type = get_tribe_orientation(dev_type, stance)
                print(f"\n{number}. {choice.name}, {tribe_type}")
                print(textwrap.fill(f"Description: {choice.description}", width=80, initial_indent="   ",
                                    subsequent_indent="   "))

        return choices

    def get_leader(self, chosen_tribe: TribeType) -> Character:
        messages = [
            {
                "role": "system",
                "content": "You are the game master for a text based strategic game, where the player rules over a tribe in a fantasy based setting. Output your answers as JSON"
            },
            {
                "role": "user",
                "content": f"""Give me the name and title for the leader of {chosen_tribe.name}. Their description is:
{chosen_tribe.description}. The leader does not have relationships with other characters."""
            }
        ]

        leader = self.llm_context.make_story_call(messages, Character, max_tokens=2000)

        if leader:
            print(textwrap.fill(f"\nLeader: {leader}", width=80, initial_indent="   ",
                                subsequent_indent="   "))

        return leader

    def get_probability_adjustment(self) -> str:
        if self.current_game_state.last_outcome not in [OutcomeType.POSITIVE]:
            return "Due to the recent neutral or negative outcome, provide higher probabilities of success for the next choices (between 0.6 and 0.9)."
        return "Provide balanced probabilities of success for the next choices (between 0.5 and 0.8)."

    def generate_choices(self, choice_type: EventType) -> None:
        recent_history = self.game_history.get_recent_short_history(num_events=3)
        prob_adjustment = self.get_probability_adjustment()

        action_context = ""
        if self.current_game_state.chosen_action:
            action_context = f"""Action: {self.current_game_state.chosen_action.caption}
    {self.current_game_state.chosen_action.description}

    Outcome: {self.current_game_state.event_result}"""

        choice_types = {
            EventType.SINGLE_EVENT: {"instruction": "create a new event", "extra": ""},
            EventType.REACTION: {"instruction": "create a reaction to this last outcome",
                                 "extra": "Include a foreign character in the reaction."},
            EventType.FOLLOWUP: {"instruction": "create a follow up to this last outcome",
                                 "extra": "Keep choices thematically close to the last action."}
        }

        if choice_type not in choice_types:
            raise ValueError(f"Invalid choice type: {choice_type}")

        choice_type_fields = choice_types[choice_type]

        context = f"""State: {self.current_game_state.to_context_string()}
History: {recent_history}
Previous: {self.current_game_state.previous_situation}

{action_context}"""
        instruction = f"""{choice_type_fields['instruction']} and add to:
{self.current_game_state.situation}
Save a summary in "situation".

Present 3 possible next actions:
{choice_type_fields['extra']}
Each has a caption, description, and probability of success 
Include consequences and strategic considerations in the description.
{prob_adjustment}
- One choice should have high probability
- Include at least one aggressive option (probability based on tribe info)"""

        summary = self.llm_context.get_summary(context)

        messages2 = [
            {
                "role": "system",
                "content": "You are a game master for a tribal fantasy strategy game. Output answers as JSON"
            },
            {
                "role": "user",
                "content": summary + instruction}]

        #next_choices = self.llm_context.make_api_call(messages1, NextReactionChoices, max_tokens=2000)
        next_choices = self.llm_context.make_story_call(messages2, NextReactionChoices, max_tokens=2000)
        if next_choices:
            self.current_game_state.situation = next_choices.situation
            self.current_game_state.current_action_choices = NextChoices(choices=next_choices.choices)

            print("\n=== Available Actions ===")
            for i, choice in enumerate(next_choices.choices, 1):
                print(f"\n{i}. {choice.caption}")
                print(textwrap.fill(f"Description: {choice.description}",
                                    width=80,
                                    initial_indent="   ",
                                    subsequent_indent="   "))
                print(f"   Probability of Success: {choice.probability:.2f}")

    def update_game_state(self) -> None:
        recent_history = self.game_history.get_recent_short_history(num_events=5)
        tribes_prompt = ""
        enemy_count = sum(
            1 for tribe in self.current_game_state.foreign_tribes if tribe.diplomatic_status == DiplomaticStatus.ENEMY)
        neutral_count = sum(
            1 for tribe in self.current_game_state.foreign_tribes if
            tribe.diplomatic_status == DiplomaticStatus.NEUTRAL)

        if self.current_game_state.turn > 5 and enemy_count < 1:
            tribes_prompt += "* Add one or two enemy factions\n"
        if self.current_game_state.turn > 3 and neutral_count < 2:
            tribes_prompt += "* Add one or two neutral factions\n"

        context = f"""State: {self.current_game_state.to_context_string()}
Development: {self.current_game_state.tribe.development.value}
Stance: {self.current_game_state.tribe.stance.value}

History: {recent_history}
Previous: {self.current_game_state.previous_situation}
Action: {self.current_game_state.chosen_action.caption} - {self.current_game_state.chosen_action.description}
Outcome: {self.current_game_state.last_outcome.name}"""
        instructions = f"""Required Updates:

1. Leaders (max 5):
   - Update names, titles and relationships, both to tribe leaders as well as to leaders of foreign tribes
   - Add special roles if warranted
   - Only major changes based on events

2. Situation:
   - Current state summary

3. Foreign Relations:
   - For each tribe: status, name, description, development, stance, leaders (max 2)
   - For each leader pf a foreign tribe: Names, titles, relationships to the leaders of the players tribe
   - Status changes require multiple turns
   - Consider development/stance compatibility
   {tribes_prompt}
4. Event Results:
   - 2 paragraphs narrative (event_result)
   - Events may continue

Maintain consistency with game state, action outcomes, and existing relationships."""
        summary = self.llm_context.get_summary(context)
        messages = [
            {
                "role": "system",
                "content": "You are the game master for a text-based strategic game, where the player rules over a tribe in a fantasy setting. Your task is to update the game state based on the player's actions and their outcomes."
            },
            {
                "role": "user",
                "content": summary + instructions
            }
        ]

        new_state = self.llm_context.make_story_call(messages, GameStateBase, max_tokens=5000)
        self.current_game_state.update(new_state)


    @staticmethod
    def determine_outcome(probability: float) -> Tuple[OutcomeType, float]:
        roll = round(random.random(),2)
        # roll = 0.0
        # roll = 1.0
        if roll >= 0.95:
            return OutcomeType.CATASTROPHE, roll
        if roll <= probability:
            return OutcomeType.POSITIVE, roll
        elif roll < (probability + (1 - probability) / 2):
            return OutcomeType.NEUTRAL, roll
        else:
            return OutcomeType.NEGATIVE, roll

    @staticmethod
    def decide_event_type() -> EventType:
        return random.choices(
            [EventType.SINGLE_EVENT, EventType.REACTION, EventType.FOLLOWUP],
            weights=[0.2, 0.4, 0.4], k=1)[0]


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


def create_gui():
    config = Config.default_config()
    # Use the function to initialize the LLMContext
    llm_context = LLMContext(config)
    game_manager = GameStateManager(llm_context)

    def perform_action(action_index):
        game_manager.perform_action(action_index)
        return update_game_display()

    def update_game_display():
        if not game_manager.current_game_state:
            return (
                gr.update(visible=False, value="No active game"),
                gr.update(visible=False, value="No history"),
                gr.update(visible=False, value="No relationships"),
                gr.update(visible=False, value="No current situation"),
                *[gr.update(visible=False)] * 6
            )

        tribe_overview = game_manager.current_game_state.tribe_string()
        relationships = game_manager.current_game_state.relationship_string()
        recent_history = game_manager.game_history.get_recent_history()
        current_situation = game_manager.current_game_state.situation

        if game_manager.current_game_state.current_action_choices is not None:
            choices = game_manager.current_game_state.current_action_choices.choices
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
            gr.update(visible=True, value=relationships),
            gr.update(visible=True, value=current_situation),
            *updates
        )

    def save_current_game():
        game_manager.save_all_game_states('test.json')
        return "Game saved successfully!"

    def load_saved_game(filename):
        try:
            game_manager.load_game_state(filename)
            current_action_choices = game_manager.current_game_state.current_action_choices

            # Update all necessary GUI elements
            tribe_overview = game_manager.current_game_state.tribe_string()
            relationships = game_manager.current_game_state.relationship_string()
            recent_history = game_manager.game_history.get_recent_history()
            current_situation = game_manager.current_game_state.situation

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
                gr.update(visible=True, value=relationships),  # Show and update relationships
                gr.update(visible=True, value=current_situation),  # Show and update current situation
                *action_updates
            )
        except FileNotFoundError:
            return (
                "No saved game found.",
                gr.update(visible=True),  # Show tribe selection group
                gr.update(visible=False),  # Keep tribe overview hidden
                gr.update(visible=False),  # Keep recent history hidden
                gr.update(visible=False),  # Keep relationships hidden
                gr.update(visible=False),  # Keep current situation hidden
                *[gr.update(visible=False)] * 6
            )

    def generate_tribes(theme):
        game_manager.reset()
        game_manager.generate_tribe_choices(theme)
        if game_manager.current_tribe_choices:
            result = ""
            for number, choice in enumerate(game_manager.current_tribe_choices.choices):
                tribe_type = get_tribe_orientation(choice.development, choice.stance)
                result += f"{number + 1}. {choice.name}, "
                result += f"{tribe_type}\n"
                result += f"{choice.description}\n\n"

            return result, gr.update(visible=True, choices=[1, 2, 3])
        else:
            return "Error generating initial choices. Please try again.", gr.update(visible=False)

    def start_game(choice):
        game_manager.reset()
        if game_manager.current_tribe_choices:
            chosen_tribe = next(
                (tribe for index, tribe in enumerate(game_manager.current_tribe_choices.choices, 1) if index == choice),
                None)
            if chosen_tribe:
                leader = game_manager.get_leader(chosen_tribe)
                game_manager.initialize(chosen_tribe, leader)
                game_manager.current_game_state.previous_situation = "Humble beginnings of the " + chosen_tribe.name
                game_manager.current_game_state.last_outcome = OutcomeType.NEUTRAL
                game_manager.generate_choices(EventType.SINGLE_EVENT)

                # Unpack the return values from update_game_display
                tribe_overview, recent_history, relationships, current_situation, *action_updates = update_game_display()

                return (
                    gr.update(visible=False),  # Hide tribe selection group
                    tribe_overview,
                    recent_history,
                    relationships,
                    current_situation,
                    *action_updates
                )
            else:
                return (
                    gr.update(visible=True),
                    gr.update(visible=False, value="Invalid choice. Please select a number between 1 and 3."),
                    gr.update(visible=False),
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
                gr.update(visible=False),
                *[gr.update(visible=False)] * 6
            )

    # Set up your Gradio interface here, using the above functions
    with gr.Blocks(title="Fantasy Tribe Game") as app:
        gr.Markdown("# Fantasy Tribe Game")

        # Game Tab
        with gr.Tab("Game"):
            # Create tribe selection group

            with gr.Group() as tribe_selection_group:
                race_theme = gr.Dropdown(
                    choices=["", "Men", "High Elves ", "Wood Elves", "Dark Elves", "Dwarves", "Halflings", "Orcs",
                             "Goblins", "Gnomes", "Trolls", "Ogres", "Kobolds", "Skaven", "Vampires", "Lycanthropes",
                             "Giants", "Valkyries", "Norns", "Nephilim", "Fairies", "Sprites", "Pixies", "Changelings",
                             "Angels", "Demons", "Celestials", "Fauns", "Ghuls", "Ifrits", "Unicorns", "Griffin",
                             "Wyverns", "Centaurs", "Minotaurs", "Merfolk", "Naga", "Djinn", "Aasimar", "Tieflings",
                             "Dragonborn", "Kitsune", "Tengu", "Sylphs", "Dryads", "Nymphs", "Harpies", "Satyrs",
                             "Phoenix", "Basilisks", "Chimeras", "Gryphons", "Liches", "Elementals", "Golems",
                             "Gargoyles", "Wendigo", "Yokai", "Rakshasa", "Selkies", "Banshees", "Revenants",
                             "Succubi/Incubi", "Doppelgangers", "Wraiths", "Fomorians", "Firbolg", "Lizardfolk",
                             "Kenku", "Aarakocra", "Tabaxi", "Yuan-ti", "Genasi", "Warforged", "Automata"],
                    value="",
                    label="Choose theme to influence race generation"
                )
                start_button = gr.Button("Generate Tribe Choices")
                tribe_choices = gr.Textbox(label="Available Tribes", lines=10)
                tribe_selection = gr.Radio(choices=[1, 2, 3], label="Select your tribe", visible=False)
                select_tribe_button = gr.Button("Select Tribe")

            with gr.Row():
                with gr.Column(scale=1):
                    # Initialize these elements as hidden
                    tribe_overview = gr.Textbox(label="Tribe Overview", lines=10, visible=False)
                    relationships = gr.Textbox(label="Relationships", lines=5, visible=False)
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
                gr.Markdown("## Story LLM Configuration")
                story_provider_dropdown = gr.Dropdown(
                    choices=[provider.value for provider in LLMProvider],
                    value=config.story_config.provider.value,
                    label="Story LLM Provider"
                )
                story_model_dropdown = gr.Dropdown(
                    choices=["claude-3-5-sonnet-20240620"],
                    value="claude-3-5-sonnet-20240620",
                    label="Story Model (Anthropic)",
                    visible=config.story_config.provider == LLMProvider.ANTHROPIC
                )
                story_model_openai_dropdown = gr.Dropdown(
                    choices=["gpt-4o", "gpt-4o-mini"],
                    value="gpt-4o",
                    label="Story Model (OpenAI)",
                    visible=config.story_config.provider == LLMProvider.OPENAI
                )
                story_model_local_input = gr.Textbox(
                    value=config.story_config.model_name if config.story_config.provider == LLMProvider.LOCAL else "Qwen2.5-14B-Instruct:latest",
                    label="Story Model (Local)",
                    placeholder="Enter model name",
                    visible=config.story_config.provider == LLMProvider.LOCAL
                )
                story_local_url_input = gr.Textbox(
                    value=config.story_config.local_url or "http://127.0.0.1:1234/v1",
                    label="Story Local API URL",
                    placeholder="http://127.0.0.1:1234/v1",
                    visible=config.story_config.provider == LLMProvider.LOCAL
                )

            with gr.Group():
                gr.Markdown("## Summary LLM Configuration")
                summary_provider_dropdown = gr.Dropdown(
                    choices=[provider.value for provider in LLMProvider],
                    value=config.summary_config.provider.value,
                    label="Summary LLM Provider"
                )
                summary_model_dropdown = gr.Dropdown(
                    choices=["claude-3-5-sonnet-20240620"],
                    value="claude-3-5-sonnet-20240620",
                    label="Summary Model (Anthropic)",
                    visible=config.summary_config.provider == LLMProvider.ANTHROPIC
                )
                summary_model_openai_dropdown = gr.Dropdown(
                    choices=["gpt-4o", "gpt-4o-mini"],
                    value="gpt-4o-mini",
                    label="Summary Model (OpenAI)",
                    visible=config.summary_config.provider == LLMProvider.OPENAI
                )
                summary_model_local_input = gr.Textbox(
                    value=config.summary_config.model_name if config.summary_config.provider == LLMProvider.LOCAL else "Qwen2.5-14B-Instruct:latest",
                    label="Summary Model (Local)",
                    placeholder="Enter model name",
                    visible=config.summary_config.provider == LLMProvider.LOCAL
                )
                summary_local_url_input = gr.Textbox(
                    value=config.summary_config.local_url or "http://127.0.0.1:1234/v1",
                    label="Summary Local API URL",
                    placeholder="http://127.0.0.1:1234/v1",
                    visible=config.summary_config.provider == LLMProvider.LOCAL
                )
                summary_mode_dropdown = gr.Dropdown(
                    choices=[mode.value for mode in SummaryMode],
                    value=config.summary_config.mode.value,
                    label="Summary Mode"
                )

            settings_save_btn = gr.Button("Save Settings")
            settings_message = gr.Textbox(label="Settings Status", interactive=False)

            # Add visibility rules for model and URL inputs
            def update_story_input_visibility(provider):
                return {
                    story_model_dropdown: gr.update(visible=provider == LLMProvider.ANTHROPIC.value),
                    story_model_openai_dropdown: gr.update(visible=provider == LLMProvider.OPENAI.value),
                    story_model_local_input: gr.update(visible=provider == LLMProvider.LOCAL.value),
                    story_local_url_input: gr.update(visible=provider == LLMProvider.LOCAL.value)
                }

            story_provider_dropdown.change(
                update_story_input_visibility,
                inputs=[story_provider_dropdown],
                outputs=[story_model_dropdown, story_model_openai_dropdown, story_model_local_input,
                         story_local_url_input]
            )

            def update_summary_input_visibility(provider):
                return {
                    summary_model_dropdown: gr.update(visible=provider == LLMProvider.ANTHROPIC.value),
                    summary_model_openai_dropdown: gr.update(visible=provider == LLMProvider.OPENAI.value),
                    summary_model_local_input: gr.update(visible=provider == LLMProvider.LOCAL.value),
                    summary_local_url_input: gr.update(visible=provider == LLMProvider.LOCAL.value)
                }

            summary_provider_dropdown.change(
                update_summary_input_visibility,
                inputs=[summary_provider_dropdown],
                outputs=[summary_model_dropdown, summary_model_openai_dropdown, summary_model_local_input,
                         summary_local_url_input]
            )

            # Function to save settings
            def save_settings(story_provider, story_model_anthropic, story_model_openai, story_model_local,
                              story_local_url,
                              summary_provider, summary_model_anthropic, summary_model_openai, summary_model_local,
                              summary_local_url, summary_mode):
                story_model = story_model_anthropic if story_provider == LLMProvider.ANTHROPIC.value else \
                    story_model_openai if story_provider == LLMProvider.OPENAI.value else \
                        story_model_local
                summary_model = summary_model_anthropic if summary_provider == LLMProvider.ANTHROPIC.value else \
                    summary_model_openai if summary_provider == LLMProvider.OPENAI.value else \
                        summary_model_local

                new_config = Config(
                    story_config=ModelConfig(
                        provider=LLMProvider(story_provider),
                        model_name=story_model,
                        local_url=story_local_url if story_provider == LLMProvider.LOCAL.value else None
                    ),
                    summary_config=SummaryModelConfig(
                        provider=LLMProvider(summary_provider),
                        model_name=summary_model,
                        mode=SummaryMode(summary_mode),
                        local_url=summary_local_url if summary_provider == LLMProvider.LOCAL.value else None
                    )
                )
                llm_context.update(new_config)
                return "Settings saved successfully!"

            settings_save_btn.click(
                save_settings,
                inputs=[story_provider_dropdown, story_model_dropdown, story_model_openai_dropdown,
                        story_model_local_input, story_local_url_input,
                        summary_provider_dropdown, summary_model_dropdown, summary_model_openai_dropdown,
                        summary_model_local_input, summary_local_url_input,
                        summary_mode_dropdown],
                outputs=[settings_message]
            )


        start_button.click(
            generate_tribes,
            inputs=[race_theme],  # Add the dropdown as input
            outputs=[tribe_choices, tribe_selection]
        )

        select_tribe_button.click(
            start_game,
            inputs=[tribe_selection],
            outputs=[
                tribe_selection_group,
                tribe_overview,
                recent_history,
                relationships,
                current_situation,
                action_button1, action_desc1,
                action_button2, action_desc2,
                action_button3, action_desc3
            ]
        )

        action_button1.click(
            lambda: perform_action(0),
            outputs=[
                tribe_overview, recent_history,
                relationships, current_situation,
                action_button1, action_desc1,
                action_button2, action_desc2,
                action_button3, action_desc3
            ]
        )

        action_button2.click(
            lambda: perform_action(1),
            outputs=[
                tribe_overview, recent_history,
                relationships, current_situation,
                action_button1, action_desc1,
                action_button2, action_desc2,
                action_button3, action_desc3
            ]
        )

        action_button3.click(
            lambda: perform_action(2),
            outputs=[
                tribe_overview, recent_history,
                relationships, current_situation,
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
                relationships,
                current_situation,
                action_button1, action_desc1,
                action_button2, action_desc2,
                action_button3, action_desc3
            ]
        )

    return app


if __name__ == "__main__":
    interface = create_gui()
    interface.launch()
