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
import os
import gradio as gr
from datetime import datetime

from typing import List, Literal,  Tuple, Optional, Dict, Any

from pydantic import BaseModel, Field
from Language import Language, translations
from LLM_manager import (LLMContext, Config, ModelConfig,
                         SummaryModelConfig, LLMProvider, SummaryMode)
from prompt_generator import generate_game_prompt

SUPPORTED_LANGUAGES = ["english", "german"]
random.seed(datetime.now().timestamp())


class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, BaseModel):
            return obj.model_dump(exclude_unset=True)
        return json.JSONEncoder.default(self, obj)


OutcomeType = Literal["positive", "neutral", "negative", "catastrophe"]
OUTCOME_TYPES = ["positive", "neutral", "negative", "catastrophe"]
EventType = Literal["single_event", "reaction", "followup"]
EVENT_TYPES = ["single_event", "reaction", "followup"]

DevelopmentType = Literal["magical", "hybrid", "practical"]
DiplomaticStance = Literal["peaceful", "neutral", "aggressive"]
DEVELOPMENT_TYPES = ["magical", "hybrid", "practical"]
DIPLOMATIC_STANCES = ["peaceful", "neutral", "aggressive"]


DiplomaticStatus = Literal["ally", "neutral", "enemy"]
DIPLOMATIC_STATUS = ["ally", "neutral", "enemy"]


class ActionChoice(BaseModel):
    caption: str
    description: str
    probability: float


class NextChoices(BaseModel):
    choices: List[ActionChoice]


class Relationship(BaseModel):
    target: str
    type: str
    opinion: int  # -100 to 100


class StartCharacter(BaseModel):
    name: str
    title: str


class Character(BaseModel):
    name: str
    title: str
    relationships: Optional[List[Relationship]]
    #relationships: List[Relationship]


class ForeignCharacter(Character):
    tribe: str


class TribeType(BaseModel):
    name: str
    description: str
    development: DevelopmentType
    stance: DiplomaticStance

class TribesResponse(BaseModel):
    tribes: List[TribeType]

class ForeignTribeType(TribeType):
    diplomatic_status: DiplomaticStatus
    leaders: List[Character]


DIPLOMATIC_STATUS_TRANSLATIONS = {
    "english": {
        "ally": "Ally",
        "neutral": "Neutral",
        "enemy": "Enemy",
    },
    "german": {
        "ally": "Verbündeter",
        "neutral": "Neutral",
        "enemy": "Feind",
    },
}


def get_tribe_orientation(development: DevelopmentType, stance: DiplomaticStance, language: Language) -> str:
    """
    Get the orientation-specific description for a tribe based on their development path and diplomatic stance.
    """
    descriptions = {
        "magical": {
            "peaceful": {
                "english": "Mystic sages who commune with the very essence of nature",
                "german": "Mystische Weise, die mit dem Wesen der Natur in Verbindung stehen"
            },
            "neutral": {
                "english": "Arcane scholars who balance the powers of magic and wisdom",
                "german": "Arkane Gelehrte, die die Kräfte der Magie und Weisheit im Gleichgewicht halten"
            },
            "aggressive": {
                "english": "Battle-mages who wield the raw forces of destructive magic",
                "german": "Kampfmagier, die die rohen Kräfte zerstörerischer Magie beherrschen"
            }
        },
        "hybrid": {
            "peaceful": {
                "english": "Technomancers who weave together the threads of science and spirit",
                "german": "Technomanten, die die Fäden von Wissenschaft und Geist verweben"
            },
            "neutral": {
                "english": "Arcane engineers who forge marvels of magic and machinery",
                "german": "Arkane Ingenieure, die Wunderwerke aus Magie und Maschinerie erschaffen"
            },
            "aggressive": {
                "english": "Magitech warriors who harness both arcane energy and steel",
                "german": "Magitech-Krieger, die sowohl arkane Energie als auch Stahl beherrschen"
            }
        },
        "practical": {
            "peaceful": {
                "english": "Master builders dedicated to the advancement of technology",
                "german": "Meisterbaumeister, die sich dem technologischen Fortschritt verschrieben haben"
            },
            "neutral": {
                "english": "Innovators who forge a path of progress and strength",
                "german": "Innovatoren, die einen Weg des Fortschritts und der Stärke beschreiten"
            },
            "aggressive": {
                "english": "Warriors at the forefront of military innovation and might",
                "german": "Krieger an der Spitze militärischer Innovation und Macht"
            }
        }
    }

    return descriptions[development][stance][language]


class TextFormatter:
    @staticmethod
    def get_sentiment(opinion: int, language: Literal["english", "german"]) -> str:
        sentiments = {
            "english": {
                90: "devoted",
                70: "strongly positive",
                40: "favorable",
                10: "somewhat positive",
                -10: "neutral",
                -40: "somewhat negative",
                -70: "unfavorable",
                -90: "strongly negative",
                float('-inf'): "hostile"
            },
            "german": {
                90: "ergeben",
                70: "sehr positiv",
                40: "günstig",
                10: "etwas positiv",
                -10: "neutral",
                -40: "etwas negativ",
                -70: "ungünstig",
                -90: "sehr negativ",
                float('-inf'): "feindselig"
            }
        }

        thresholds = [90, 70, 40, 10, -10, -40, -70, -90, float('-inf')]

        for threshold in thresholds:
            if opinion >= threshold:
                return sentiments[language][threshold]

        return sentiments[language][float('-inf')]  # fallback case

    @staticmethod
    def format_tribe_description(tribe: TribeType, leaders: List[Character], relation: bool, language: Language) -> str:
        tribe_type = get_tribe_orientation(tribe.development, tribe.stance, language)
        text = f"""{tribe.name}
{tribe_type}

{tribe.description}

{"Leaders:" if language == "english" else "Anführer:"}"""

        for leader in leaders:
            text += TextFormatter._format_leader(leader, relation, language)
        return text

    @staticmethod
    def format_tribe_description_llm(tribe: TribeType, leaders: List[Character], relation: bool) -> str:
        tribe_type = get_tribe_orientation(tribe.development, tribe.stance, "english")
        text = f"""Tribe name: {tribe.name}
Tribe development type: {tribe.development}
Tribe stance: {tribe.stance}
Resulting tribe type: {tribe_type}

Tribe description: {tribe.description}

{"Leaders:"}"""
        for leader in leaders:
            text += TextFormatter._format_leader(leader, relation, "english")
        return text

    @staticmethod
    def _format_leader(leader: Character, relation: bool, language: Language) -> str:
        text = f"\n  • {leader.title}: {leader.name}"
        if leader.relationships and relation:
            text += "\n    " + ("Relationships:" if language == "english" else "Beziehungen:")
            for rel in leader.relationships:
                sentiment = TextFormatter.get_sentiment(rel.opinion, language)
                if language == "english":
                    text += f"\n      - {rel.type} relationship with  {rel.target}, {sentiment}: {rel.opinion}"
                elif language == "german":
                    text += f"\n      - Beziehung zu {rel.target}: {rel.type}, {sentiment}: {rel.opinion}"
        return text

    @staticmethod
    def format_foreign_tribes(foreign_tribes: List[ForeignTribeType], relation: bool, language: Language) -> str:
        if not foreign_tribes:
            return ""

        text = "\n\n" + ("Foreign Tribes:" if language == "english" else "Fremde Stämme:")
        status_order = ["ally", "neutral", "enemy"]
        tribes_by_status = {status: [] for status in status_order}

        for tribe in foreign_tribes:
            tribes_by_status[tribe.diplomatic_status].append(tribe)

        for status in status_order:
            if tribes_by_status[status]:
                status_translation = DIPLOMATIC_STATUS_TRANSLATIONS[language][status]
                text += f"\n\n{status_translation}:"
                for tribe in tribes_by_status[status]:
                    text += TextFormatter._format_foreign_tribe(tribe, relation, language)
        return text

    @staticmethod
    def _format_foreign_tribe(tribe: ForeignTribeType, relation: bool, language: Language) -> str:
        foreign_type = get_tribe_orientation(tribe.development, tribe.stance, language)
        text = f"\n\n{tribe.name}\n  {foreign_type}\n{tribe.description}"

        if tribe.leaders:
            text += "\n" + ("Leaders:" if language == "english" else "Anführer:")
            for leader in tribe.leaders:
                text += TextFormatter._format_leader(leader, relation, language)
        return text

    @staticmethod
    def export_tribe_only(tribe: TribeType, leaders: List[Character],
                          foreign_tribes: List[ForeignTribeType], language: Language) -> str:
        text = TextFormatter.format_tribe_description(tribe, leaders, False, language)
        text += TextFormatter.format_foreign_tribes(foreign_tribes, False, language)
        return text

    @staticmethod
    def export_relationships_only(leaders: List[Character],
                                  foreign_tribes: List[ForeignTribeType], language: Language) -> str:
        text = ""

        # Player tribe relationships
        for leader in leaders:
            if leader.relationships:
                text += f"\n{leader.name} ({leader.title}):\n"
                for rel in leader.relationships:
                    sentiment = TextFormatter.get_sentiment(rel.opinion, language)
                    if language == "english":
                        text += f"  - {rel.type} relationship with  {rel.target}, {sentiment}\n"
                    elif language == "german":
                        text += f"  - Beziehung zu {rel.target}: {rel.type}, {sentiment}\n"

        # Foreign tribe relationships
        for foreign_tribe in foreign_tribes:
            if foreign_tribe.leaders:
                for leader in foreign_tribe.leaders:
                    if leader.relationships:
                        text += f"\n{leader.name} ({leader.title}) {'from' if language == 'english' else 'von'} {foreign_tribe.name}:\n"
                        for rel in leader.relationships:
                            sentiment = TextFormatter.get_sentiment(rel.opinion, language)
                            if language == "english":
                                text += f"  - {rel.type} relationship with  {rel.target}, {sentiment}\n"
                            elif language == "german":
                                text += f"  - Beziehung zu {rel.target}: {rel.type}, {sentiment}\n"

        return text


# Class for managing game history
class GameHistory:
    def __init__(self):
        self.history: List['GameState'] = []

    def add_state(self, state: 'GameState'):
        self.history.append(copy.deepcopy(state))

    def get_recent_history(self, language: Language) -> str:
        formatted_events = []
        if self.history:
            for event in reversed(self.history):
                turn_header = f"=== Turn {event.turn} ===" if language == "english" else f"=== Runde {event.turn} ==="
                formatted_events.extend([turn_header, event.event_result])
        return "\n\n".join(formatted_events)

    def get_recent_short_history(self, num_events: int = 5) -> str:
        recent_events = self.history[-num_events:]
        return "\n\n".join([
            f"- {event.event_result} (Outcome: {event.last_outcome if event.last_outcome else 'N/A'})"
            for event in recent_events
        ])


class HiddenGameState(BaseModel):
    secret_relationships: Dict[str, Dict[str, str]] = {}  # character -> {actual_loyalty, secret_role, etc}
    long_term_plots: List[Dict[str, Any]] = []           # list of plots with their details
    #hidden_goals: Dict[str, Dict[str, Any]] = {}         # tribe -> {goal, progress}
    prophecies: List[Dict[str, Any]] = []                # list of prophecies
    #power_struggles: Dict[str, Dict[str, Any]] = {}      # tribe -> {description, intensity}

    def to_context_string(self) -> str:
        return f"""
        Hidden Game State, not visible to the player:
        Secret Relationships: {json.dumps(self.secret_relationships, indent=2)}
        Active Plots: {json.dumps(self.long_term_plots, indent=2)}
        Prophecies: {json.dumps(self.prophecies, indent=2)}
        """
    #
    # Hidden Goals: {json.dumps(self.hidden_goals, indent=2)}
    # Power Struggles: {json.dumps(self.power_struggles, indent=2)}

# Main GameState class
class GameStateBase(BaseModel):
    tribe: TribeType
    leaders: List[Character]
    foreign_tribes: List[ForeignTribeType]
    situation: str
    event_result: str
    hidden_game_state: HiddenGameState


class CombinedStateAndChoices(GameStateBase):
    choices: List[ActionChoice]


class GameState(GameStateBase):
    previous_situation: Optional[str] = None
    current_action_choices: Optional['NextChoices'] = None
    chosen_action: Optional['ActionChoice'] = None
    last_outcome: Optional[OutcomeType] = None
    short_history: str = ""
    turn: int = 0
    streak_count: int = 0
    tier: int = 1
    power: int = 5

    # Composition instead of inheritance
    text_formatter: TextFormatter = Field(default_factory=lambda: TextFormatter(), exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def add_action_set(self, actions: List[str]):
        self.action_history.add_action_set(actions)

    def to_context_string(self, language) -> str:
        tribe_text = self.text_formatter.format_tribe_description_llm(self.tribe, self.leaders, True)
        foreign_tribes_text = self.text_formatter.format_foreign_tribes(self.foreign_tribes, True, language)
        hidden_state_text = self.hidden_game_state.to_context_string()

        return f"""
    {tribe_text}
    {foreign_tribes_text}
    {hidden_state_text}
        """

    def tribe_string(self, language) -> str:
        text = self.text_formatter.export_tribe_only(self.tribe, self.leaders, self.foreign_tribes, language)
        return text

    def relationship_string(self, language) -> str:
        text = self.text_formatter.export_relationships_only(self.leaders, self.foreign_tribes, language)
        return text

    @classmethod
    def initialize(cls, tribe: TribeType, leader: StartCharacter) -> 'GameState':
        leader_character = Character(name=leader.name, title=leader.title, relationships=[])
        return cls(
            tribe=tribe,
            leaders=[leader_character],
            foreign_tribes=[],
            situation="",
            event_result="",
            hidden_game_state=HiddenGameState()
        )

    def update(self, new_state: GameStateBase):
        for field in GameStateBase.model_fields:
            setattr(self, field, getattr(new_state, field))

    def update_streak(self, outcome: OutcomeType) -> None:
        """Updates the streak counter based on the outcome."""
        if outcome == "positive":
            self.streak_count += 1
        else:
            self.streak_count = 0

    def adjust_power(self, outcome: OutcomeType) -> None:
        """Adjusts power level based on the outcome type."""
        power_adjustments = {
            "positive": 2,
            "neutral": 0,
            "negative": -1,
            "catastrophe": -3
        }
        self.power += power_adjustments[outcome]

    def update_tier(self) -> None:
        """Updates the tier based on the current power level."""
        self.tier = max(min((self.power-5) // 10, 3) + 1, 1)

    def check_streak_catastrophe(self) -> bool:
        """Checks if streak has reached catastrophe level."""
        if self.streak_count >= 4:
            self.streak_count = 0
            return True
        return False


class GameStateManager:
    def __init__(self, llm_context: LLMContext, language: Language = "english"):
        self.current_game_state: Optional[GameState] = None
        self.game_history = GameHistory()
        self.llm_context = llm_context
        self.current_tribe_choices = None
        self.language = language

    def reset(self):
        self.current_game_state = None
        self.game_history = GameHistory()

    def initialize(self, tribe: TribeType, leader: StartCharacter):
        self.current_game_state = GameState.initialize(tribe, leader)
        self.current_game_state.short_history = ""
        self.game_history = GameHistory()
        self.current_game_state.previous_situation = "Humble beginnings of the " + self.current_game_state.tribe.name
        self.current_game_state.last_outcome = "neutral"
        self.update_and_generate_choices("single_event")
        self.game_history.add_state(self.current_game_state)
        base_filename = "Debug_History"
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Create unique filename
        filename = f"{base_filename}_turn{self.current_game_state.turn}_{timestamp}.json"
        self.save_all_game_states(filename)
        self.current_game_state.turn += 1

    def perform_action(self, index: int):
        action = self.current_game_state.current_action_choices.choices[index]
        self.current_game_state.chosen_action = action
        self.current_game_state.previous_situation = self.current_game_state.situation
        outcome, roll = self.determine_outcome(action.probability)
        print(f"Debug: Roll = {roll:.2f}, Probability = {action.probability:.2f}, Outcome = {outcome}")
        # Handle streak catastrophe
        if self.current_game_state.check_streak_catastrophe():
            outcome = "catastrophe"

        # Update game state
        self.current_game_state.update_streak(outcome)
        self.current_game_state.adjust_power(outcome)
        self.current_game_state.update_tier()
        self.current_game_state.last_outcome = outcome
        event_type = self.decide_event_type()
        print(f"Event type: {event_type}")
        self.update_and_generate_choices(event_type)
        self.game_history.add_state(self.current_game_state)
        base_filename = "Debug_History"
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Create unique filename
        filename = f"{base_filename}_turn{self.current_game_state.turn}_{timestamp}.json"
        self.save_all_game_states(filename)
        self.current_game_state.turn += 1

    def save_all_game_states(self, filename: str):
        filename  = "save//" + filename
        if not os.path.exists("save"):
            os.makedirs("save")
        with open(filename, 'w') as f:
            json.dump(self.game_history.history, f, cls=EnumEncoder)

        # Also write the story version
        filename = filename.replace('.json', '_story.txt')
        story = []
        for state in self.game_history.history:
            story.append(f"=== Turn {state.turn} ===")
            story.append(f"Situation: {state.previous_situation}")
            if state.chosen_action:
                story.append(f"Taken action: {state.chosen_action.description}")
            outcome = state.last_outcome
            story.append(f"The result was: {outcome} and resulted in: {state.event_result}\n")

        with open(filename, 'w') as f:
            f.write("\n".join(story))

    def load_game_state(self, filename: str):
        self.game_history = GameHistory()
        with open(filename, 'r') as f:
            loaded_history = json.load(f)
        # Convert each history entry to a GameState object
        history: List[GameState] = [GameState(**state) for state in loaded_history]
        # Set the current game state to the last item in the history
        self.current_game_state = copy.deepcopy(history[-1])

        # Add all states to the history
        for state in history:
            self.game_history.add_state(state)
        self.current_game_state.turn += 1

    def generate_tribe_choices(self, external_choice: str = "") -> TribesResponse:
        # Generate 3 unique combinations of development type and diplomatic stance

        possible_combinations = [
            (dev, stance)
            for dev in DEVELOPMENT_TYPES
            for stance in DIPLOMATIC_STANCES
        ]
        selected_combinations = random.sample(possible_combinations, 3)

        # Get orientation descriptions for each combination
        orientations = [
            get_tribe_orientation(dev_type, stance, self.language)
            for dev_type, stance in selected_combinations
        ]
        if external_choice != "":
            tribe1 = f"- Must be {external_choice}. Mention the race in the description."
        else:
            tribe1 = ""

        # Construct the prompt with the specific requirements for each tribe
        messages = [
            {
                "role": "system",
                "content": "You are the game master for a text based strategic game, where the player rules over a tribe in a fantasy based setting. Output your answers as JSON"
            },
            {
                "role": "user",
                "content": f"""Create exactly three tribes as JSON:

ALL fields must be included for each tribe. 

1. First tribe MUST have:
   - tribe_development: {selected_combinations[0][0]}
   - tribe_stance: {selected_combinations[0][1]}
   - Theme: {orientations[0]}
   {tribe1}

2. Second tribe MUST have:
   - tribe_development: {selected_combinations[1][0]}
   - tribe_stance: {selected_combinations[1][1]}
   - Theme: {orientations[1]}

3. Third tribe MUST have:
   - tribe_development: {selected_combinations[2][0]}
   - tribe_stance: {selected_combinations[2][1]}
   - Theme: {orientations[2]}

Each tribe must have a unique name and detailed description. The description should mention the race but not the tribe name.
ALL four fields are required for each tribe."""
            }
        ]

        tribes = self.llm_context.make_story_call(messages, TribesResponse, max_tokens=2000)

        if tribes:
            self.current_tribe_choices = tribes
            print("\n=== Available Tribe Choices ===")
            for number, (choice, (dev_type, stance)) in enumerate(zip(tribes.tribes, selected_combinations)):
                tribe_type = get_tribe_orientation(dev_type, stance, self.language)
                print(f"\n{number}. {choice.name}, {tribe_type}")
                print(textwrap.fill(f"Description: {choice.description}", width=80, initial_indent="   ",
                                    subsequent_indent="   "))

        return tribes

    def get_leader(self, chosen_tribe: TribeType) -> StartCharacter:
        messages = [
            {
                "role": "system",
                "content": "You are the game master for a text based strategic game, where the player rules over a tribe in a fantasy based setting. Output your answers as JSON"
            },
            {
                "role": "user",
                "content": f"""Give me the name and title for the leader of {chosen_tribe.name}. Their description is:
{chosen_tribe.description}. """
            }
        ]

        leader = self.llm_context.make_story_call(messages, StartCharacter, max_tokens=2000)

        if leader:
            print(textwrap.fill(f"\nLeader: {leader}", width=80, initial_indent="   ",
                                subsequent_indent="   "))

        return leader

    def get_probability_adjustment(self) -> str:
        if self.current_game_state.last_outcome not in ["positive"]:
            return "Due to the recent neutral or negative outcome, provide higher probabilities of success for the next choices (between 0.6 and 0.9)."
        return "Provide balanced probabilities of success for the next choices (between 0.5 and 0.8)."

    def get_choice_type_instructions(self, choice_type: EventType) -> dict:
        # Base instructions for all types
        base_instructions = {
            "single_event": {
                "instruction": "create three distinct choices that reflect the tribe's nature",
                "extra": ""
            },
            "reaction": {
                "instruction": "create a reaction to the last outcome",
                "extra": "Include a foreign character in the reaction."
            },
            "followup": {
                "instruction": "create a follow up to the last outcome",
                "extra": "Keep choices thematically close to the last action."
            }
        }

        # Only add specific extra instructions for single_event type
        if choice_type == "single_event":
            # Get development-specific instructions
            dev_instructions = {
                "magical": """
                    - Include one ritual or mystical option
                    - Consider supernatural consequences
                    - Add choices involving magical resources""",
                "practical": """
                    - Focus on infrastructure and resource management
                    - Include diplomatic or trade options
                    - Add choices about technological advancement""",
                "hybrid": """
                    - Balance magical and practical approaches
                    - Include choices that combine both aspects
                    - Consider the interaction between magic and technology"""
            }

            # Get stance-specific instructions
            stance_instructions = {
                "peaceful": """
                    - Prioritize diplomatic solutions
                    - Focus on defensive and growth options
                    - Avoid direct confrontation choices""",
                "neutral": """
                    - Balance aggressive and peaceful options
                    - Include pragmatic choices
                    - Add opportunities for strategic positioning""",
                "aggressive": """
                    - Include direct confrontation options
                    - Focus on expansion or dominance
                    - Add choices that challenge rivals"""
            }

            # Only add relevant instructions based on tribe's development and stance
            development = self.current_game_state.tribe.development
            stance = self.current_game_state.tribe.stance

            extra = ""
            if development in dev_instructions:
                extra += dev_instructions[development]
            if stance in stance_instructions:
                extra += "\n" + stance_instructions[stance]

            base_instructions["single_event"]["extra"] = extra

        return base_instructions[choice_type]

    def update_and_generate_choices(self, choice_type: EventType) -> None:
        # 1. Update history
        recent_history = self.game_history.get_recent_short_history(num_events=1)
        if recent_history != "":
            self.current_game_state.short_history = self.llm_context.get_summary(
                self.current_game_state.short_history + recent_history)

        # 2. Get probability adjustment
        prob_adjustment = self.get_probability_adjustment()

        # 3. Build tribes prompt
        tribes_prompt = ""
        enemy_count = sum(1 for tribe in self.current_game_state.foreign_tribes
                          if tribe.diplomatic_status == "enemy")
        neutral_count = sum(1 for tribe in self.current_game_state.foreign_tribes
                            if tribe.diplomatic_status == "neutral")

        if self.current_game_state.turn > 5 and enemy_count < 1:
            tribes_prompt += "* Add one enemy faction if it fits the current situation\n"
        if self.current_game_state.turn > 3 and neutral_count < 2:
            tribes_prompt += "* Add one or two neutral factions if it fits the current situation\n"

        # 4. Build action context
        action_context = (
            f"""Action: {self.current_game_state.chosen_action.caption}

            {self.current_game_state.chosen_action.description}

            Outcome: {self.current_game_state.last_outcome}"""
            if self.current_game_state.chosen_action
            else
            f"""This is the beginning of the {self.current_game_state.tribe.name}. 
        Treat the last action and the outcome as neutral, and tell something about their background."""
        )

        # 5. Define choice types and get instructions
        event_prompt = generate_game_prompt(
            event_type=choice_type,
            tier=self.current_game_state.tier,
            development=self.current_game_state.tribe.development,
            stance=self.current_game_state.tribe.stance
        )

        # 6. Create messages with JSON structure first
        messages = [
            {
                "role": "system",
                "content": """You are a game master for a tribal fantasy strategy game managing both state updates and story progression."""
            },
            {
                "role": "user",
                "content": f"""Current game state: 
    {self.current_game_state.to_context_string(self.language)}

    History: {self.current_game_state.short_history}
    Previous: {self.current_game_state.previous_situation}

    {action_context}

    Update the game state following these rules:

    1. Tribe: Only change the tribe name and its stances after major events

    2. Leaders (max 5):
       - Update names, titles and relationships
       - Add special roles if warranted
       - Big changes only after important events

    3. Foreign Tribes:
       - For each tribe: status, name, description, development, stance, leaders (max 2)
       - For each leader: Names, titles, relationships
       - DiplomaticStatus changes require multiple turns
       - Consider development/stance compatibility
       {tribes_prompt}

    4. Event Results:
       - 2 paragraphs narrative based on the outcome
       - Events may continue or conclude with closure

    5. Situation:
       - Current state summary based on outcome
       - Add new developments as needed
       
   6. Hidden game state:
       - Track potential future developments, secrets, and hidden relationships
       - This information must be revealed through either:
         a) Event Results: Weave revelations naturally into the narrative
         b) Situation Updates: Include discoveries as new developments
       - Rules for using hidden information:
       - Cannot directly reference hidden information in choices
       - Must first introduce/reveal information through narrative
       - Revelations should feel natural and properly timed
       - Can partially reveal information to build suspense
    
    7. New choices:
{event_prompt}

{prob_adjustment}"""
            }
        ]


        # Define a combined response class that includes both state and choices
        if combined_response := self.llm_context.make_story_call(messages, CombinedStateAndChoices, max_tokens=7000):
            # Update game state using dict unpacking
            #state_fields = {
            #    field: getattr(combined_response, field)
            #    for field in ['tribe', 'leaders', 'foreign_tribes', 'situation', 'event_result', 'hidden_game_state']
            #}
            #self.current_game_state.update(GameStateBase(**state_fields))
            new_state = GameStateBase.model_validate(combined_response)
            self.current_game_state.update(new_state)

            # Update choices
            self.current_game_state.current_action_choices = NextChoices(choices=combined_response.choices)

            # Print available actions
            print("\n=== Available Actions ===")
            for i, choice in enumerate(combined_response.choices, 1):
                print(f"\n{i}. {choice.caption}")
                print(textwrap.fill(f"Description: {choice.description}",
                                    width=80,
                                    initial_indent="   ",
                                    subsequent_indent="   "))
                print(f"   Probability of Success: {choice.probability:.2f}")

    @staticmethod
    def determine_outcome(probability: float) -> Tuple[OutcomeType, float]:
        roll = random.randint(0, 100) / 100.0
        # roll = 0.0
        # roll = 1.0
        if roll >= 0.95:
            return "catastrophe", roll
        if roll <= probability:
            return "positive", roll
        elif roll < (probability + (1 - probability) / 2):
            return "neutral", roll
        else:
            return "negative", roll

    @staticmethod
    def decide_event_type() -> EventType:
        return random.choices(
            ["single_event", "reaction", "followup"],
            weights=[0.2, 0.4, 0.4], k=1)[0]


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

        tribe_overview = game_manager.current_game_state.tribe_string(game_manager.language)
        relationships = game_manager.current_game_state.relationship_string(game_manager.language)
        recent_history = game_manager.game_history.get_recent_history(game_manager.language)
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


    def load_saved_game(filename):
        try:
            game_manager.load_game_state(filename)
            current_action_choices = game_manager.current_game_state.current_action_choices

            # Update all necessary GUI elements
            tribe_overview = game_manager.current_game_state.tribe_string(game_manager.language)
            relationships = game_manager.current_game_state.relationship_string(game_manager.language)
            recent_history = game_manager.game_history.get_recent_history(game_manager.language)
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
            for number, tribe in enumerate(game_manager.current_tribe_choices.tribes):
                tribe_type = get_tribe_orientation(tribe.development, tribe.stance, game_manager.language)
                result += f"{number + 1}. {tribe.name}, "
                result += f"{tribe_type}\n"
                result += f"{tribe.description}\n\n"

            return result, gr.update(visible=True, choices=[1, 2, 3])
        else:
            return "Error generating initial choices. Please try again.", gr.update(visible=False)

    def start_game(choice):
        game_manager.reset()
        if game_manager.current_tribe_choices:
            chosen_tribe = next(
                (tribe for index, tribe in enumerate(game_manager.current_tribe_choices.tribes, 1) if index == choice),
                None)
            if chosen_tribe:
                leader = game_manager.get_leader(chosen_tribe)
                game_manager.initialize(chosen_tribe, leader)

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
        title_markdown = gr.Markdown(f"# {translations['english']['title']}")

        # Game Tab
        game_tab = gr.Tab(translations['english']["game_tab"])
        with game_tab:
            # Create tribe selection group
            with gr.Group() as tribe_selection_group:
                race_theme = gr.Dropdown(
                    choices=translations['english']["race_themes"],
                    value="",
                    label=translations['english']["race_theme_label"]
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
                load_file = gr.File(label="Select Save File", file_types=[".json"])
                load_button = gr.Button("Load Game")

            message_display = gr.Textbox(label="System Messages", lines=2)

        # Add Settings Tab
        with gr.Tab("Settings"):
            with gr.Group():
                gr.Markdown("## Language Configuration")
                language_dropdown = gr.Dropdown(
                    choices=SUPPORTED_LANGUAGES,
                    value=config.language,
                    label="Language"
                )
                gr.Markdown("## Story LLM Configuration")
                story_provider_dropdown = gr.Dropdown(
                    choices=[provider.value for provider in LLMProvider],
                    value=config.story_config.provider.value,
                    label="Story LLM Provider"
                )
                story_model_dropdown = gr.Dropdown(
                    choices=["claude-3-5-sonnet-20241022", "claude-3-5-sonnet-20240620", "claude-3-5-haiku-20241022"],
                    value="claude-3-5-sonnet-20241022",
                    label="Story Model (Anthropic)",
                    visible=config.story_config.provider == LLMProvider.ANTHROPIC
                )
                story_model_openai_dropdown = gr.Dropdown(
                    choices=["gpt-4o", "gpt-4o-mini"],
                    value="gpt-4o",
                    label="Story Model (OpenAI)",
                    visible=config.story_config.provider == LLMProvider.OPENAI
                )
                story_model_gemini_dropdown = gr.Dropdown(
                    choices=["models/gemini-1.5-pro", "models/gemini-1.5-flash-latest"],
                    value="models/gemini-1.5-pro",
                    label="Story Model (Gemini)",
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
                    choices=["claude-3-5-sonnet-20241022", "claude-3-5-sonnet-20240620", "claude-3-5-haiku-20241022"],
                    value="claude-3-5-sonnet-20241022",
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
                    story_model_gemini_dropdown: gr.update(visible=provider == LLMProvider.GEMINI.value),
                    story_model_local_input: gr.update(visible=provider == LLMProvider.LOCAL.value),
                    story_local_url_input: gr.update(visible=provider == LLMProvider.LOCAL.value)
                }

            story_provider_dropdown.change(
                update_story_input_visibility,
                inputs=[story_provider_dropdown],
                outputs=[story_model_dropdown, story_model_openai_dropdown, story_model_gemini_dropdown,
                         story_model_local_input, story_local_url_input]
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
                outputs=[summary_model_dropdown, summary_model_openai_dropdown,
                         summary_model_local_input, summary_local_url_input]
            )

            def update_ui_language(language):
                lang = translations.get(language, translations["english"])  # Default to English if language not found

                return {
                    'title_markdown': gr.update(value=f"# {lang['title']}"),
                    'game_tab': gr.update(label=lang['game_tab']),
                    'race_theme': gr.update(choices=lang['race_themes'], label=lang['race_theme_label']),
                    # Add other UI elements that need updating here
                }

            # Function to save settings
            def save_settings(language, story_provider, story_model_anthropic, story_model_openai, story_model_gemini, story_model_local,
                              story_local_url, summary_provider, summary_model_anthropic, summary_model_openai,
                              summary_model_local, summary_local_url, summary_mode):
                story_model = story_model_anthropic if story_provider == LLMProvider.ANTHROPIC.value else \
                    story_model_openai if story_provider == LLMProvider.OPENAI.value else \
                    story_model_gemini if story_provider == LLMProvider.GEMINI.value else \
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
                    ),
                    language=language
                )
                llm_context.update(new_config)
                game_manager.language = language
                # Add the settings message to the UI updates
                ui_updates = update_ui_language(language)

                return (
                    ui_updates.get('title_markdown', gr.update()),
                    ui_updates.get('game_tab', gr.update()),
                    ui_updates.get('race_theme', gr.update()),
                    gr.update(value="Settings saved successfully!")
                )

            settings_save_btn.click(
                save_settings,
                inputs=[language_dropdown, story_provider_dropdown, story_model_dropdown,
                        story_model_openai_dropdown, story_model_gemini_dropdown, story_model_local_input, story_local_url_input,
                        summary_provider_dropdown, summary_model_dropdown, summary_model_openai_dropdown,
                        summary_model_local_input, summary_local_url_input, summary_mode_dropdown],
                outputs=[title_markdown, game_tab, race_theme, settings_message]
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
    app = create_gui()
    app.launch(server_name="0.0.0.0")
