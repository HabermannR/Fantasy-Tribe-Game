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
from Language import Language, translations
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


DIPLOMATIC_STATUS_TRANSLATIONS = {
    Language.ENGLISH: {
        DiplomaticStatus.ALLY: "Ally",
        DiplomaticStatus.NEUTRAL: "Neutral",
        DiplomaticStatus.ENEMY: "Enemy",
    },
    Language.GERMAN: {
        DiplomaticStatus.ALLY: "Verbündeter",
        DiplomaticStatus.NEUTRAL: "Neutral",
        DiplomaticStatus.ENEMY: "Feind",
    },
}


def get_tribe_orientation(development: DevelopmentType, stance: DiplomaticStance, language: Language) -> str:
    """
    Get the orientation-specific description for a tribe based on their development path and diplomatic stance.

    Args:
        development (Development): The tribe's development focus (magical, hybrid, or practical)
        stance (DiplomaticStance): The tribe's diplomatic stance
        language (Language): The language of the UI
    """
    if development == DevelopmentType.MAGICAL:
        if stance == DiplomaticStance.PEACEFUL:
            if language == Language.ENGLISH:
                return "Mystic sages who commune with the very essence of nature"
            if language == Language.GERMAN:
                return "Mystische Weise, die mit dem Wesen der Natur in Verbindung stehen"
        elif stance == DiplomaticStance.NEUTRAL:
            if language == Language.ENGLISH:
                return "Arcane scholars who balance the powers of magic and wisdom"
            if language == Language.GERMAN:
                return "Arkane Gelehrte, die die Kräfte der Magie und Weisheit im Gleichgewicht halten"
        else:  # AGGRESSIVE
            if language == Language.ENGLISH:
                return "Battle-mages who wield the raw forces of destructive magic"
            if language == Language.GERMAN:
                return "Kampfmagier, die die rohen Kräfte zerstörerischer Magie beherrschen"
    elif development == DevelopmentType.HYBRID:
        if stance == DiplomaticStance.PEACEFUL:
            if language == Language.ENGLISH:
                return "Technomancers who weave together the threads of science and spirit"
            if language == Language.GERMAN:
                return "Technomanten, die die Fäden von Wissenschaft und Geist verweben"
        elif stance == DiplomaticStance.NEUTRAL:
            if language == Language.ENGLISH:
                return "Arcane engineers who forge marvels of magic and machinery"
            if language == Language.GERMAN:
                return "Arkane Ingenieure, die Wunderwerke aus Magie und Maschinerie erschaffen"
        else:  # AGGRESSIVE
            if language == Language.ENGLISH:
                return "Magitech warriors who harness both arcane energy and steel"
            if language == Language.GERMAN:
                return "Magitech-Krieger, die sowohl arkane Energie als auch Stahl beherrschen"
    else:  # PRACTICAL
        if stance == DiplomaticStance.PEACEFUL:
            if language == Language.ENGLISH:
                return "Master builders dedicated to the advancement of technology"
            if language == Language.GERMAN:
                return "Meisterbaumeister, die sich dem technologischen Fortschritt verschrieben haben"
        elif stance == DiplomaticStance.NEUTRAL:
            if language == Language.ENGLISH:
                return "Innovators who forge a path of progress and strength"
            if language == Language.GERMAN:
                return "Innovatoren, die einen Weg des Fortschritts und der Stärke beschreiten"
        else:  # AGGRESSIVE
            if language == Language.ENGLISH:
                return "Warriors at the forefront of military innovation and might"
            if language == Language.GERMAN:
                return "Krieger an der Spitze militärischer Innovation und Macht"


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
    def get_sentiment(opinion: int, language: Language) -> str:
        if opinion >= 90:
            return "devoted" if language == Language.ENGLISH else "ergeben"
        elif opinion >= 70:
            return "strongly positive" if language == Language.ENGLISH else "sehr positiv"
        elif opinion >= 40:
            return "favorable" if language == Language.ENGLISH else "günstig"
        elif opinion >= 10:
            return "somewhat positive" if language == Language.ENGLISH else "etwas positiv"
        elif opinion > -10:
            return "neutral" if language == Language.ENGLISH else "neutral"
        elif opinion >= -40:
            return "somewhat negative" if language == Language.ENGLISH else "etwas negativ"
        elif opinion >= -70:
            return "unfavorable" if language == Language.ENGLISH else "ungünstig"
        elif opinion >= -90:
            return "strongly negative" if language == Language.ENGLISH else "sehr negativ"
        else:
            return "hostile" if language == Language.ENGLISH else "feindselig"

    @staticmethod
    def format_tribe_description(tribe: TribeType, leaders: List[Character], relation: bool, language: Language) -> str:
        tribe_type = get_tribe_orientation(tribe.development, tribe.stance, language)
        text = f"""{tribe.name}
{tribe_type}

{tribe.description}

{"Leaders:" if language == Language.ENGLISH else "Anführer:"}"""

        for leader in leaders:
            text += TextFormatter._format_leader(leader, relation, language)
        return text

    @staticmethod
    def format_tribe_description_llm(tribe: TribeType, leaders: List[Character], relation: bool) -> str:
        tribe_type = get_tribe_orientation(tribe.development, tribe.stance, Language.ENGLISH)
        text = f"""Tribe name: {tribe.name}
Tribe development type: {tribe.development.value}
Tribe stance: {tribe.stance.value}
Resulting tribe type: {tribe_type}

Tribe description: {tribe.description}

{"Leaders:"}"""
        for leader in leaders:
            text += TextFormatter._format_leader(leader, relation, Language.ENGLISH)
        return text

    @staticmethod
    def _format_leader(leader: Character, relation: bool, language: Language) -> str:
        text = f"\n  • {leader.title}: {leader.name}"
        if leader.relationships and relation:
            text += "\n    " + ("Relationships:" if language == Language.ENGLISH else "Beziehungen:")
            for rel in leader.relationships:
                sentiment = TextFormatter.get_sentiment(rel.opinion, language)
                if language == Language.ENGLISH:
                    text += f"\n      - {rel.type} relationship with  {rel.target}, {sentiment}: {rel.opinion}"
                elif language == Language.GERMAN:
                    text += f"\n      - Beziehung zu {rel.target}: {rel.type}, {sentiment}: {rel.opinion}"
        return text

    @staticmethod
    def format_foreign_tribes(foreign_tribes: List[ForeignTribeType], relation: bool, language: Language) -> str:
        if not foreign_tribes:
            return ""

        text = "\n\n" + ("Foreign Tribes:" if language == Language.ENGLISH else "Fremde Stämme:")
        status_order = [DiplomaticStatus.ALLY, DiplomaticStatus.NEUTRAL, DiplomaticStatus.ENEMY]
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
            text += "\n" + ("Leaders:" if language == Language.ENGLISH else "Anführer:")
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
                    if language == Language.ENGLISH:
                        text += f"  - {rel.type} relationship with  {rel.target}, {sentiment}\n"
                    elif language == Language.GERMAN:
                        text += f"  - Beziehung zu {rel.target}: {rel.type}, {sentiment}\n"

        # Foreign tribe relationships
        for foreign_tribe in foreign_tribes:
            if foreign_tribe.leaders:
                for leader in foreign_tribe.leaders:
                    if leader.relationships:
                        text += f"\n{leader.name} ({leader.title}) {'from' if language == Language.ENGLISH else 'von'} {foreign_tribe.name}:\n"
                        for rel in leader.relationships:
                            sentiment = TextFormatter.get_sentiment(rel.opinion, language)
                            if language == Language.ENGLISH:
                                text += f"  - {rel.type} relationship with  {rel.target}, {sentiment}\n"
                            elif language == Language.GERMAN:
                                text += f"  - Beziehung zu {rel.target}: {rel.type}, {sentiment}\n"

        return text


# Class for managing game history
class GameHistory:
    def __init__(self):
        self.history: List['GameState'] = []
        self.short_history = ""

    def add_state(self, state: 'GameState'):
        self.history.append(copy.deepcopy(state))

    def get_recent_history(self, language: Language) -> str:
        formatted_events = []
        if self.history:
            for event in reversed(self.history):
                turn_header = f"=== Turn {event.turn} ===" if language == Language.ENGLISH else f"=== Runde {event.turn} ==="
                formatted_events.extend([turn_header, event.event_result])
        return "\n\n".join(formatted_events)

    def get_recent_short_history(self, num_events: int = 5) -> str:
        recent_events = self.history[-num_events:]
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


class CombinedStateAndChoices(GameStateBase):
    choices: List[ActionChoice]


class GameState(GameStateBase):
    previous_situation: Optional[str] = None
    current_action_choices: Optional['NextChoices'] = None
    chosen_action: Optional['ActionChoice'] = None
    last_outcome: Optional[OutcomeType] = None
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
        return tribe_text + foreign_tribes_text

    def tribe_string(self, language) -> str:
        text = self.text_formatter.export_tribe_only(self.tribe, self.leaders, self.foreign_tribes, language)
        return text

    def relationship_string(self, language) -> str:
        text = self.text_formatter.export_relationships_only(self.leaders, self.foreign_tribes, language)
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
        self.tier = max(min((self.power-5) // 10, 3) + 1, 1)

    def check_streak_catastrophe(self) -> bool:
        """Checks if streak has reached catastrophe level."""
        if self.streak_count >= 4:
            self.streak_count = 0
            return True
        return False


class GameStateManager:
    def __init__(self, llm_context: LLMContext, language: Language = Language.ENGLISH):
        self.current_game_state: Optional[GameState] = None
        self.game_history = GameHistory()
        self.llm_context = llm_context
        self.current_tribe_choices = None
        self.language = language

    def reset(self):
        self.current_game_state = None
        self.game_history = GameHistory()

    def initialize(self, tribe: TribeType, leader: Character):
        self.current_game_state = GameState.initialize(tribe, leader)
        self.game_history = GameHistory()
        self.current_game_state.previous_situation = "Humble beginnings of the " + self.current_game_state.tribe.name
        self.current_game_state.last_outcome = OutcomeType.NEUTRAL
        self.update_and_generate_choices(EventType.SINGLE_EVENT)
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
        print(f"Debug: Roll = {roll:.2f}, Probability = {action.probability:.2f}, Outcome = {outcome.name}")
        # Handle streak catastrophe
        if self.current_game_state.check_streak_catastrophe():
            outcome = OutcomeType.CATASTROPHE

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

    def generate_tribe_choices(self, external_choice: str = "") -> InitialChoices:
        # Generate 3 unique combinations of development type and diplomatic stance

        possible_combinations = [
            (dev, stance)
            for dev in DevelopmentType
            for stance in DiplomaticStance
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
                tribe_type = get_tribe_orientation(dev_type, stance, self.language)
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

    def update_and_generate_choices(self, choice_type: EventType) -> None:
        recent_history = self.game_history.get_recent_short_history(num_events=1)
        if recent_history != "":
            self.game_history.short_history = self.llm_context.get_summary(
                self.game_history.short_history + recent_history)
        prob_adjustment = self.get_probability_adjustment()

        # Build tribes prompt for potential new factions
        tribes_prompt = ""
        enemy_count = sum(
            1 for tribe in self.current_game_state.foreign_tribes if tribe.diplomatic_status == DiplomaticStatus.ENEMY)
        neutral_count = sum(
            1 for tribe in self.current_game_state.foreign_tribes if
            tribe.diplomatic_status == DiplomaticStatus.NEUTRAL)

        if self.current_game_state.turn > 5 and enemy_count < 1:
            tribes_prompt += "* Add one enemy faction if it fits the current situation\n"
        if self.current_game_state.turn > 3 and neutral_count < 2:
            tribes_prompt += "* Add one or two neutral factions if it fits the current situation\n"

        # Build action context if there's a chosen action
        action_context = ""
        if self.current_game_state.chosen_action:
            action_context = f"""Action: {self.current_game_state.chosen_action.caption}
        
    {self.current_game_state.chosen_action.description}

    Outcome: {self.current_game_state.last_outcome.name}"""
        else:
            action_context = f"""This is the beginning of the {self.current_game_state.tribe.name}. 
Treat the last action and the outcome as neutral, and tell something about their background."""

        # Define choice type specific instructions
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

        # Build the combined context
        context = f"""Gamestate: 
{self.current_game_state.to_context_string(self.language)}

    History: {self.game_history.short_history}
    Previous: {self.current_game_state.previous_situation}

    {action_context}"""

        # Combined instructions for both state update and choice generation
        instructions = f"""
    First, update the game state:

    1. Tribe: Only change the tribe name and its stances after major events
    
    2. Leaders (max 5):
       - Update names, titles and relationships, both to tribe leaders as well as to leaders of foreign tribes
       - Add special roles if warranted
       - Big changes only after important events
       
   3. Foreign Tribes:
       - For each tribe: status, name, description, development, stance, leaders (max 2)
       - For each leader of a foreign tribe: Names, titles, relationships to the leaders of the players tribe
       - DiplomaticStatus changes require multiple turns from Neutral to Ally
       - Consider development/stance compatibility
       {tribes_prompt}

    4. Event Results:
       - 2 paragraphs narrative (event_result) based on the outcome of the last event
       - Events may continue, or you can decide to finish an event and introduce a new one. When finishing, give some closure here. 
       
   5. Situation:
       - Current state summary based on the outcome of the last event
       - You can add new developments to make it more interesting, or to introduce new tribes

    Then, {choice_type_fields['instruction']} and provide 3 possible next choices:
    {choice_type_fields['extra']}
    Each choice has:
    - A caption
    - Description that weaves implications and broader context organically
    - Probability of success
    {prob_adjustment}
    One choice should have high probability
    Include at least one aggressive option (probability based on tribe info)

    Maintain consistency with game state, action outcomes, and existing relationships.
    Output the response as JSON with both state updates and next choices."""

        messages = [
            {
                "role": "system",
                "content": "You are a game master for a tribal fantasy strategy game managing both state updates and story progression. Output answers as JSON."
            },
            {
                "role": "user",
                # "content": summary + instructions
                "content": context + instructions
            }
        ]

        # Define a combined response class that includes both state and choices
        combined_response = self.llm_context.make_story_call(messages, CombinedStateAndChoices, max_tokens=7000)

        if combined_response:
            # Update game state
            gamestate = GameStateBase(tribe=combined_response.tribe, leaders=combined_response.leaders,
                                      foreign_tribes=combined_response.foreign_tribes,
                                      situation=combined_response.situation,
                                      event_result=combined_response.event_result)
            self.current_game_state.update(gamestate)

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

    def save_current_game():
        game_manager.save_all_game_states('test.json')
        return "Game saved successfully!"

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
            for number, choice in enumerate(game_manager.current_tribe_choices.choices):
                tribe_type = get_tribe_orientation(choice.development, choice.stance, game_manager.language)
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
        title_markdown = gr.Markdown(f"# {translations['en']['title']}")

        # Game Tab
        game_tab = gr.Tab(translations['en']["game_tab"])
        with game_tab:
            # Create tribe selection group
            with gr.Group() as tribe_selection_group:
                race_theme = gr.Dropdown(
                    choices=translations['en']["race_themes"],
                    value="",
                    label=translations['en']["race_theme_label"]
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
                gr.Markdown("## Language Configuration")
                language_dropdown = gr.Dropdown(
                    choices=[lang.value for lang in Language],
                    value=config.language.value,
                    label="Language"
                )
                gr.Markdown("## Story LLM Configuration")
                story_provider_dropdown = gr.Dropdown(
                    choices=[provider.value for provider in LLMProvider],
                    value=config.story_config.provider.value,
                    label="Story LLM Provider"
                )
                story_model_dropdown = gr.Dropdown(
                    choices=["claude-3-5-sonnet-20241022", "claude-3-5-sonnet-20240620"],
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
                    choices=["claude-3-5-sonnet-20241022", "claude-3-5-sonnet-20240620"],
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

            def update_ui_language(language):
                lang = translations.get(language, translations["en"])  # Default to English if language not found

                return {
                    'title_markdown': gr.update(value=f"# {lang['title']}"),
                    'game_tab': gr.update(label=lang['game_tab']),
                    'race_theme': gr.update(choices=lang['race_themes'], label=lang['race_theme_label']),
                    # Add other UI elements that need updating here
                }

            # Function to save settings
            def save_settings(language, story_provider, story_model_anthropic, story_model_openai, story_model_local,
                              story_local_url, summary_provider, summary_model_anthropic, summary_model_openai,
                              summary_model_local, summary_local_url, summary_mode):
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
                    ),
                    language=Language(language)
                )
                llm_context.update(new_config)
                game_manager.language = Language(language)
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
                        story_model_openai_dropdown, story_model_local_input, story_local_url_input,
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
    app = create_gui()
    app.launch()
