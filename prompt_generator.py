from typing import Literal

EventType = Literal["single_event", "reaction", "followup"]
DevelopmentType = Literal["magical", "hybrid", "practical"]
DiplomaticStance = Literal["peaceful", "neutral", "aggressive"]


def generate_game_prompt(
        event_type: EventType,
        tier: int,
        development: DevelopmentType,
        stance: DiplomaticStance
) -> str:
    """Generate focused prompt for LLM game event generation."""

    # Base event type instructions
    event_instructions = {
        "single_event": """
- Create three distinct choices that reflect the tribe's nature
- Ensure each choice has clear consequences
- Include options that use current capabilities""",
        "reaction": """
- Create a response to the last outcome by creating three distinct choices
- Include a foreign character in the reaction
- Show how prior choices affect this situation""",
        "followup": """
- Build directly on the last decision by creating three distinct choices
- Keep choices thematically connected
- Show how the situation has evolved"""
    }

    # Core stance instructions that apply to all development types
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

    # Development-specific guidelines
    dev_guidelines = {
        "magical": {
            "peaceful": """
- Include rituals that promote harmony
- Add choices about magical cooperation
- Consider spirit or elemental reactions""",
            "neutral": """
- Balance protective and power-seeking magic
- Include diplomatic magical solutions
- Consider magical resource management""",
            "aggressive": """
- Include combat magic options
- Add magical territory control choices
- Consider power accumulation costs"""
        },
        "practical": {
            "peaceful": """
- Include infrastructure improvements
- Add diplomatic or trade options
- Consider resource sustainability""",
            "neutral": """
- Balance defense and development
- Include strategic positioning choices
- Consider resource allocation""",
            "aggressive": """
- Include military advancement options
- Add strategic conquest choices
- Consider resource acquisition"""
        },
        "hybrid": {
            "peaceful": """
- Combine magic and technology constructively
- Include enhancement of existing systems
- Consider balance between forces""",
            "neutral": """
- Mix magical and practical solutions
- Include hybrid defense options
- Consider technical-magical synergy""",
            "aggressive": """
- Include magitech warfare options
- Add hybrid power projection choices
- Consider combined resource costs"""
        }
    }

    # Tier-specific focus
    tier_focus = {
        1: """
- Focus on immediate survival needs
- Include basic resource gathering
- Consider local threat response""",
        2: """
- Focus on regional expansion
- Include power consolidation
- Consider neighboring relations""",
        3: """
- Focus on realm-wide influence
- Include major infrastructure/systems
- Consider far-reaching consequences""",
        4: """
- Focus on world-altering projects
- Include reality-shaping options
- Consider cosmic implications"""
    }

    prompt = f"""Generate a {event_type} for a {development} tribe with {stance} stance at tier {tier}.

Event Requirements:
{event_instructions[event_type]}

Core Stance Focus:
{stance_instructions[stance]}

Development Path Focus:
{dev_guidelines[development][stance]}

Tier {tier} Considerations:
{tier_focus[tier]}"""

    return prompt.strip()


# Example usage:
if __name__ == "__main__":
    prompt = generate_game_prompt(
        event_type="single_event",
        tier=2,
        development="magical",
        stance="peaceful"
    )
    print(prompt)