# Fantasy Tribe Game

Play a fantasy tribe from humble beginnings to earth-shattering godlike power, powered by Large Language Models (LLMs).

## Description

Fantasy Tribe Game is an interactive text-based strategy game where you lead your chosen tribe through various challenges and quests. The game leverages LLMs to create dynamic storytelling, generating unique tribes, leaders, quests, and events that respond to your choices.

![{0AEB88EF-DA1C-47F0-BA2E-2DB8D8B92B95}](https://github.com/user-attachments/assets/ba99a51b-db80-46e3-9db3-d71569e52049)


### Key Features

- Choose from three unique, procedurally generated tribes
- seed the tribe generation using one of 74 mystical races
- Make strategic decisions that impact your tribe's development
- Dynamic quest system with multiple outcomes
- Relationship system (Allies, Neutrals and Enemies)
- Tier progression system (not working right now)
- Persistent game state with save/load functionality

## Installation

### Prerequisites

- Python 3.x

### Dependencies

Install the required packages using pip:

```
pip install -r requirements.txt
```

- pydantic
- openai
- gradio
- anthropic (optional)
- google.generativeai (optional)

### LLM Provider Configuration

The game supports four LLM providers:

1. **Local Model**
   - Tested with LM Studio
   - Configure the local URL in settings
   - Default: `http://127.0.0.1:1234/v1`

2. **OpenAI**
   - Requires OpenAI API key
   - Supports models: gpt-4o-mini, gpt-4o

3. **Anthropic**
   - Requires Anthropic API key

3. **Gemini**
   - Requires Google API key

For the story LLM, a structured output is needed. For the summary LLM, which compresses the last turns internally, you can choose between json mode, requiring structured output, and raw mode, using normal text mode.

## Getting Started

1. Run the game:
   ```
   python Fantasy-Tribe-Game.py
   ```

2. Open the page in your webbrowser, standard is http://127.0.0.1:7860
3. Configure your preferred LLM provider in the Settings tab
4. Start a new game or load a saved one
   
## Gameplay

### Game Flow
1. **Tribe Selection**
   - Choose from three procedurally generated tribes
   - Each tribe comes with unique characteristics and a leader

2. **Main Game Loop**
   - Handle random events
   - Complete quests
   - Make strategic decisions
   - (Progress through different tiers of power)

### Game State Elements
- Tribe name and description
- Leader information
- Relationship of named Characters
- Diplomatic relations (Allies, Neutrals, Enemies)

### Interface
- Tribe Overview: Displays current tribe status, its leaders and other tribes
- Relationships: SHows the realtionship of the named characters
- Recent History: Shows the outcomes of your decisions
- Current Situation: Describes the curren situation
- Action Choices: Three possible actions for each situation

### Save System
- Save your progress at any time
- Load previously saved games
- Automatic history tracking (saved to History.json)

## Debug Features

- Console logging for outcome probabilities and tier progression
- Detailed game state history saved to History.json
- System message display for important events

## License

This project is licensed under the MIT License.

## Copyright

Copyright (c) 2024 HabermannR

## Repository

Source code available at: https://github.com/HabermannR/Fantasy-Tribe-Game
