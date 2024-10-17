# Fantasy Tribe Game

Play a fantasy tribe from humble beginnings to earth-shattering godlike power, powered by Large Language Models (LLMs).

## Description

Fantasy Tribe Game is an interactive text-based strategy game where you lead your chosen tribe through various challenges and quests. The game leverages LLMs to create dynamic storytelling, generating unique tribes, leaders, quests, and events that respond to your choices.

![Fantasy](https://github.com/user-attachments/assets/3425ab7c-73f3-49b1-b35c-f46a43e0604e)

### Key Features

- Choose from three unique, procedurally generated tribes
- Make strategic decisions that impact your tribe's development
- Dynamic quest system with multiple outcomes
- Resource management (Gold, Territory, Power Level)
- Relationship system (Allies and Enemies)
- Tier progression system
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

### LLM Provider Configuration

The game supports three LLM providers:

1. **Local Model**
   - Tested with LM Studio
   - Configure the local URL in settings
   - Default: `http://127.0.0.1:1234/v1`

2. **OpenAI**
   - Requires OpenAI API key
   - Supports models: gpt-4o-mini, gpt-4o

3. **Anthropic**
   - Requires Anthropic API key

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
   - Manage your tribe's resources
   - Complete quests
   - Handle random events
   - Make strategic decisions
   - Progress through different tiers of power

### Game State Elements
- Tribe name and description
- Leader information
- Resources (Gold, Territory Size, Power Level)
- Diplomatic relations (Allies, Enemies)
- Tier level
- Quest progress

### Interface
- Tribe Overview: Displays current tribe status and resources
- Recent History: Shows the outcomes of your decisions
- Current Situation: Describes the active quest or event
- Action Choices: Up to three possible actions for each situation

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
