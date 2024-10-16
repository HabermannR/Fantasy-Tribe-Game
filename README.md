# Fantasy Tribe Game
Play a fantasy tribe from humble beginnings to earth shattering godlike power thanks to LLMs

## Description
This game allows you to play a fantasy tribe from humble beginnings to earth shattering godlike power thanks to LLMs. You can choose from three different tribes and make decisions to shape the fate of your tribe. The game uses LLMs to generate the game world, NPCs, and storylines, providing a unique and immersive experience.

## Installation
To play the game, you need to have Python installed on your computer. You also need to install the following pip modules:

- pydantic
- openai
- gradio
- anthropic (optional)

You need to set up your LLM provider by choosing one of the following options:

- Local model: Set up a local model by setting the local_url variable to the URL of your local model. Only LM Studio is tested.
- OpenAI model: Set up an OpenAI model by setting the openAIKey variable to your API key.
- Anthropic model: Set up an Anthropic model by setting the anthropicKey variable to your API key.

Once you have set up your LLM provider, you can run the game by executing the Fantasy-Tribe-Game.py file.

## Gameplay
The game is divided into two phases:

- Tribe selection: You choose one of the three tribes to play as.
- Game play: You make decisions to shape the fate of your tribe. The game uses LLMs to generate the game world, leaders, and storylines. The result of your last action is displayed as last entry under recent history. Maybe you need to scroll down.


Saving and loading: You can save and load your game state at any time.

## Debug Mode
The game has a debug prints that prints additional information to the console and writes a history.json file. 

## License
This game is licensed under the MIT License.

## Copyright
Copyright (c) 2024 HabermannR

## GitHub
You can find the source code for this game on GitHub at https://github.com/HabermannR/Fantasy-Tribe-Game.
