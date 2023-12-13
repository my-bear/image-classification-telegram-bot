# Telegram bot 

## Create your telegram bot based on this. Connect modules and models to suit your needs.

Pre-installed models:  
   - GPT-3.5-turbo OpenAI  
   - DALL-E OpenAI  
   - Handwritten digit recognition


This repository is designed for demonstration purposes showing the general principle of project development and its structure. It is based on a convolutional neural network model that helps users achieve their goals by interacting through bot telegrams. The owner, who intends to use this as the basis of his application, trains the model on the data provided by him. The final path and purpose of the project is determined by the owner.

Optimal hyperparameters installed by default in config file training_config.json.

<br/>

# Getting Started

Clone repository 
```
git clone https://github.com/my-bear/image-classification-telegram-bot
```

Сreate a virtual environment at the root of the project
```
python -m venv
```
Run traning specifying arguments using config file training_config.json and destination_path.json or the console command, for example.

```
python scripts/train.py --path_data data/processed --path_checkpoint /chekpoints
```
The model may end with an unexpected stop during training, which means that the quality of the model's prediction has stopped improving. Select the best model from the checkpoints folder and place it in the model directory. This is the main model on which predictions will be based.

To get the data of a trained model from training data or testing on validation data, Run inference specifying arguments using config file inference_config.json and destination_path.json or the console command, for example

```
python scripts/evaluate.py --path_data data/processed --model_load /model
```
To see the output of the training model to build a graph, run the script plotting

```
python scripts/plotting.py --model_load /model
```
Launch a telegram bot with your token. 
Create a ".env" file and put a token in it
```
python scripts/telegram_bot.py
```
