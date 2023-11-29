import os
import logging
import dotenv

from openai import AsyncOpenAI
from image_recognition.predict.num import PredictImage

# Сonnecting the token
dotenv_path = os.path.join(os.getcwd(), 'scripts/.env')
dotenv.load_dotenv(dotenv_path)
OPENAI_TOKEN = os.getenv("OPENAI_TOKEN")
client = AsyncOpenAI(api_key = OPENAI_TOKEN)


async def generate_text(prompt) -> dict:
    '''
    Function for generating text in gpt-3.5-turbo
    '''
    try:
        response = await client.chat.completions.create(
            model = "gpt-3.5-turbo",
            messages = [
                {"role": "user", "content": prompt}
            ]
        )
        # return response
        # 
        return response.choices[0].message.content, response.usage.total_tokens
    except Exception as e:
        logging.error(e)

  
async def generate_image(prompt, n = 1, size = "1024x1024") -> list[str]:
    '''
    Function for generating an image in Dalle
    '''
    try:
        response = await client.images.generate(
            model="dall-e-2",
            prompt = prompt,
            n = n,
            size = size
        )
        return response.data[0].url
    except Exception as e:
        logging.error(e)
        return []


predict = PredictImage()

def predict_image(id)-> str:
    '''
    Function for guessing handwritten numbers
    '''
    try:
        response = predict(id)
    
    except Exception as e:
        logging.error(e)
        return " "
    else:
        return response