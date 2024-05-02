import telebot
import pandas as pd
import numpy as np
from utils import sentiment
from utils import preprocessing

import io
import telebot
from telebot import types

import warnings
import logging

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)



# bot = telebot.TeleBot('===PROM BOT TOKEN===') # prom stand
bot = telebot.TeleBot('===IFT BOT TOKEN===') # ift stand

preproc = preprocessing.Preprocessing
# sentim = sentiment.Sentiment() # prom stand
sentim = sentiment.Sentiment_TEST() # ift stand

@bot.message_handler(commands=['start', 'button'])
def welcome(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    bot.send_message(message.chat.id, "Определение тональности текстов\nДобавьте файл в формате .xlsx\nСтолбец с текстом для обработки должен называть 'Текст сообщения'")


@bot.message_handler(content_types=['document'])
def handle_docs_photo(message):
    try:
        chat_id = message.chat.id

        file_info = bot.get_file(message.document.file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        df = pd.read_excel(downloaded_file)
        
        minutes = round(df.shape[0] * 1.2,) // 60
        seconds = round(df.shape[0] * 1.2,) % 60
        bot.reply_to(message, f"Взял в обработку, нужно немного времени\nПримерное время обработки {minutes}:{seconds}")
        
        redused = df[~df['Текст сообщения'].isna()]
        redused['Текст сообщения'] = redused['Текст сообщения'].apply(lambda x: preproc.soft_preprocessing(x))
        sentences = redused['Текст сообщения'].tolist()
        sentiment_result, sureness = sentim.get_sentiment(sentences)
        redused['model sentiment'] = sentiment_result
        redused['model sureness'] = sureness
        
        df = df.join(redused[['model sentiment', 'model sureness']])
            
        buf = io.BytesIO()
        
        df.to_excel(buf, encoding='utf-8', index = False, header = True, )
        
        filename_old = message.document.file_name
        filename_new = filename_old.split('.')[0] + '__processed__.' + filename_old.split('.')[1]
        
        bot.send_document(chat_id, buf.getvalue(), visible_file_name=filename_new)
#         bot.send_message(message.chat.id, text="Файл обработан. Переименуйте его с расширением .xlsx")
        buf.close()
        
    except Exception as e:
        bot.reply_to(message, e)    
    
logging.info('Server started')
bot.polling(none_stop=True, interval=0)