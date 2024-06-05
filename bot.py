import pandas as pd
import numpy as np
from utils import sentiment
from utils import preprocessing
from utils import relevance

import io
import telebot
import json
from telebot import types

import warnings
import logging
import telebot
import time


warnings.filterwarnings("ignore")

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

with open('secrets.json', 'r') as json_file:
    secrets = json.load(json_file)
    
stand = secrets['ift']
token = stand['token']
logging.info('Secrets loaded')

bot = telebot.TeleBot(token) # ift stand
    
# stopwords    
with open('stopwords-ru.txt', 'r') as f:
    stopwords_ru = f.read() 
    
stopwords_ru = stopwords_ru.split('\n')

with open('stopwords-eng.txt', 'r') as f:
    stopwords_eng = f.read() 
    
stopwords_eng = stopwords_eng.split('\n')
stopwords = stopwords_ru + stopwords_eng
logging.info('Stopwords loaded')

path = 'relevance_model'
model_name = 'clf_relevance_05_06.pickle'
vec_name = 'vec_relevance_05_06.pickle'

rel = relevance.Relevance(path, model_name, vec_name)
logging.info('Relevance model loaded')

cleaning = preprocessing.Preprocessing(stopwords)
logging.info('Preprocessing util loaded')
# sentim = sentiment.Sentiment() # load with model
sentim = sentiment.Sentiment() # test model without model
logging.info('Sentiment model loaded')


@bot.message_handler(commands=['start'])
def welcome(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    bot.send_message(message.chat.id, "Определение тональности/релевантности текстов\nДобавьте файл в формате .xlsx\nСтолбец с текстом для обработки должен называть 'Текст сообщения'")


@bot.message_handler(content_types=['document'])
def handle_docs_photo(message):
    try:
        chat_id = message.chat.id

        file_info = bot.get_file(message.document.file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        df = pd.read_excel(downloaded_file)
        
        minutes = round(df.shape[0] * 1.2,) // 60
        seconds = round(df.shape[0] * 1.2,) % 60
        logging.info('File has loaded')
        bot.reply_to(message, f"Взял в обработку, нужно немного времени\nПримерное время обработки {minutes}:{seconds}")

        redused = df[~df['Текст сообщения'].isna()]
#         print(redused.shape)
        redused = redused.drop_duplicates(subset='Текст сообщения')
#         print(redused.shape)
        
        redused['Текст сообщения_soft_cleaned'] = redused['Текст сообщения'].apply(lambda x: cleaning.soft_preprocessing(x))
        redused['Текст сообщения_hard_cleaned'] = redused['Текст сообщения'].apply(lambda x: cleaning.hard_preprocessing(x))
        logging.info('Texts are processed')
        
        sentences = redused['Текст сообщения_soft_cleaned'].tolist()
        sentiment_result, sureness = sentim.get_sentiment(sentences)
        redused['model sentiment'] = sentiment_result
        redused['model sureness'] = sureness
        logging.info('Sentiment is predicted')
        
        sentences_hard_cleaned = redused['Текст сообщения_hard_cleaned'].tolist()
        relevance_result = rel.get_relevance(sentences)
        redused['model relevance'] = relevance_result
        redused['model relevance'] = redused['model relevance'].map({1:'нерелевантный', 0:'релевантный'})
        logging.info('Relevance is predicted')
        
#         df = df.join()
        df = df.merge(redused[['Текст сообщения', 'model sentiment', 'model sureness', 'model relevance']], on='Текст сообщения', how='left')
            
        buf = io.BytesIO()
        
        df.to_excel(buf, encoding='utf-8', index = False, header = True, )
        
        filename_old = message.document.file_name
        filename_new = filename_old.split('.')[0] + '__processed__.' + filename_old.split('.')[1]
        
        bot.send_document(chat_id, buf.getvalue(), visible_file_name=filename_new)
#         bot.send_message(message.chat.id, text="Файл обработан. Переименуйте его с расширением .xlsx")
        logging.info('File sent successifully')
        buf.close()
        
    except Exception as e:
        bot.reply_to(message, e)    
    
logging.info('Server started')
bot.polling(none_stop=True, interval=0)