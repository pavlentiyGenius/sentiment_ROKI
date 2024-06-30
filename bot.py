import re
import io
import telebot
import json
from telebot import types

import warnings
import logging
import telebot
import time

import pandas as pd
import numpy as np
import torch

from utils import preprocessing
from utils import sentiment
from utils import aspect_extractor
from utils import relevance
#from utils.sentiment_ff import BertClassifier, SenitmentTorch

from pymorphy3 import MorphAnalyzer

from tqdm import tqdm
tqdm.pandas()

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
data_path = 'data'

logging.info('Secrets loaded')

bot = telebot.TeleBot(token) # ift stand

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Inference found {device} device')

with open(f'{data_path}/stopwords-ru.txt', 'r') as f:
    stopwords_ru = f.read() 
    
stopwords_ru = stopwords_ru.split('\n')

with open(f'{data_path}/stopwords-eng.txt', 'r') as f:
    stopwords_eng = f.read() 
    
stopwords_eng = stopwords_eng.split('\n')
stopwords = stopwords_ru + stopwords_eng
logging.info('Stopwords loaded')

path = 'relevance_model'
model_name = 'clf_relevance_30_05.pickle'
vec_name = 'vec_relevance_30_05.pickle'

rel = relevance.Relevance(path, model_name, vec_name)
logging.info('Relevance model loaded')

cleaning = preprocessing.Preprocessing(stopwords)
logging.info('Preprocessing util loaded')

#model = BertClassifier().to(device)
#model.load_state_dict(torch.load('sentiment_model/LaBSE_head_v2.pth', map_location=torch.device(device)))
#model.eval()
#sentiment = SenitmentTorch(model, device)

sentiment_inference = sentiment.Sentiment() # load with model
logging.info('Sentiment model loaded')

aspects_extract = aspect_extractor.Aspects()
aspects_extract.load_list_medicine(f'{data_path}/list_meds.txt')

logging.info('Medication list loaded')

def rasparse(results):
    '''
    Функция для агрегации итогового результата
    '''
    new_list = []
    for res in results:
        new_list.append([res[0], res[1], res[2]])
    return new_list

def get_aspect_sentiment_sentences(text):
    '''
    Сценарий детекции сентимента по контексту препарата
    Input:
        text (str): текст для поиска
    Output:
        result (list): список с результатами
    '''
    results = []
    
    # test_frame = pd.DataFrame(text, columns=['Текст сообщения'])
    # test_frame_loader = sentiment.prepare_dataloader(test_frame)
    
    # Если в тексте есть данное ключевое слово, то точно нейтральный сентимент
    # if 'Текст на изображении:' in text:
    #     results.append(['none', 'none', 'нейтральная'])
    #     results = rasparse(results)
    #     return results
    
    text = cleaning.soft_preprocessing(text)
    med_tokens = aspects_extract.get_med_tokens_from_sent(text) # получаем препараты и контекст
    
    # если ничего из препаратов не найдено, то возвращаем сентимент всего текста
    if len(med_tokens) == 0:
        aspect_sentiment = sentiment_inference.get_sentiment([text])
        # aspect_sentiment = sentiment.prediction(text)
        results.append(['none', 'none', aspect_sentiment[0][0]])
    
    # если много препаратов скорее всего мусор
    elif len(med_tokens) >= 4:
        results.append(['none', 'none', 'нейтральная'])
        
    # если найдено, то возвращаем сентимент каждой части
    elif len(med_tokens) >= 1:
        for part in med_tokens:
            aspect_sentiment = sentiment_inference.get_sentiment([part[3]])
            # aspect_sentiment = sentiment.prediction(part[3])
            results.append([part[0], part[3], aspect_sentiment[0][0]])

    results = rasparse(results)
    return results


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
        redused = redused.drop_duplicates(subset='Текст сообщения')
        
#        redused['Текст сообщения_soft_cleaned'] = redused['Текст сообщения'].apply(lambda x: cleaning.soft_preprocessing(x))
        redused['Текст сообщения_hard_cleaned'] = redused['Текст сообщения'].progress_apply(lambda x: cleaning.hard_preprocessing(x))
        logging.info('Texts are processed')
        
        
        redused['analysis'] = redused['Текст сообщения'].progress_apply(get_aspect_sentiment_sentences)
#        sentences = redused['Текст сообщения_soft_cleaned'].tolist()
#        sentiment_result, sureness = sentim.get_sentiment(sentences)
#        redused['model sentiment'] = sentiment_result
#        redused['model sureness'] = sureness


        redused_exploded = redused.explode('analysis')
        redused_exploded['aspect'] = redused_exploded['analysis'].apply(lambda x: x[0])
        redused_exploded['aspect_context'] = redused_exploded['analysis'].apply(lambda x: x[1])
        redused_exploded['aspect_sentiment_pred'] = redused_exploded['analysis'].apply(lambda x: x[2])
        logging.info('Sentiment is predicted')
        
        sentences_hard_cleaned = redused['Текст сообщения_hard_cleaned'].tolist()
        relevance_result = rel.get_relevance(sentences_hard_cleaned)
        redused['model relevance'] = relevance_result
        redused['model relevance'] = redused['model relevance'].map({1:'нерелевантный', 0:'релевантный'})
        logging.info('Relevance is predicted')
        
#         df = df.join()
        redused_exploded = redused_exploded.merge(redused[['Текст сообщения', 'model relevance']], on='Текст сообщения', how='left')
            
        buf = io.BytesIO()
        
        redused_exploded.to_excel(buf, encoding='utf-8', index = False, header = True, )
        
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