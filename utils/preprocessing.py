import re
import demoji
from cleantext import clean
from pymorphy3 import MorphAnalyzer


class Preprocessing:
    def __init__(self, stopwords):
        '''
        stopwords list: list of stopwords 
        '''
        self.stopwords = stopwords
        self.morph = MorphAnalyzer()
    
    def soft_preprocessing(self, text):
        '''
        cleaning for NN
        '''
        
        processed = clean(text, to_ascii=False, lower=False, no_line_breaks=True, no_urls=True,no_emails=True,              
                          no_phone_numbers=True, no_numbers=True, no_digits=False, no_currency_symbols=True, no_punct=False, lang="ru" )
        
        processed = re.sub(r'\[id\d+\|.+\],', '', processed)
        processed = re.sub(r'@\W+,', '', processed)
        processed = ' '.join(re.sub("(@[A-Za-zА-Яа-я0-9]+)|(#[A-Za-zА-Яа-я0-9]+)|(\w+:\/\/\S+)", " ", processed).split())
        
        processed = demoji.replace(processed, ' ')
        
        processed = re.sub('_МАМИНОЗДОРОВЬЕ _СОВМЕСТИМОСТЬСГВ', '', processed)
        processed = re.sub('_lady', '', processed)
        
        return processed
    
    def hard_preprocessing(self, text):
        '''
        better use after soft_processing()
        '''
        
        text = text.lower()
        text = re.sub(r'[^\w]', ' ', text)
        text = re.sub(r'_', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        text = text.split()
        text = [self.morph.normal_forms(i)[0] for i in text]
        text = [i for i in text if i not in self.stopwords]
        text = ' '.join(text)
        
        return text
        
        