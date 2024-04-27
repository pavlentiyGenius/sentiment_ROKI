import re
import demoji
from cleantext import clean
from pymorphy3 import MorphAnalyzer


class Preprocessing:
    def soft_preprocessing(text):
        processed = clean(text, to_ascii=False, lower=False, no_line_breaks=True, no_urls=True,no_emails=True,              
                          no_phone_numbers=True, no_numbers=True, no_digits=False, no_currency_symbols=True, no_punct=False, lang="ru" )
        
        processed = re.sub(r'\[id\d+\|.+\],', '', processed)
        processed = re.sub(r'@\W+,', '', processed)
        processed = ' '.join(re.sub("(@[A-Za-zА-Яа-я0-9]+)|(#[A-Za-zА-Яа-я0-9]+)|(\w+:\/\/\S+)", " ", processed).split())
        
        processed = demoji.replace(processed, ' ')
        
        processed = re.sub('_МАМИНОЗДОРОВЬЕ _СОВМЕСТИМОСТЬСГВ', '', processed)
        processed = re.sub('_lady', '', processed)
        
        return processed
        
        