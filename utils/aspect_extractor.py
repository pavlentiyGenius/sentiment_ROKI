import re
import numpy as np
import pandas as pd
from itertools import chain

from fuzzywuzzy import fuzz
from fuzzywuzzy import process

from itertools import groupby

from nltk import word_tokenize

class Aspects:
    def __init__(self):
        self.list_unique_med = None 
    
    def load_list_medicine(self, filename):
        with open(filename, 'r', encoding='utf8') as f:
            self.list_unique_med = f.read().split('\n')[:-1]
        
    def get_list_unique_med(self):
        return self.list_unique_med
    
    def lewinshtain_nearest(self, word, sorting=True, top=True, thrashold=87):
        '''
        Функция близости слова к слову из списка
        Input:
            word (str): слово
            list_of_words (list): список слов для поиска
            sorting (bool): флаг сортировки слов в порядке убывания скоров
            top (bool): вывод топ слов по заданному трешхолду
            thrashold (int): порог отбора слов (от 0 до 100)
        Output:
            ratio (list): список слов с близостью 
        '''
        ratio = {}
        for i in self.list_unique_med:
            ratio[i] = fuzz.ratio(word, i)

        if sorting:
            ratio = {k: v for k, v in sorted(ratio.items(),reverse=True, key=lambda item: item[1])}
            if top:
                return {k: v for k, v in ratio.items() if v >= thrashold}
            else:
                return ratio
        else:
            if top:
                return {k: v for k, v in ratio.items() if v >= thrashold}
            else:
                return ratio
            
    def merge_context(self, res):
        '''
        Функция соединения контекстов. Когда название препарата встречается более одного раза в тексте
        Input:
            res (): результат работы метода get_med_tokens_from_sent()
        Output:
            result
        '''
        new_medicals = []
        new_result_ratios = []
        new_token_num = []
        new_contexts = []

        for key, group in groupby(sorted(res, key= lambda x: x[0]), lambda x: x[0]):
            new_medicals.append(key)

            new_result_ratios_tmp = []
            new_token_num_tmp = []
            new_contexts_tmp = []
            for thing in group:
                new_result_ratios_tmp.append(thing[1])
                new_token_num_tmp.append(thing[2])
                new_contexts_tmp.append(thing[3])
            new_result_ratios.append(np.mean(new_result_ratios_tmp))
            new_token_num.append(new_token_num_tmp)
            new_contexts.append(' '.join(new_contexts_tmp))

        result = list(zip(new_medicals, new_result_ratios, new_token_num, new_contexts))
        return result
    
    def merge_and_cases(self, res):
    
        new_res = [list(i) for i in res]

        for i in new_res:
            cont = i[3]
            tok = [i.lower() for i in word_tokenize(cont)]
            if (len(tok) <= 3) and ('и' in tok):
                save_i = i[2]
                for j in new_res:
                    if (save_i + 2) == j[2]:
                        # print(i[3] + ' ' + j[3])
                        concat = i[3] + ' ' + j[3]
                        i[3] = concat
                        j[3] = concat 

        new_res = [tuple(i) for i in new_res]
        return new_res
    
    def get_med_tokens_from_sent(self, text):
        '''
        Функция поиска препарата с контекстом в тексте
        Input: 
            text(str): текст
        Output:
            result_ratios (float): расстояние левенштейна, уверенность в слове 
            token_num (int): номер токена в списке
            contexts (str): контекст препарата
        '''
        
        # sentence_text = re.sub(r'\s+', ' ', ', '.join(text.split(','))) # ставим пробелы к запятым
        # sentence_text = re.sub(r'\.', ' ', sentence_text) # удаляем точки 
        tokens = [i for i in word_tokenize(text)]
        # print(tokens)
        
        medicals = []
        result_ratios = []
        token_num = []
        contexts = []
        context = []
        first_flag = True
        
        for i, word in enumerate(tokens):
            ratios = self.lewinshtain_nearest(word.lower(), top=True)
            context.append(word)
            if len(ratios) >= 1:
                medical = list(ratios.items())[0][0]
                score_ratios = list(ratios.items())[0][1]
                if first_flag:
                    medicals.append(medical)
                    result_ratios.append(score_ratios)
                    token_num.append(i)
                    first_flag=False
                else:
                    medicals.append(medical)
                    result_ratios.append(score_ratios)
                    token_num.append(i)
                    context = context[:-1]
                    
                    context = ' '.join(context)
                    context = re.sub(r'< NUMBER >', '<NUMBER>', context)
                    context = re.sub(r'< PHONE >', '<PHONE>', context)
                    context = re.sub(r'< EMAIL >', '<EMAIL>', context)
                    context = re.sub(r'< URL >', '<URL>', context)
                    context = re.sub(r' ([^A-ZА-Яa-zа-я0-9<])', r'\1', context)
                    contexts.append(context)
                    
                    context = []
                    context.append(word)
                    
        context = ' '.join(context)
        context = re.sub(r'< NUMBER >', '<NUMBER>', context)
        context = re.sub(r'< PHONE >', '<PHONE>', context)
        context = re.sub(r'< EMAIL >', '<EMAIL>', context)
        context = re.sub(r'< URL >', '<URL>', context)
        context = re.sub(r' ([^A-ZА-Яa-zа-я0-9<])', r'\1', context)
        contexts.append(context)

        result = list(zip(medicals, result_ratios, token_num, contexts))
        
        
        if len(result_ratios) != 0: 
            result = self.merge_and_cases(result)
            result = self.merge_context(result)

        return result