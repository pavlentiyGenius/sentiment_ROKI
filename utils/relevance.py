import pickle
import sklearn


class Relevance:
    def __init__(self, path, model_name, vec_name):
        with open(f'{path}/{model_name}', 'rb') as f:
            self.clf = pickle.load(f)

        with open(f'{path}/{vec_name}', 'rb') as f:
            self.vec = pickle.load(f)
            
            
    def get_relevance(self, sentences):
        '''
        1 - not relevant text
        0 - relevant (ok) text
        '''
        vectors = self.vec.transform(sentences)
        predict = self.clf.predict(vectors)
        return predict