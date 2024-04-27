from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class Sentiment:
    def __init__(self):
        # self.model = SentenceTransformer('tuned_model_v3').eval()
        self.model = SentenceTransformer('pavlentiy/reviews-sentiment-multilingual-e5-base').eval()
        self.embeddings_classes = self.model.encode(['негатив', 'нейтрально', 'позитив'])
        
        
    def get_sentiment(self, sentences):
        embeddings = self.model.encode(sentences)
        sim_values = cosine_similarity(embeddings, self.embeddings_classes)
        simularity_result = sim_values.argmax(axis=1)
        
        mapping = {
            0:'негативная', 
            1:'нейтральная', 
            2:'позитивная'}
            
        sentiment_result = [mapping.get(i) for i in simularity_result]
        sureness = list(sim_values.max(axis=1))
        return sentiment_result, sureness
        
class Sentiment_TEST:
    def __init__(self):
        pass
        
        
    def get_sentiment(self, sentences):
        
        sentiment_result = ['нейтральная_test'] * len(sentences)
        sureness = [0.0] * len(sentences)
        
        return sentiment_result, sureness