from sentence_transformers import SentenceTransformer, util
import numpy as np

class Sentiment:
    def __init__(self):
        # self.model = SentenceTransformer('tuned_model_v3').eval()
        self.model = SentenceTransformer('pavlentiy/reviews-sentiment-multilingual-e5-base').eval()
        self.embeddings_classes = self.model.encode(['негатив', 'нейтрально', 'позитив'], show_progress_bar=False)
        
        
    def get_sentiment(self, sentences):
        embeddings = self.model.encode(sentences, show_progress_bar=False)
        
        # Compute cosine-similarities
        cosine_scores = np.array(util.cos_sim(embeddings, self.embeddings_classes))

        a = lambda t: {0:'негативная', 1:'нейтральная', 2:'позитивная'}[t]
        argmax = cosine_scores.argmax(axis=1)
        simularity_result = list(map(a, argmax))
        
        sureness = list(cosine_scores.max(axis=1))
        return simularity_result, sureness
        
class Sentiment_TEST:
    def __init__(self):
        pass
        
        
    def get_sentiment(self, sentences):
        
        sentiment_result = ['нейтральная_test'] * len(sentences)
        sureness = [0.0] * len(sentences)
        
        return sentiment_result, sureness