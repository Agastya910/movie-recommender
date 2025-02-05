import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class Recommender:
    def __init__(self):
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.movies = pd.read_csv("data/movies.csv")
        self.embeddings = self._load_embeddings()

    def _load_embeddings(self):
        try:
            with open("ml/embeddings.pkl", "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            return self._generate_embeddings()

    def _generate_embeddings(self):
        embeddings = self.model.encode(self.movies['title'].tolist())
        with open("ml/embeddings.pkl", "wb") as f:
            pickle.dump(embeddings, f)
        return embeddings

    def get_recommendations(self, movie_title, top_n=4):
        idx = self.movies[self.movies['title'] == movie_title].index[0]
        sim_scores = cosine_similarity([self.embeddings[idx]], self.embeddings)[0]
        similar_movies = sorted(enumerate(sim_scores), key=lambda x: x[1], reverse=True)[1:top_n+1]
        return [self.movies.iloc[i[0]]['title'] for i in similar_movies]

# Initialize once for the API
model = Recommender()