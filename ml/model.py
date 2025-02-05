import pandas as pd
import pickle
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity

class Recommender:
    def __init__(self):
        self.movies = pd.read_csv("data/movies.csv")
        self.tokenized_titles = [title.lower().split() for title in self.movies["title"].tolist()]
        self.bm25 = self._load_bm25()

    def _load_bm25(self):
        try:
            with open("ml/bm25.pkl", "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            print("BM25 index not found. Generating BM25 index...")
            return self._generate_bm25()

    def _generate_bm25(self):
        bm25 = BM25Okapi(self.tokenized_titles)
        with open("ml/bm25.pkl", "wb") as f:
            pickle.dump(bm25, f)
        print("BM25 index generated and saved.")
        return bm25

    def get_recommendations(self, movie_title, top_n=4):
        query_tokens = movie_title.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        similar_movies = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[1:top_n+1]
        return [self.movies.iloc[i[0]]['title'] for i in similar_movies]

# Initialize once for the API
model = Recommender()
