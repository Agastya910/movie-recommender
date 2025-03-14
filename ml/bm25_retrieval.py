from rank_bm25 import BM25Okapi
import pandas as pd

# Load synthetic profiles
profiles_df = pd.read_csv("synthetic_profiles_clustered.csv")

# Create genre documents for BM25
genre_documents = []
for cluster_id in profiles_df["cluster"].unique():
    cluster_movies = df[df["cluster"] == cluster_id]["Series_Title"].tolist()
    genre_documents.append(" ".join(cluster_movies))

# Initialize BM25
bm25 = BM25Okapi([doc.split() for doc in genre_documents])

# Function to retrieve top documents
def retrieve_top_documents(query, top_k=5):
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
    return [genre_documents[i] for i in top_indices]

# Example usage
query = "I like action movies with car chases."
top_docs = retrieve_top_documents(query)
print("Top Documents:", top_docs)
