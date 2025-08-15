import os
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter

# NLP & clustering
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Transformers
from transformers import pipeline
from sentence_transformers import SentenceTransformer

# BERTopic
from bertopic import BERTopic

# GSDMM
from gsdmm import MovieGroupProcess

# Word2Vec
from gensim.models import Word2Vec

# Load dataset
df = pd.read_csv("googleplaystore_user_reviews.csv")

# Text cleaning
def clean_text(text):
    text = re.sub(r'<[^>]+>', '', str(text))
    text = re.sub(r'[^\w\s]', '', text.lower())
    text = re.sub(r'\d+', '', text)
    return text.strip()

df['cleaned'] = df['content'].apply(clean_text)
df = df[df['cleaned'].str.strip().astype(bool)].reset_index(drop=True)

# TF-IDF + KMeans
tfidf = TfidfVectorizer(max_df=0.9, min_df=10, stop_words='english')
X_tfidf = tfidf.fit_transform(df['cleaned'])
kmeans = KMeans(n_clusters=10, random_state=42)
df['cluster'] = kmeans.fit_predict(X_tfidf)
print(f"Silhouette Score: {silhouette_score(X_tfidf, df['cluster']):.3f}")

# BERTopic
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(df['cleaned'].tolist(), show_progress_bar=True)
topic_model = BERTopic()
topics, probs = topic_model.fit_transform(df['cleaned'].tolist(), embeddings)
df['bertopic_cluster'] = topics

# GSDMM
df['tokens'] = df['cleaned'].apply(lambda x: x.split())
mgp = MovieGroupProcess(K=20, alpha=0.1, beta=0.3, n_iters=30)
vocab = set(word for doc in df['tokens'] for word in doc)
mgp.fit(df['tokens'].tolist(), len(vocab))
df['gsdmm_cluster'] = [mgp.choose_best_label(doc)[0] for doc in df['tokens']]

# Word2Vec + KMeans
tokenized_reviews = df['cleaned'].apply(lambda x: x.split())
w2v_model = Word2Vec(sentences=tokenized_reviews, vector_size=100, window=5, min_count=5, workers=4)
def get_sentence_vector(tokens):
    vectors = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(w2v_model.vector_size)
df['w2v_vector'] = tokenized_reviews.apply(get_sentence_vector)
X_w2v = np.vstack(df['w2v_vector'].values)
w2v_kmeans = KMeans(n_clusters=10, random_state=42)
df['w2v_cluster'] = w2v_kmeans.fit_predict(X_w2v)

# Zero-shot classification
classifier = pipeline("zero-shot-classification", model="distilbert-base-uncased")
df['distilbert_label'] = ""
for i, row in df.head(10).iterrows():
    result = classifier(row['content'], candidate_labels=["vague", "detailed"])
    df.at[i, 'distilbert_label'] = result['labels'][0]

# Flan-T5 response generation
generator = pipeline("text2text-generation", model="google/flan-t5-base")
vague_reviews = df[df['distilbert_label'] == "vague"].copy()
vague_reviews['flan_response'] = ""
for i, row in tqdm(vague_reviews.iterrows(), total=len(vague_reviews)):
    prompt = f"Respond to user complaint: '{row['content']}'"
    try:
        response = generator(prompt, max_length=50, do_sample=True)
        vague_reviews.at[i, 'flan_response'] = response[0]['generated_text']
    except Exception:
        vague_reviews.at[i, 'flan_response'] = "Error generating response"

df = df.merge(vague_reviews[['reviewId', 'flan_response']], on='reviewId', how='left')

# Export
df.to_csv("revai_full_pipeline_output.csv", index=False)
print("📦 Final export complete: revai_full_pipeline_output.csv")
