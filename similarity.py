import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.feature_extraction.text import TfidfVectorizer
import jellyfish

def jaccard_similarity(set1, set2):
    #calculate Jaccard similarity
    intersection = len(set1.intersection(set2))
    union = len(set1) + len(set2) - intersection
    return intersection / union

def cos_similarity(x : str, y : str):
    # Compute cosine similarity using TF-IDF vectors
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([x, y])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return cosine_sim

def euclidean_dist_similarity(x : str, y : str):
    x_set = set(x)
    y_set = set(y)
    jaccard_sim = jaccard_similarity(x_set, y_set)
    return jaccard_sim

def pearson_corr_similarity(x : str, y : str):
    # Compute Pearson correlation coefficient
    pearson_corr = np.corrcoef([x, y])[0][1]
    return pearson_corr

def levenshtein_dist_similarity(x : str, y : str):
    # Compute Levenshtein distance
    levenshtein_dist = jellyfish.levenshtein_distance(x, y)
    return levenshtein_dist

def similarity_functions() -> list:
    return [
        jaccard_similarity,
        cos_similarity,
        euclidean_dist_similarity,
        pearson_corr_similarity,
        levenshtein_dist_similarity
        ]