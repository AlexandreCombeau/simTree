import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.feature_extraction.text import TfidfVectorizer
import jellyfish
from functools import reduce
def flatten(x) -> str:
    return reduce(lambda acc,x : acc+x,x,"")

def jaccard_similarity(x: str, y : str) -> float:
    x = flatten(x)
    y = flatten(y)
    #calculate Jaccard similarity
    x = set(x)
    y = set(y)
    intersection = len(x.intersection(y))
    union = len(x) + len(y) - intersection
    return intersection / union

def cos_similarity(x : str, y : str):
    x = flatten(x)
    y = flatten(y)
    # Compute cosine similarity using TF-IDF vectors
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([x, y])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return cosine_sim



def damerau_levenshtein_dist_similarity(x : str, y: str) -> float:
    x = flatten(x)
    y = flatten(y)
    nb_diff= jellyfish.damerau_levenshtein_distance(x,y)
    sim = 1 - nb_diff / (max(len(x),len(y)))
    return sim

def jaro_similarity(x : str, y : str) -> float:
    x = flatten(x)
    y = flatten(y)
    jaro = jellyfish.jaro_similarity(x,y)
    return jaro


#####################################
############### TODO ################
#####################################
#faire attention avec les distances
#comprend prefix
def jarowinkler_similarity(x : str, y : str) -> float:
    x = flatten(x)
    y = flatten(y)    
    jarowinkler = jellyfish.jaro_winkler_similarity(x,y)
    return jarowinkler
#word2vec
def euclidean_dist_similarity(x : str, y : str) -> float:
    pass

def tf_idf_similarity(x : str, y : str) -> float:
    pass

def pearson_corr_similarity(x : str, y : str):
    # Compute Pearson correlation coefficient
    pearson_corr = np.corrcoef([x, y])[0][1]
    return pearson_corr

#very similar to hamming 
def levenshtein_dist_similarity(x : str, y : str):
    # Compute Levenshtein distance
    levenshtein_dist = jellyfish.levenshtein_distance(x, y)
    return levenshtein_dist

def hamming_dist_similarity(x: str, y: str) -> float:
    """Return the Hamming distance between equal-length sequences."""
    nb_diff = jellyfish.hamming_distance(x,y)
    sim = 1 - nb_diff / (max(len(x),len(y)))
    return sim

def similarity_functions() -> list:
    return [
        jaccard_similarity,
        cos_similarity,
        damerau_levenshtein_dist_similarity,
        jaro_similarity,
        ]