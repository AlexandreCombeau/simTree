import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

def lowercase(x : str) -> str:
    return x.lower()

def uppercase(x : str) -> str:
    return x.upper()

#remove leading and ending whitespace
def strip_whitespace(x : str) -> str:
    return x.strip()

def remove_whitespace(x : str) -> str:
    return ''.join(x.split())

def remove_punctuation(x : str) -> str:
    return x.translate(str.maketrans('', '', string.punctuation))

#####################################
######### Numerical methods #########
#####################################

#TODO

#####################################
############ NLP methods ############
#####################################
nltk.download('punkt')
def stem(x : str) -> str:
    stemmer = PorterStemmer()
    tokens = nltk.word_tokenize(x)
    stems = [stemmer.stem(token) for token in tokens]
    return ' '.join(stems)

nltk.download('stopwords')
def remove_stopwords(x : str) -> str:
    stop_words = set(stopwords.words('english'))
    tokens = x.split()
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(filtered_tokens)

#nltk.download('punkt')
def tokenize(x : str) -> list[str]:
    return nltk.word_tokenize(x)

#maybe too long
nltk.download('wordnet')
def lemmatize(x : str) -> str:
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(x)
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmas)

def transformation_functions() -> list:
    return [
        lowercase,
        uppercase,
        strip_whitespace,
        remove_whitespace,
        remove_punctuation,
        stem,
        remove_stopwords,
        tokenize,
        #lemmatize
    ]

