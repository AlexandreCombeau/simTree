import string
from functools import reduce
from typing import Union, Callable

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

valueType = Union[str, list[str]]

lift = lambda f,l : list(map(f,l))
clean = lambda x: [e for e in x if e]
lift_if_list = lambda f,x : f(x) if isinstance(x,str) else clean(lift(f,x))

def identity(x : valueType) -> valueType:
    return x

def lowercase(x : valueType) -> valueType:
    def _lowercase(x : str) -> str: 
        return x.lower()
    return lift_if_list(_lowercase,x)

def uppercase(x : valueType) -> valueType:
    def _uppercase(x : str) -> str:
        return x.upper()
    return lift_if_list(_uppercase,x)

#remove leading and ending whitespace
def strip_whitespace(x : valueType) -> valueType:
    def _strip_whitespace(x : str) -> str:
        return x.strip()
    return lift_if_list(_strip_whitespace,x)

def remove_whitespace(x : valueType) -> valueType:
    def _remove_whitespace(x : str) -> str:
        return ''.join(x.split())
    return lift_if_list(_remove_whitespace,x)

def remove_punctuation(x : valueType) -> valueType:
    def _remove_punctuation(x : str) -> str:
        return x.translate(str.maketrans('', '', string.punctuation))
    return lift_if_list(_remove_punctuation,x)

def flatten(x : valueType) -> str:
    return reduce(lambda acc,x : acc+x,x,"")
#####################################
######### Numerical methods #########
#####################################

def split_alphanum(x : valueType) -> valueType:
    def _split_alphanum(x : str) -> str:
        strings = [""]
        num = [""]
        symbols = [""]
        string_order = []
        for e in x:
            if e.isalpha():
                if not(strings[0]):
                    string_order.append(strings)
                strings[0]+=e
            elif e.isnumeric():
                if not(num[0]):
                    string_order.append(num)
                num[0]+=e
            elif e in string.punctuation:
                if not(symbols[0]):
                    string_order.append(symbols)
                symbols[0]+=e
        flattened = reduce(lambda acc,x : acc+x[0]+" ",string_order,"")
        return flattened
    return lift_if_list(_split_alphanum,x)

def sort_string(x : valueType) -> valueType:
    def _sort_string(x : str) -> str:
        return ''.join(sorted(x))
    return lift_if_list(_sort_string,x)


def removeNumber(x : valueType) -> valueType:
    def _removeNumber(x : str) -> str:
        return ''.join([i for i in x if i not in "0123456789"])
    return lift_if_list(_removeNumber,x)

def removeLetter(x : valueType) -> valueType:
    def _removeLetter(x : str) -> str:
        return ''.join([i for i in x if i in "0123456789"+string.punctuation])
    return lift_if_list(_removeLetter,x)
#TODO

#####################################
############ NLP methods ############
#####################################
nltk.download('punkt')
def stem(x : valueType) -> valueType:
    def _stem(x : str) -> str:
        stemmer = PorterStemmer()
        tokens = nltk.word_tokenize(x)
        stems = [stemmer.stem(token) for token in tokens]
        return ' '.join(stems)
    return lift_if_list(_stem,x)


nltk.download('stopwords')
def remove_stopwords(x : valueType) -> valueType:
    def _remove_stopwords(x : str) -> str:
        stop_words = set(stopwords.words('english'))
        tokens = x.split()
        filtered_tokens = [token for token in tokens if token not in stop_words]
        return ' '.join(filtered_tokens)
    return lift_if_list(_remove_stopwords,x)

#nltk.download('punkt')
def tokenize(x : valueType) -> list[str]:
    if isinstance(x,list):
        return x
    return nltk.word_tokenize(x)

#maybe too long
nltk.download('wordnet')
def lemmatize(x : valueType) -> valueType:
    def _lemmatize(x : str) -> str:
        lemmatizer = WordNetLemmatizer()
        tokens = nltk.word_tokenize(x)
        lemmas = [lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(lemmas)
    return lift_if_list(_lemmatize,x)

def transformation_functions() -> list:
    return [
        identity,
        lowercase,
        uppercase,
        strip_whitespace,
        remove_whitespace,
        remove_punctuation,
        flatten,
        stem,
        remove_stopwords,
        tokenize,
        split_alphanum,
        sort_string,
        #removeNumber,
        #removeLetter,
        #lemmatize
    ]


def get_tf_function_from_name(name : str) -> Callable[[valueType],valueType]:
    name_to_sim_function = {}
    for f in transformation_functions():
        name_to_sim_function[f.__name__] = f
    return name_to_sim_function[name]