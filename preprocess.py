import json
import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer

# Download the specific NLTK tokenizer package
nltk.download('punkt_tab')

# Initialize the stemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    """Split sentence into an array of words/tokens"""
    return nltk.word_tokenize(sentence)

def stem(word):
    """Find the root form of the word (e.g., 'anxious' -> 'anxi')"""
    return stemmer.stem(word.lower())

# Let's test if it works!
if __name__ == "__main__":
    test_sentence = "I am feeling incredibly overwhelmed and anxious today."
    print(f"Original: {test_sentence}")
    
    # 1. Tokenize
    tokenized_words = tokenize(test_sentence)
    print(f"Tokenized: {tokenized_words}")
    
    # 2. Stem
    stemmed_words = [stem(w) for w in tokenized_words if w not in ['?', '!', '.', ',']]
    print(f"Stemmed: {stemmed_words}")