import numpy as np
import gensim
import argparse
import os
import re
from collections import Counter

WORDS = None

def w2v_load(args):
    model = gensim.models.KeyedVectors.load(args.w2v_path)
    return model

def w2v_make(fline, args):
    model = gensim.models.Word2Vec(fline, size=300, window=5, min_count=1, workers=20, iter=100)
    model.save(args.w2v_path)
    return model

def spellChecker(args):
    global WORDS

    if not os.path.exists(args.w2v_path):
        fline = open(args.file_path, 'r', encoding='cp949')
        sentence = [i.strip().split() for i in fline.readlines()]
        model = w2v_make(sentence, args)
    else:
        model = w2v_load(args)
    
    words = model.wv.index2word

    w_rank = {}
    for i, word in enumerate(words):
        w_rank[word] = i
    
    WORDS = w_rank
    
    # if error words is detected and call correction function
    # return correction words
    # correction(errorword)
    
def words(text): return re.findall(r'\w+', text)

def probability(word):
    # probability of word
    # use inverse of rank as proxy
    # returns 0 if the word isn't in the dictionary
    return - WORDS.get(word, 0)

def correction(word): 
    "Most probable spelling correction for word."
    return max(candidates(word), key=probability)

def candidates(word): 
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    print(set(w for w in words if w in WORDS))
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--w2v_path', type=str, help='word2vec bin path')
    parser.add_argument('--file_path', type=str, help='file path')
    args = parser.parse_args()
    spellChecker(args)
    #a = correction('찿아')
    #print(a)
    