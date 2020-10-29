import gensim
import argparse
import os
from tqdm import tqdm
from collections import Counter
import difflib
import sys
sys.path.append('../')
from typo import jamoSplit, mergeCharSplit

SIMILARITY_THRESHOLD = 0.47

def makeFastText(fline, args):
    model = gensim.models.FastText(fline, size=300, sg=1, workers=40, word_ngrams=6, min_count=5, iter=100)
    model.save(args.w2v_path)
    return model

def w2v_load(args):
    model = gensim.models.KeyedVectors.load(args.w2v_path)
    return model

def jamosplitFile(args):
    f = open(args.file_path, 'r', encoding='cp949')
    sentence = [i.strip().split() for i in f.readlines()]
    
    w = open(args.jamo_path ,'w', encoding='cp949')
    for i in tqdm(sentence):
        temp = []
        for token in i:
            split_token, _ = jamoSplit(token) # [['ㅎ', 'ㅏ', ' '], ['ㅈ', 'ㅣ', ' '], ['ㅁ', 'ㅏ', 'ㄴ']]
            temp2 = []
            for eumjol in split_token:
                temp2.append(''.join(eumjol).replace(' ','-')) # 종성에서 공백은 -로 변경
            temp.append(''.join(temp2))
            #print(temp)
        w.write(' '.join(temp)+'\n')
    w.close()
    return 0

def spellChecker(args):
    if args.mode == 'jamo':
        if not os.path.exists(args.jamo_path):
            jamosplitFile(args)
        
        if not os.path.exists(args.w2v_path):
            f = open(args.jamo_path,'r', encoding='cp949')
            f = [i.strip().split() for i in f.readlines()]
            model = makeFastText(f, args)
        else:
            model = w2v_load(args)
    
    elif args.mode == 'word':
        if not os.path.exists(args.w2v_path):
            f = open(args.file_path,'r', encoding='cp949')
            f = [i.strip().split() for i in f.readlines()]
            model = makeFastText(f, args)
        else:
            model = w2v_load(args)

    return model

'''
jamo split return list in list type,
like this ['ㄱㅙㄴ', 'ㅊㅏㄶ', 'ㄷㅏ ']
So, we need to convert return list to FastText embedding type
to 'ㄱㅙㄴㅊㅏㄶㄷㅏ-'
'''
def convert_jamosplit_to_fasttextjamo(jamolist):
    temp = [eumjol.replace(' ','-') for eumjol in jamolist]
    return ''.join(temp)

'''
input is convert_jamosplit_to_fasttextjamo function result
output is merged char split like ['하', '지', '만']
'''
def mergejamosplit(jamos):
    return [mergeCharSplit(jamos[i*3 : (i+1)*3].replace('-',' ')) for i in range(len(jamos)//3)]

def check_most_similar(word, score, query, index):
    score1 = score > SIMILARITY_THRESHOLD
    score2 = len(word) == len(query)
    score3 = word[0] == query[0]
    score4 = len( [li for li in difflib.ndiff(word[index*3: (index+1)*3], query[index*3: (index+1)*3]) if li[0] != ' ']) == 2 # two word's diff check, if one is different len result is 2
    total = score1 + score2 + score3 + score4
    if total >= 3:
        return True
    else:
        return False

def check_most_similar_False(word, score, query, index):
    score1 = score > SIMILARITY_THRESHOLD
    score2 = word[0] == query[0]
    score3 = len( [li for li in difflib.ndiff(word[index], query[index]) if li[0] != ' ']) == 2 # two word's diff check, if one is different len result is 2
    total = socre1 + score2 + score3
    if total >= 2:
        return True
    else:
        return False

def merge(query, most, index, jamotype=True):
    if jamotype:
        return query[:index*3]+most+query[(index+1)*3:]
    else:
        return query[:index]+most+query[index+1:]

'''
most_common(1)의 결과가 1일때, 더 정확한 예측값 내기위한 함수
'''
def most_closer(query, most_list):
    temp = []
    for word, score in most_list.items():
        if len([li for li in difflib.ndiff(query, word) if li[0] != ' ']) == 2:
            temp.append(word)
    return temp[0]

def prediction(query, most_similar, index, jamotype=True):
    temp = []
    for word, score in most_similar:
        if jamotype:
            if check_most_similar(word, score, query, index):
                #temp.append(word[index*3 : (index+1)*3])
                temp.append(merge(query, word[index*3:(index+1)*3], index))
        else:
            if check_most_similar_False(word, score, query, index):
                #temp.append(word[index])
                temp.append(merge(query, word[index], index, False))

    return Counter(temp)
    #return Counter(temp).most_common(1) # True : [('ㅎㅏ-ㅈㅣ-ㅁㅏㄴ', 3)] False : [('하지만', 3)]

