import argparse
import time
from tqdm import tqdm
import re
import os
import gensim
import spellCheckerW2V as sc
import spellCheckerFT as ft
import sys
sys.path.append('../')
from typo import jamoSplit

'''
jamotype : 자소분해 사용 유무
'''
def spellCheck(flines, args, jamotype = True):
    #writepath = os.path.join(args.write_path)
    w = open(args.write_path, 'w')
    model = ft.spellChecker(args)

    out_bound = conver2dic_outbound()
    
    for i in tqdm(flines):
        if OutOfBound_checker(i) != b'':
            for tindex, tok in enumerate(i.strip().split()): # 단어
                for index, eumjol in enumerate(tok): # 음절
                    if eumjol in out_bound:
                        if jamotype:
                            # 한글 인덱스보고, 한글 부분만 convert한뒤 예측 단어 추리는게 나을듯 싶은데, 
                            # return ([['ㅎ', 'ㅏ', ' '], ['ㅈ', 'ㅣ', ' '], ['ㅁ', 'ㅏ', 'ㄴ']], [0,1,2])
                            tok2jamo, hangul_index = jamoSplit(tok)
                            hangul_jamo_part = [''.join(tok2jamo[i]) for i in hangul_index] # ['ㅎㅏ ', 'ㅈㅣ ', 'ㅁㅏㄴ']
                            most_similar = model.wv.most_similar(ft.convert_jamosplit_to_fasttextjamo(hangul_jamo_part), topn=50)
                            prediction = ft.prediction(ft.convert_jamosplit_to_fasttextjamo(hangul_jamo_part), most_similar, index) #return Counter()
                        else:
                            most_similar = model.wv.most_similar(tok, topn=50)
                            prediction = ft.prediction(tok, most_similar, False)
                        
                        i_tok = i.strip().split()

                        w.write(i+'\n')
                        w.write('error eumjol : '+eumjol+'\nerror word : '+tok+'\nmost_similar : '+str(most_similar)+'\npredictioin : '+str(prediction.items())+'\n')
                        if jamotype:
                            try:
                                # Todo : 한글이랑 영어랑 중간에 섞인 경우는 어떻게 merge할것인지 생각하기
                                if hangul_index[0] != 0:
                                    other = [''.join(tok2jamo[i]) for i in range(hangul_index[0])]
                                else:
                                    other = ''
                                # 빈도 값이 1일 경우, 가장 앞에 놓은것이 출력이 되는경우 방지
                                if prediction.most_common(1)[0][1] == 1:
                                    theBest = ft.most_closer(ft.convert_jamosplit_to_fasttextjamo(hangul_jamo_part), prediction)
                                else:
                                    theBest = prediction.most_common(1)[0][0]
                                replace_tok =  ''.join(other) + ''.join(ft.mergejamosplit(theBest))
                                w.write('replace word : '+replace_tok+'\n')
                                w.write('sentence : '+' '.join(i_tok[:tindex])+ ' '+replace_tok+ ' '+' '.join(i_tok[tindex+1:])+'\n\n')
                            except:
                                #print(prediction)
                                w.write('not change\n')
                        else:
                            try:
                                replace_tok = prediction.most_common(1)[0][0]
                                w.write('replace word : '+replace_tok+'\n')
                                w.write('sentence : '+' '.join(i_tok[:tindex])+ ' '+replace_tok+ ' '+' '.join(i_tok[tindex+1:])+'\n\n')
                            except IndexError:
                                w.write('not change\n')
                        
    w.close()

def conver2dic_outbound():
    f = open('../data/outofbound_kcc.txt','r').readlines()
    #f = open('../nodeeplearning/result/outofbound_kcc.txt','r').readlines() v100서버용path
    converted = dict()

    for i in f:
        i= i.rsplit('>>>') # 븝>>>1 --> w = 븝, f = 1 #rsplit()은 뒤에서 부터 문자열 자름
        converted[i[0]] = int(i[1])

    return converted

def OutOfBound_checker(sentence):
    hexa = sentence.encode('euc-kr')
    #return re.sub(b'[\xb0-\xc8][\xa1-\xfe]', b'', hexa)
    return re.sub(b'[\xb0-\xc8][\xa1-\xfe]\s?|[a-zA-Z0-9]\s?', b'', hexa)

def main(args):
    if args.mode =='jamo':
        fline = open(args.file_path,'r', encoding='cp949').readlines()
        spellCheck(fline, args)
    elif args.mode == 'word':
        fline = open(args.file_path,'r', encoding='cp949').readlines()
        spellCheck(fline, args, False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--w2v_path', type=str, help='word2vec bin path')
    parser.add_argument('--file_path', type=str, help='file path')
    parser.add_argument('--write_path', type=str, help='result file path')
    parser.add_argument('--jamo_path', type=str, help='jamo file path')
    parser.add_argument('--mode', type=str, default='jamo', help='word / jamo')
    args = parser.parse_args()
    main(args)