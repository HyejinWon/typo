import re
import argparse
import pickle
from tqdm import tqdm
import random
import math
import os
import time
from itertools import product
import nltk
from nltk.util import ngrams
from nltk import ConditionalFreqDist
import sys
sys.path.append('..')
import typo
import eumjol_bi as eb

parser = argparse.ArgumentParser()
parser.add_argument("--file_name", type=str)
parser.add_argument("--result_filename", type=str, help='음절 ngram결과를 저장할 파일 이름')
parser.add_argument("--pickle_name", type=str, help='ngram을 저장한 pickle')
parser.add_argument("--pickle_name2", type=str, help='ngram is save in pickle by reverse')

args = parser.parse_args()

EUMJOL_DIC = dict()
OUT_DIC = dict()
ASCII_COUNT = 0

WORD_FREQ = dict()
FREQ_LIST = dict()
TOTAL_SENTENCE = 0


def read_file(file_name):
    f = open(file_name, 'r', encoding='cp949')
    return f.readlines()

def count_eumjol(sentence):
    global ASCII_COUNT
    for i in sentence:
        # 아래는 초성, 아스키 빼고 잡음
        # 원래는 check_hangul(i) is not None and 도 조건으로 넣엇으나, 이미 outbound에서 한글만 잡아서 빼버림 / 10분 단축!
        if  OutOfBound(i) is not None:
            if i in EUMJOL_DIC:
                EUMJOL_DIC[i] += 1
            else:
                EUMJOL_DIC[i] = 1
        # 아스키 빼고 초성은 들어감
        # 문장기호도 들어가는듯.
        # 원래는 OutOfBound(i) is None and 도 조건으로 넣엇으나, 위의 if문에서 이미 한번 거른애들이라서 빼버림/ 7분 단축!
        elif check_except(i) is None:
            if i == ' ':
                continue
            if i in OUT_DIC:
                OUT_DIC[i] += 1
            else:
                OUT_DIC[i] = 1
        elif check_except(i) is not None: # 아스키만 들어감 
            ASCII_COUNT += 1
        else:
            continue

# 카운터로 짜본건데 오히려 딕셔너리보다 느리다...
def count_eumjol2(sentence):
    global ASCII_COUNT

    for i in sentence:
        if OutOfBound(i) is not None:
            EUM.update(i)
        elif OutOfBound(i) is None and check_except(i) is None:
            OUT.update(i)
        elif check_except is not None:
            ASCII_COUNT +=1

def check_except(word):
    '''
    쉼표, 영어 등이 들어간 경우 return 값 가짐
    '''
    return re.match(r'[\x00-\x7F]',word)

# 위에 정규식을 바이트 단위로 다시 짜서 수정함.
def twoByteASCII(word):
    hexa = word.encode('euc-kr')
    return re.match(b'[\xa1-\xa3][\xa1-\xc8]',hexa)

def check_hangul(word):
    return re.match('[ㄱ-ㅎㅏ-ㅣ가-힣]', word)

def OutOfBound(eumjol):
    hexa = eumjol.encode('euc-kr')
    return re.match(b'[\xb0-\xc8][\xa1-\xfe]',hexa)

def OutOfBound_checker(sentence):
    hexa = sentence.encode('euc-kr')
    #return re.sub(b'[\xb0-\xc8][\xa1-\xfe]', b'', hexa)
    return re.sub(b'[\xb0-\xc8][\xa1-\xfe]\s?|[a-zA-Z0-9]\s?', b'', hexa)

def filter_outbound(error_list):
    result = []
    for i in error_list:
        if i == None:
            continue
        elif OutOfBound_checker(i) == b'':
            result.append(i)
    return result

def makePickle(dictionary, pickle_name):
    with open(pickle_name,'wb') as handle:
        pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('save dictionary to pickle...')

def openPickle(pickle_name):
    with open(pickle_name,'rb') as handle:
        dictionary_data = pickle.load(handle)
    return dictionary_data

def window_target(sentence, lower_dic):
    temp = list()
    for index, i in enumerate(sentence):
        if i in lower_dic:
            if index-3 <= 0:
                temp.append((sentence[0:index+4].strip(),i))
            else:
                temp.append((sentence[index-3:index+4].strip(), i))
        else:
            continue
    return temp

def convert2dic():
    #f = open('./result/outofbound_kcc_merge.txt','r').readlines()
    f = open('./result/outofbound_kcc.txt','r').readlines()
    converted = dict()

    for i in f:
        i= i.rsplit('>>>') # 븝>>>1 --> w = 븝, f = 1 #rsplit()은 뒤에서 부터 문자열 자름
        converted[i[0]] = int(i[1])

    return converted

def convert2dic_lower():
    f = open('./result/lower_eumjol_kcc.txt','r').readlines()
    converted = dict()

    for line, i in enumerate(f):
        i = i.split(' : ')[1]
        i = i.split('>>>')
        converted[i[0]] = int(i[1].strip())
        if line == 518: # 685
            break

    return converted

def remove_overlap(filename):
    f = open(filename,'r').readlines()
    result = [tuple(i.strip().split(' : ')) for i in f]
    result = set(result)

    return result

def setfileWrite(file, result):
    for e, w in result:
        file.write(e + ' : '+w+'\n')
    #file.write('=====================================\n')

#어절 n-gram 분석
#sentence: 분석할 문장, num_gram: n-gram 단위
def word_ngram(sentence, num_gram):
    # in the case a file is given, remove escape characters
    sentence = sentence.replace('\n', ' ').replace('\r', ' ')
    text = tuple(sentence.split(' ')) # '그','사과는','정말','맛있어'
    ngram = [text[x:x+num_gram] for x in range(0, len(text)-num_gram+1)] # [('그','사과는','정말'),('사과는','정말','맛있어')]
    make_freqlist(tuple(ngram))
    return ngram
 
#n-gram 빈도 리스트 생성
def make_freqlist(ngram):
 
    for ng in ngram:
      if (ng in FREQ_LIST):
          FREQ_LIST[ng] += 1
      else:
          FREQ_LIST[ng] = 1

    return 0

def used_nltk_bi(flines):
    sentences = []

    for i in flines:
        ngrame = ngrams(i.split(), 2 , pad_left=True, pad_right=True, left_pad_symbol="<s>", right_pad_symbol="</s>")
        sentences += [t for t in ngrame]
    
    '''
    문장을 어절단위로 bigram 분리하고, ConfitionalFreqDist를 통해 bigram의 출현 확률을 계산.
    ex) (사과, 맛잇어) 일때, cfd는 조건이 사과에서 맛잇어가 출현할 확률이고, cfd_second는 조건이 맛잇어에서 사과가 출현할 확률이다.
    '''
    cfd = ConditionalFreqDist(sentences) # take bi-grames probability

    # 기존 방식에는 이 부분 사용, 추후 필요하면 주석 해제 및 if문 추가할 것,
    """ cfd_second = ConditionalFreqDist() # take bigram probability, 위에랑 조건의 순서를 달리해서, 뒷부분이 조건이 됌.
    for i in sentences:
        condition = i[1]
        cfd_second[condition][i[0]] += 1
    """
    #print(cfd.conditions())
    #print(cfd['<s>'].most_common(3))
    #print(freq.most_common(3))

    # 기존 방식에서는 이부분 주석 해제
    # 지금은 사용안할꺼라서 주석처리해둠
    """ 
    w_file = open('./bigram_file.txt','w')
    for sentence in flines:
        detected_error_bi(sentence, cfd, cfd_second, w_file)
    w_file.close()
    """
    return cfd

def get_nltk_ngram(word, cfd):
    return cfd[word]

def unigram_freqlist(sentence):
    global WORD_FREQ
    token = sentence.strip().split()
    for i in token:
        if i in WORD_FREQ:
            WORD_FREQ[i] += 1
        else:
            WORD_FREQ[i] = 1
    
    return 0 

def calculate_bigramprobaility(sentence):
    # sentence를 받아서 n gram으로 쪼개고, 거기서 이제 빈도 딕셔너리에서 빈도 값 찾아서 확률 계산
    sentence_prob = 0
    denominator = 0

    bigrams = word_ngram(sentence, 2)
    #print(bigrams)
    for bigram in bigrams:
        '''
        if bigram not in FREQ_LIST:
            test_word = bigram[1]
            if test_word not in WORD_FREQ:
                test_word = '<UNK>'
            numerator = WORD_FREQ[test_word]
            denominator = # 음,,, 분모를 뭐로 잡아야 하지
        '''
        if '<s>' == bigram[0] or '</s>' == bigram[0]:
            numerator = FREQ_LIST[bigram]
            denominator = len(TOTAL_SENTENCE) + 1
        else:
            numerator = FREQ_LIST[bigram]
            denominator = WORD_FREQ[bigram[0]]

        bigram_prob = numerator / float(denominator)
        sentence_prob += math.log(bigram_prob)
    
    return math.exp(sentence_prob)

def detected_error_bi(fline, cfd, cfd_second, w):
    line_split = fline.split()
    #print(line_split)
    for index, j in enumerate(line_split):
        for i in j:
            if i in OUT_DIC:
                # 음 디택되면 해당 문장의 확률을 계산하고, 해당 부분의 확률이 제일 높은 걸 추천
                # Todo : 가장 빈도 높은거를 뽑으라 했는데, 고빈도가 여러개일 가능성도 있음. 따라서 loop 돌려서 여러개 뽑도록 해야함.
                
                #probaility = calculate_bigramprobaility(fline) # 굳이 필요없긴함
                if index is not 0 and index+1 is not len(line_split):
                    max_result = find_maxmun_bi(line_split[index-1], cfd) 
                    max_result2 = find_maxmun_bi(line_split[index+1], cfd_second) #문장의 제일 마지막에 위치할 경우를 처리하자
                    #print(max_result,max_result2)

                    #print(max_result2,max_result2[0][0])
                    #print(fline,detected_error, i+'\n',str(max_result)+'--->'+' '.join(line_split[:index])+' '+max_result[0][0]+' '+' '.join(line_split[index+1:])+'\n')
                    #print(fline,detected_error, i+'\n',str(max_result)+'--->'+' '.join(line_split[:index])+' '+max_result2[0][0]+' '+' '.join(line_split[index+1:])+'\n')                    
                    #w.write('\n'+fline+'\n'+'detected error is : '+ i+'\n'+'probaility is : '+str(probaility)+'\n')
                    
                    w.write('\n'+fline+'\n'+'detected error is : '+ i+'\n')
                    w.write('recommand change : '+ str(max_result)+'\n'+'-->'+' '.join(line_split[:index])+' '+max_result[0][0]+' '+' '.join(line_split[index+1:])+'\n')
                    w.write('recommand change : '+ str(max_result2)+'\n'+'-->'+' '.join(line_split[:index])+' '+max_result2[0][0]+' '+' '.join(line_split[index+1:])+'\n')

                elif index+1 is len(line_split):
                    max_result = find_maxmun_bi(line_split[index-1], cfd) 
                    #print('The last one : ',max_result)
                    #print(fline,detected_error, i+'\n',str(max_result)+'--->'+' '.join(line_split[:index])+' '+max_result[0][0]+' '+' '.join(line_split[index+1:])+'\n')                    
                    w.write('\n'+fline+'\n'+'detected error is : '+ i+'\n')
                    w.write('The last token recommand change: '+str(max_result)+'\n'+'-->'+' '.join(line_split[:index])+' '+max_result[0][0]+' '+' '.join(line_split[index+1:])+'\n')

                else:
                    max_result = find_maxmun_bi(line_split[index+1], cfd_second) 
                    #print('The first one : ',max_result)
                    #print(fline,detected_error, i+'\n',str(max_result)+'--->'+' '.join(line_split[:index])+' '+max_result[0][0]+' '+' '.join(line_split[index+1:])+'\n')
                    w.write('\n'+fline+'\n'+'detected error is : '+ i+'\n')
                    w.write('The first token recommand change: '+str(max_result)+'\n'+'-->'+' '.join(line_split[:index])+' '+max_result[0][0]+' '+' '.join(line_split[index+1:])+'\n')


def find_maxmun_bi(word, cfd):
    return cfd[word].most_common(1)

def used_nltk_tri(flines):
    sentences = []

    for i in flines:
        trigram = ngrams(i.split(), 3, pad_left=True, pad_right=True, left_pad_symbol="<s>", right_pad_symbol="</s>")
        sentences += [t for t in trigram]
    
    '''
    문장, '어 저 사과는 정말 맛잇어' 에서 '사과는'을 에러라 가정하고
     (어, 저) : 사과는 , (정말, 맛잇어) : 사과는 으로 딕셔너리가 구성되어있음.
    '''
    cfd = ConditionalFreqDist()
    cfd_second = ConditionalFreqDist()

    for i in sentences:
        condition = (i[0], i[1])
        cfd[condition][i[2]] += 1

        condition2 = (i[1], i[2])
        cfd_second[condition2][i[0]] += 1

    w_file = open('./trigram_file.txt','w')
    for sentence in flines:
        detected_error_tri(sentence, cfd, cfd_second, w_file)
    w_file.close()

def detected_error_tri(fline, cfd, cfd_second, w):
    line_split = fline.split()

    for index, j in enumerate(line_split):
        for i in j:
            if i in OUT_DIC:
                # Todo : 가장 빈도 높은거를 뽑으라 했는데, 고빈도가 여러개일 가능성도 있음. 따라서 loop 돌려서 여러개 뽑도록 해야함.
                try:
                    if index > 1 and index < len(line_split)-2: #len(line_split)-2 인 이유는 len은 1부터 세니까 

                        max_result = find_maxmun_tri((line_split[index-2], line_split[index-1]), cfd) 
                        max_result2 = find_maxmun_tri((line_split[index+1], line_split[index+2]), cfd_second) 

                        w.write('\n'+fline+'\n'+'detected error is : '+ i+'\n')
                        w.write('recommand change : '+ str(max_result)+'\n'+'-->'+' '.join(line_split[:index])+' '+max_result[0][0]+' '+' '.join(line_split[index+1:])+'\n')
                        w.write('recommand change : '+ str(max_result2)+'\n'+'-->'+' '.join(line_split[:index])+' '+max_result2[0][0]+' '+' '.join(line_split[index+1:])+'\n')
                    
                    elif index >= len(line_split) -2:
                        max_result = find_maxmun_tri((line_split[index-2], line_split[index-1]), cfd) 
                        w.write('\n'+fline+'\n'+'detected error is : '+ i+'\n')
                        w.write('The last token recommand change: '+str(max_result)+'\n'+'-->'+' '.join(line_split[:index])+' '+max_result[0][0]+' '+' '.join(line_split[index+1:])+'\n')

                    else:
                        max_result = find_maxmun_tri((line_split[index+1], line_split[index+2]), cfd_second) 
                        w.write('\n'+fline+'\n'+'detected error is : '+ i+'\n')
                        w.write('The first token recommand change: '+str(max_result)+'\n'+'-->'+' '.join(line_split[:index])+' '+max_result[0][0]+' '+' '.join(line_split[index+1:])+'\n')
                except:
                    print('line = ', fline)
                    print('error = ', i, 'index number : ', index)
                    
def find_maxmun_tri(tu, cfd):
    return cfd[tu].most_common(1)

# eumjol_bi 파일에서 구현된 함수 불러와서 오타리스트 생성
def get_keyboard_error(eumjol):
   
    wordlist, _ = typo.jamoSplit(eumjol) #[['ㅅ', 'ㅠ', 'ㄴ']] 으로 결과출력
    try:
        error_list = typo.keyboardDistance_error_onlyeumjol(wordlist[0]) # put detected error eumjol and make noise
    except: # 주변단어에서 영어나 숫자가 들어간 경우 자소 분해가 되어도 wordlist를 만들어지지가 않기 떄문에 그냥 그 값자체를 리턴시킴
        return wordlist[0]
    return eb.make_candidate(error_list) # error candidate list

def put_keyboard_error(word, word_index, error_list, AtomType=True):
    result = []
    if AtomType:
        for i in error_list:
            try:
                result.append(word[:word_index]+i+word[word_index+1:])
            except TypeError:
                result.append(None)
        return result
    else: # findNoneAndRevision()을 위한것, windows size 1로 생성된 오타 후보들과, 기존의 단어부분을 합침
        for i in error_list:
            try:
                if word_index -1 <= 0:
                    result.append(''.join(i)+word[word_index+2:])
                else:
                    result.append(word[:word_index-1]+''.join(i)+word[word_index+2:])
            except TypeError:
                result.append(None)
        return result

def put_eumjol_keyboard_error(error_list, windows):
    result = []
    for i in error_list:
        result.append((windows[0][:-1] + i, i + windows[1][1:])) 
        
    return result

def find_list_to_word(wordlist, word, AtomType=False):
    # bigram 리스트에서 가능한 후보들 중에서 기존 단어와 음절 및 자소가 같은것을 기준으로
    # 후보 단어 리스트를 내보내줌
    # True는 음절 기준, Fasle는 자소 기준
    if AtomType:
        """ for oneword in wordlist:
            if len(oneword) != len(word):
                continue
            else:
                 """
        pass
    else:
        word_jamo, _ = typo.jamoSplit(word)
        temp = []
        for oneword, value in wordlist.items():
            oneword_jamo, _ = typo.jamoSplit(oneword)
            # 자소분리된 결과의 개수가 같고, 첫번째 글자가 같은 경우에만 진행
            if len(oneword_jamo) == len(word_jamo) and oneword_jamo[0][0] == word_jamo[0][0]:
                count = [jamo for o, w in zip(oneword_jamo, word_jamo) for jamo in o if jamo not in w]
                if len(count) <= 2:
                    temp.append((oneword, value))
            else:
                continue
    return temp

def get_list_to_word(wordlist):
    temp = (None, 0)
    for k, v in wordlist:
        if v > temp[1]:
            temp = (k, v)
    return temp

# Unigram frequency 함수가 실행되어야함.
def keyboard_error_probability(error_word_list):
    temp = dict()
    for i in error_word_list:
        try:
            value = WORD_FREQ[i]
            temp[i] = value
        except KeyError:
            continue

    '''
    def f1(x):
        return temp[x]
    '''
    #print(error_word_list,temp)
    try:
        key_name = max(temp, key = temp.get)    
        
    except:
        key_name = None

    return key_name

def findNoneAndRevision(token, index, lowDic, AtomType = False):
    # max_key값이 없을 때, None의 결과를 받은 것을 기준으로 keyNone.txt
    # max_key값이 없을 때, 음절빈도가 낮은것을 기준으로 keyLow.txt
    # windows 1로 잡고, 주변 음절의 빈도가 낮은것이 들어가면 오타로 생각하고
    # 해당 음절의 주변 음절을 수정하는 방법
    temp = list()
    if AtomType: # 음절빈도가 낮은것을 기준으로 keyLow.txt
        if index >0:
            for i in token[index-1 : index+2]:
                if i in lowDic or i == token[index]:
                    temp.append(get_keyboard_error(i))
                else:
                    temp.append(i)
        else:
            for i in token[index:index+2]:
                if i in lowDic or i == token[index]:
                    temp.append(get_keyboard_error(i))
                else:
                    temp.append(i)
    else: # None의 결과를 받은 것을 기준으로 keyNone.txt
        if index > 0:
            for i in token[index-1:index+2]:
                temp.append(get_keyboard_error(i))
        else:
            for i in token[index:index+2]:
                temp.append(get_keyboard_error(i))

    all_case = list(product(*temp)) # 데카르트 곱집합구하기 list에 튜플로 값이 들어가있음
    all_case = put_keyboard_error(token, index, all_case, False)
    max_key = keyboard_error_probability(all_case)
    if max_key == token:
        return None
    #print(temp, all_case, max_key)
    else:
        return max_key, all_case

def find_before_candidate_max(before_token, all_case, cfd):
    temp = tuple()
    for i in all_case:
        try:
            result = max(temp[1], cfd[before_token][i]) 
            if result == temp[1]:
                continue
            else:
                temp = (i, cfd[before_token][i])
        except IndexError:
            temp = (i, cfd[before_token][i])

    return temp

def findErrorAndRevision_HandleNone_bigram_front(fline, lowDic, cfd):
    w7 = open('./result/keyLow_bigram_jamo_front.txt', 'w')

    count = 0
    for i in tqdm(fline):
        if OutOfBound_checker(i) != b'':
            for tindex, tok in enumerate(i.strip().split()):
                for index, eumjol in enumerate(tok):
                    if eumjol in OUT_DIC:
                        error_list = get_keyboard_error(eumjol)
                        error_word_list = put_keyboard_error(tok, index, error_list)
                        max_key = keyboard_error_probability(error_word_list)
                        
                        if max_key:
                            continue
                            
                        else:
                            i_tok = i.strip().split()
                            #revision_list = get_nltk_ngram(i_tok[tindex-1], cfd)
                            
                            w7.write('sentence : '+i.strip()+'\n'+'error eumjol : '+eumjol+'\n')
                            revision = find_before_candidate_max(i_tok[tindex-1], error_word_list, cfd)

                            if revision[1] != 0:
                                
                                w7.write('all result : '+str(error_word_list)+'\n')
                                w7.write('max result : '+str(revision)+'\n') 
                                w7.write('None to sentence : '+' '.join(i_tok[:tindex])+ ' '+str(revision[0])+ ' '+' '.join(i_tok[tindex+1:])+'\n\n')
                                count += 1
                            else:
                                w7.write('max result : '+str(None)+'\n')
                                w7.write('can not find alther : '+i.strip()+'\n\n')
                                
                                
    w7.close()
    print('how many None is changed',count)
    return 0


def findErrorAndRevision_HandleNone_bigram(fline, lowDic, cfd):
    w7 = open('./result/keyLow_bigram_jamo_True.txt', 'w')

    count = 0
    for i in tqdm(fline):
        if OutOfBound_checker(i) != b'':
            for tindex, tok in enumerate(i.strip().split()):
                for index, eumjol in enumerate(tok):
                    if eumjol in OUT_DIC:
                        error_list = get_keyboard_error(eumjol)
                        error_word_list = put_keyboard_error(tok, index, error_list)
                        max_key = keyboard_error_probability(error_word_list)
                        
                        

                        if max_key:
                            continue
                            """ i_tok = i.strip().split()
                            w7.write('error candidate : '+str(error_word_list)+'\n')
                            w7.write('max result : '+str(max_key)+'\n')
                            w7.write('revision sentence : '+' '.join(i_tok[:tindex])+ ' '+max_key+ ' '+' '.join(i_tok[tindex+1:])+'\n\n') """
                        else:
                            i_tok = i.strip().split()
                            revision_list = get_nltk_ngram(i_tok[tindex-1], cfd)
                            
                            candidate_list = find_list_to_word(revision_list, tok)
                            revision = get_list_to_word(candidate_list)

                            if revision[0] != None and tok != revision[0]:
                                w7.write('sentence : '+i.strip()+'\n'+'error eumjol : '+eumjol+'\n')
                                #w7.write('all result : '+str(revision_list.items())+'\n')
                                w7.write('candidate list : ' + str(candidate_list)+'\n')
                                w7.write('max result : '+str(revision)+'\n') 
                                w7.write('None to sentence : '+' '.join(i_tok[:tindex])+ ' '+str(revision[0])+ ' '+' '.join(i_tok[tindex+1:])+'\n\n')
                                
                            else:
                                #w7.write('max result : '+str(None)+'\n')
                                #w7.write('can not find alther : '+i.strip()+'\n\n')
                                
                                count += 1
    w7.close()
    print('how many None is changed',count)
    return 0


def findErrorAndRevision_HandleNone_temp(fline, lowDic):
    w7 = open('./result/keyboard.txt', 'w')
    for i in tqdm(fline):
        if OutOfBound_checker(i) != b'':
            for tindex, tok in enumerate(i.strip().split()):
                for index, eumjol in enumerate(tok):
                    if eumjol in OUT_DIC:
                        error_list = get_keyboard_error(eumjol)
                        error_word_list = put_keyboard_error(tok, index, error_list)
                        max_key = keyboard_error_probability(error_word_list)

                        if max_key:
                        
                            i_tok = i.strip().split()
                            w7.write('sentence : '+i.strip()+'\n'+'error eumjol : '+eumjol+'\n')
                            w7.write('error candidate : '+str(error_word_list)+'\n')
                            w7.write('max result : '+str(max_key)+'\n')
                            w7.write('revision sentence : '+' '.join(i_tok[:tindex])+ ' '+max_key+ ' '+' '.join(i_tok[tindex+1:])+'\n\n')
                        

    w7.close()

    return 0

def findErrorAndRevision_HandleNone(fline, lowDic):
    w7 = open('./result/keyboard.txt', 'w')
    #w8 = open('./result/keyLow_None.txt','w')
    count = 0
    for i in tqdm(fline):
        if OutOfBound_checker(i) != b'':
            for tindex, tok in enumerate(i.strip().split()):
                for index, eumjol in enumerate(tok):
                    if eumjol in OUT_DIC:
                        error_list = get_keyboard_error(eumjol)
                        error_word_list = put_keyboard_error(tok, index, error_list)
                        max_key = keyboard_error_probability(error_word_list)

                        #w7.write('sentence : '+i.strip()+'\n'+'error eumjol : '+eumjol+'\n')
                        
                        if max_key:
                        
                            i_tok = i.strip().split()
                            w7.write('error candidate : '+str(error_word_list)+'\n')
                            w7.write('max result : '+str(max_key)+'\n')
                            w7.write('revision sentence : '+' '.join(i_tok[:tindex])+ ' '+max_key+ ' '+' '.join(i_tok[tindex+1:])+'\n\n')
                        
                        else:
                            #continue
                            # True는 음절빈도가 낮은거 기준으로, False는 기준없이
                            
                            revision, all_candidate_case = findNoneAndRevision(tok, index, lowDic, False)
                            if revision:
                                i_tok = i.strip().split()
                                w7.write('sentence : '+i.strip()+'\n'+'error eumjol : '+eumjol+'\n')
                                w7.write('error candidate : '+str(all_candidate_case)+'\n')
                                w7.write('max result : '+str(revision)+'\n')
                                w7.write('None to sentence : '+' '.join(i_tok[:tindex])+ ' '+revision+ ' '+' '.join(i_tok[tindex+1:])+'\n\n')
                            
                                """ None에 해당하는 값만 따로 취함 
                                w8.write('sentence : '+i.strip()+'\n'+'error eumjol : '+eumjol+'\n')
                                w8.write('error candidate : '+str(all_candidate_case)+'\n')
                                w8.write('max result : '+str(revision)+'\n')
                                w8.write('None to sentence : '+' '.join(i_tok[:tindex])+ ' '+revision+ ' '+' '.join(i_tok[tindex+1:])+'\n\n') """
                        
                            #else:
                                """ w7.write('max result : '+str(None)+'\n')
                                w7.write('can not find alther : '+i.strip()+'\n\n') """
                                """ w8.write('sentence : '+i.strip()+'\n'+'error eumjol : '+eumjol+'\n')
                                w8.write('max result : '+str(None)+'\n')
                                w8.write('can not find alther : '+i.strip()+'\n\n') """
    w7.close()
    #w8.close()
    print('how many None is changed',count)
    return 0

def findErrorAndRevision(fline):
    w7 = open('./result/keyboard_error_result_updown.txt', 'w')
    count1 = 0
    count2 = 0
    for i in tqdm(fline):
        if OutOfBound_checker(i) != b'':
            for tindex, tok in enumerate(i.strip().split()):
                for index, eumjol in enumerate(tok):
                    if eumjol in OUT_DIC:
                        error_list = get_keyboard_error(eumjol)
                        error_word_list = put_keyboard_error(tok, index, error_list)
                        max_key = keyboard_error_probability(error_word_list)

                        w7.write('sentence : '+i.strip()+'\n'+'error eumjol : '+eumjol+'\n')
                        w7.write('error candidate : '+str(error_word_list)+'\n'+'max result : '+str(max_key)+'\n')

                        if max_key:
                            i_tok = i.strip().split()
                            w7.write('revision sentence : '+' '.join(i_tok[:tindex])+ ' '+max_key+ ' '+' '.join(i_tok[tindex+1:])+'\n\n')
                            count1 += 1

                        else:
                            w7.write('can not find alther : '+i.strip()+'\n\n')
                            count2 += 1
    w7.close()
    print('revision sentence : {}, not change sentnece : {}'.format(count1, count2))

    return 0

# Todo : 음절 공백 유무에 따라서 ngram모델 만듬, 
# 이를 기반으로 ngram 생성모델 같은 느낌으로 만드는것, 위와 차이점은 ngram의 단위가 음절이라는것,
# 공백은 _로 살릴예정 아니면 그냥 잘라도 될지도.
# 키보드 에러넣어서 후보군 생성한뒤 가장 높은 빈도를 차지하는 것 을 출력했음
# 키보드 에러 안넣은것도 결과 한번 뽑아보기
def eumjol_ngram(fline, ERROR=True):
    ngram_list = [2,3,4]
    ngram_result = []
    w = open(args.result_filename,'w')

    for _ngram_list in ngram_list:
        if not os.path.exists(str(_ngram_list)+args.pickle_name):
            ndict1, ndict2 = eumjol_ngram_nltk(_ngram_list, fline)
        else:
            ndict1 = openPickle(str(_ngram_list)+args.pickle_name) # --pikcle_name='eumjol_ngram.pkl
            ndict2 = openPickle(str(_ngram_list)+args.pickle_name2) # --pickle_name2='eumjol_ngram_rev.pkl
        
        w.write('------------'+str(_ngram_list)+'------------\n')
        temp = [0,0]
        for sentence in fline:
            if OutOfBound_checker(sentence) != b'':
                #errordicted(i.replace(' ','_'), _ngram_list, ndict1, ndict2)
                for index, i in enumerate(sentence.replace(' ','_')):
                    if i in OUT_DIC: # Todo : if i - ngram < 0 Should handle
                        
                        windows = (sentence[index-_ngram_list+1 : index+1].replace(' ','_'), sentence[index : index+_ngram_list].replace(' ','_'))
                        if ERROR:
                            error_list = get_keyboard_error(i)
                            error_list = list(set(filter_outbound(error_list)))
                            windows_append_error = put_eumjol_keyboard_error(error_list, windows)
                            reverse, forward = find_fronteumjol_candidate_max(windows_append_error, _ngram_list, ndict1, ndict2)
                            w.write('sentence :'+ sentence+'\nerror eumjol : '+i+'\nerror_list : '+str(error_list)+'\n')
                            try:

                                if forward[1] != 0:
                                    w.write('forward :'+sentence[:index+(-_ngram_list+2)]+''.join(forward[0]).replace('_',' ')+sentence[index+1:]+'\n')
                                    temp[0] += 1
                                else:
                                    w.write('not change\n')
                            except IndexError:
                                w.write('not change\n')

                            try:
                                if reverse[1] != 0:
                                    w.write('reverse :'+sentence[:index]+''.join(reverse[0]).replace('_',' ')+sentence[index+_ngram_list-1:]+'\n\n')
                                    temp[1] += 1
                                else:
                                    w.write('not change\n\n')
                            except IndexError:
                                w.write('not change\n\n')


                        else:
                            if _ngram_list == 2:
                                reverse = find_maxmun_bi(sentence[index+1], ndict2)
                                forward = find_maxmun_bi(sentence[index-1], ndict1)
                                
                            elif _ngram_list == 3:
                                reverse = find_maxmun_tri(sentence[index+1: index+3], ndict2)
                                forward = find_maxmun_tri(sentence[index-2:index], ndict1)
                                
                            else:
                                reverse = find_maxmun_tri(sentence[index +1 : index+4], ndict2)
                                forward = find_maxmun_tri(sentence[index-3:index], ndict1)
                            w.write('sentence :'+ sentence+'\nerror eumjol : '+i+'\n')
                            if len(forward) == 0 :
                                w.write('not change\n')
                            else:
                                w.write('forward :'+sentence[:index-1]+str(forward[0][0])+sentence[index+1:]+'\n')
                            if len(reverse) == 0:
                                w.write('not change\n')
                            else:
                                w.write('reverse :'+sentence[:index]+str(reverse[0][0])+sentence[index+1:]+'\n\n')

                        

        ngram_result.append(temp)

    w.close()
    print('-----bigram-----')
    print('forward : {}'.format(ngram_result[0][0]))
    print('reverse : {}'.format(ngram_result[0][1]))
    print('-----trigram-----')
    print('forward : {}'.format(ngram_result[1][0]))
    print('reverse : {}'.format(ngram_result[1][1]))
    print('-----fourgram-----')
    print('forward : {}'.format(ngram_result[2][0]))
    print('reverse : {}'.format(ngram_result[2][1]))
    
    return 0
'''
def errordicted(sentence, ngram, ndict1, ndict2):
# 에러 있는 문장을 찾았으면, 어디가 에러인지 확인하고, 교정
    for index, i in enumerate(sentence):
        if i in OUT_DIC: # Todo : if i - ngram < 0 Should handle
            
            windows = (sentence[index-ngram+1 : index+1], sentence[index : index+ngram])
            error_list = get_keyboard_error(i)
            error_list = filter_outbound(error_list)
            windows_append_error = put_eumjol_keyboard_error(error_list, windows)
            revers, forward = find_fronteumjol_candidate_max(windows_append_error, ngram, ndict1, ndict2)
            # error eumjol, error eumjol index, windows error list, revers max result, forward max result
            return i, index, windows_append_error, revers, forward
        else:
            continue
    return 0
'''

def find_fronteumjol_candidate_max(all_case, ngram, cfd1, cfd2):
    temp = tuple()
    temp2 = tuple()
    for i, j in all_case:
        if ngram == 2:
            condition2 = i[-1]
            condition1 = j[0]
        elif ngram == 3:
            if len(i) < 2 and len(j) < 2:
                condition1 = ()
                condition2 = ()
            elif len(i) < 2:
                condition2 = ()
                condition1 = (j[0], j[1])
            elif len(j) < 2:
                condition1 = ()
                condition2 = (i[1], i[2])
            else:
                condition2 = (i[1], i[2])
                condition1 = (j[0], j[1])

        else:
            if len(i) < 3 and len(j) < 3:
                condition1 = ()
                condition2 = ()
            elif len(i) < 3:
                condition2 = ()
                condition1 = (j[0], j[1], j[2])
            elif len(j) < 3:
                condition1 = ()
                condition2 = (i[1], i[2], i[3])
            else:
                condition2 = (i[1], i[2], i[3])
                condition1 = (j[0], j[1], j[2])

        # forward
        try:   
            result = max(temp[1], cfd1[condition1][j[-1]]) 
            
            if result != temp[1]:
                temp = (condition1, cfd1[condition1][j[-1]])
            
        except IndexError:
            temp = (condition1, cfd1[condition1][j[-1]]) 

        # reverse
        try:
            result2 = max(temp2[1], cfd2[condition2][i[0]])
            if result2 != temp2[1]:
                temp2 = (condition2, cfd2[condition2][i[0]])
        
        except IndexError:
             temp2 = (condition2, cfd2[condition2][i[0]])

    return temp, temp2

def eumjol_ngram_nltk(ngram, fline):
    sentences = []

    for i in fline:
        ngrame = ngrams(i.replace(' ','_'), ngram) # 공백 _로 변환
        sentences += [t for t in ngrame]
    
    cfd = ConditionalFreqDist()
    cfd_second = ConditionalFreqDist()

    if ngram == 2:
        cfd = ConditionalFreqDist(sentences)
        for i in sentences:
            condition = i[1]
            cfd_second[condition][i[0]] += 1

    elif ngram == 3:
        for i in sentences:
            condition = (i[0], i[1])
            cfd[condition][i[2]] += 1

            condition2 = (i[1], i[2])
            cfd_second[condition2][i[0]] += 1
    
    elif ngram == 4:
        for i in sentences:
            condition = (i[0], i[1], i[2])
            cfd[condition][i[3]] += 1

            condition2 = (i[1], i[2], i[3])
            cfd_second[condition2][i[0]] += 1

    makePickle(cfd, str(ngram)+args.pickle_name)
    makePickle(cfd_second, str(ngram)+args.pickle_name2)
    return cfd, cfd_second

if __name__ == "__main__":
    
    start = time.time()

    fline = read_file(args.file_name) #euc-kr file
    TOTAL_SENTENCE = len(fline)
    
    
    '''
    print('counting eumjol...')
    for i in tqdm(fline):
        count_eumjol(i)
        # 하단의 피클에 저장함.
        unigram_freqlist(i) 



    '''
    #makePickle(WORD_FREQ)
    #WORD_FREQ = openPickle(args.pickle_name) #--pickle_name='./word_unigram.pkl'
    
    OUT_DIC = convert2dic()
    
    print('total outofbound eumjol is ', len(OUT_DIC))
    
    eumjol_ngram(fline, False)
    '''
    sorted_dic = sorted(OUT_DIC.items(), key = (lambda x:x[1]), reverse=True)

    print('out bound ks완성형',len(sorted_dic))
    #w = open('./result/outofbound_kcc_merge.txt','w')
    w = open('./result/outofbound_kcc.txt','w')
    for k, v in sorted_dic:
        w.write(str(k)+'>>>'+str(v)+'\n')
    w.close()
    '''
    '''
    sorted_eumjol = sorted(EUMJOL_DIC.items(), key=(lambda x: x[1]))

    lower_10 = len(sorted_eumjol)*0.5
    lower_eumjol = dict(sorted_eumjol[:int(lower_10)])
    
    
    makePickle(lower_eumjol)

    print('total eumjol length : ',len(sorted_eumjol))

    w2 = open('./result/lower_eumjol_kcc_merge.txt','w')
    for i, (k, v) in enumerate(sorted_eumjol):
        if i == int(lower_10):
            break
        w2.write(str(i)+' : '+str(k)+'>>>'+str(v)+'\n')
    w2.close()
    #print(OUT_DIC)

    

    # 이제 로드된 딕셔너리를 기준으로 어절빈도 낮은거의 주변 단어 뽑기
    #load_dic = openPickle()
    #print('load dictionary from pickle...')
    #print('total dictionary count is : ', len(load_dic))
    
    load_dic = convert2dic_lower()

    w3 = open('./result/lower_window3Word_merge.txt','w')
    for i in tqdm(fline):
        result_list = window_target(i, load_dic)
        for window_word, eumjol in result_list:
            w3.write(eumjol + ' : '+window_word+'\n')
    w3.close()

    # euc-kr 범위 밖에 있는 음절에도 주변 단어 뽑기
    w4 = open('./result/outof_window3Word_merge.txt','w')
    convedDic = convert2dic()
    for i in tqdm(fline):
        result_list = window_target(i, convedDic)
        for window_word, eumjol in result_list:
            w4.write(eumjol + ' : '+window_word+'\n')
    w4.close()
    
    
    #중복제거한거도 카운팅하기
    removed_result = remove_overlap('./result/outof_window3Word_merge.txt')
    print('outbound remove overlap count : ', len(removed_result)+1)

    #print(random.sample(removed_result, 1))

    removed_result_low = remove_overlap('./result/lower_window3Word_merge.txt')
    print('lower eumjol remove overlap count : ', len(removed_result_low)+1)
    
    '''
    '''
    # 돌릴 필요 없음
    #어차피 겹치는 부분이 없어서 위와 같은 결과나옴
    #교집합, 차집합 구하기
    w5 = open('./result/intersection_result.txt','w')
    inter_result = removed_result & removed_result_low
    print('intersection_result count : ', len(inter_result))
    setfileWrite(w5, inter_result)

    w6 = open('./result/difference_result.txt','w')
    outbound_diff = removed_result- removed_result_low
    lower_diff = removed_result_low - removed_result
    print('out bound diff count : ', len(outbound_diff)+1)
    print('lower eumjol diff count : ', len(lower_diff)+1)
    
    w6.write('-----------out bound diff--------------\n')
    setfileWrite(w6, outbound_diff)
    w6.write('=====================================\n')
    w6.write('-----------lower diff----------------\n')
    setfileWrite(w6, lower_diff)
    w6.close()
    '''

    # outofbound euc-kr의 결과를 기준으로 기존 kcc150의 에러 탐지

    '''
    # 일단은 전체 파일을 기준으로 어절 n-gram생성
    for i in fline:
        ngrams = word_ngram(i, 3)

    #print(FREQ_LIST) #제대루 나옴
    print('unique tri gram count : ',len(FREQ_LIST)+1)

    
    # 위에서 호출할 경우 삭제해도됌!
    convedDic = convert2dic()

    for i in fline:
        detected_error(i)
    '''
    '''
    # bigram 용
    print('make bigram...')
    bigram = used_nltk_bi(fline)
    '''
    '''
    # trigram 용
    used_nltk_tri(fline) #  ('I', 'am', 'going') am converting it to (('I', 'am'), 'going')
    '''
    '''
    print('counting unigram word')
    for i in tqdm(fline):
        unigram_freqlist(i)

    OUT_DIC = convert2dic()
    '''

   
    
    #print('find out keyboard error and choose max one...')
    #findErrorAndRevision(fline)

    #lower_dic = convert2dic_lower()
    #findErrorAndRevision_HandleNone(fline, lower_dic)
    #findErrorAndRevision_HandleNone_temp(fline, lower_dic)

    # 밑에 코드는 위에 used_nltk_bi 함수 풀어줘야함
    #findErrorAndRevision_HandleNone_bigram(fline, lower_dic, bigram)
    #findErrorAndRevision_HandleNone_bigram_front(fline, lower_dic, bigram)

    print('time : %f'%(time.time()-start))
