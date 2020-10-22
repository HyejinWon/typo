from tqdm import tqdm
from nltk import ConditionalFreqDist
import typo
import lower_eumjol as le

# lower_eumjol.py에서 파일 읽고서,
# 벗어나는 음절 저장해 둔거 불러와서, 파일 라인에서
# ks완성형 범위벗어나는 음절을 만났을 때,
# 음절단위 bigram과 같은 것을 해서 수정이 되는지 확인하는것
# keyboard distance로 nosie generate도 같이 진행됌

# 필요한거 : 벗어난 음절 딕셔너리, 파일
# 에러 있는 음절에 대해서 windows size 1로 저장하는것,
def error_eumjol_bi(_file, outdic):
    eumjol_list = list() # (사,꽈) 꽈가 오타인 부분임
    eumjol_list2 = list() #(꽈,자)

    # 파일에서 범위벗어나는거 음절단위 bigram으로 잡는 부분
    for i in _file:
        if le.OutOfBound(i) is None and le.check_except(i) is None:
            for index, eumjol in enumerate(i):
                if eumjol in outdic: 
                    if index == 0:
                        eumjol_list2.append((eumjol[0],eumjol[1]))
                    elif index == len(eumjol)-1: # 마지막일때
                        eumjol_list.append((eumjol[-2], eumjol[-1]))
                    else:
                        eumjol_list.append((eumjol[index-1], eumjol[index]))
                        eumjol_list2.append((eumjol[index], eumjol[index+1]))
  
    cfd = ConditionalFreqDist(eumjol_list) # take bi-grames probability
    cfd_second = ConditionalFreqDist(eumjol_list2) # take bigram probability, 위에랑 조건의 순서를 달리해서, 뒷부분이 조건이 됌.
     
    return cdf, cfd_second

# 이제 필요한거 : keyboard error make condidate 
# 이 함수는 typo의 keyboardDistance_error_onlyeumjol 함수의 결과를 머지하기위한 것
def make_candidate(error_list):
    result = []

    
    for i in error_list:
        if len(i[0]) >= 2:
            for j in i[0]:
                result.append(typo.mergeCharSplit([j,i[1], i[2]]))
        elif len(i[1]) >= 2:
            for j in i[1]:
                result.append(typo.mergeCharSplit([i[0], j, i[2]]))
        elif len(i[2]) >= 2:
            for j in i[2]:
                result.append(typo.mergeCharSplit([i[0], i[1], j]))
                    
    return result
