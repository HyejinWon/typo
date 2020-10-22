import argparse
import os


# Todo : 음절 공백 유무에 따라서 ngram모델 만듬, 
# 이를 기반으로 ngram 생성모델 같은 느낌으로 만드는것, 위와 차이점은 ngram의 단위가 음절이라는것,
# 공백은 _로 살릴예정 아니면 그냥 잘라도 될지도.
def eumjol_ngram():



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", type=str, default='../data/KCC150_Korean_sentences_EUCKR_earse.txt')
