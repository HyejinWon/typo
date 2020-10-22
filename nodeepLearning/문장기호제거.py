import re

f = open('../data/KCC150_Korean_sentences_EUCKR.txt','r',encoding = 'cp949')
w = open('../data/KCC150_Korean_sentences_EUCKR_earse.txt','w', encoding = 'cp949')

def clean_text(sentence):
    return re.sub('[-=\+\,#/\?:^$.@*\"~&%\\\!\|\(\)\[\]\<\>`]', '',sentence)

for i in f.readlines():
    w.write(clean_text(i)+'\n')
w.close()
f.close()