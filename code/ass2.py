import sys
import math
from collections import defaultdict

import numpy as np

from utils import read_data_lines

def linston(s, freq, lam , x):
    return (freq + lam) / (s + lam *x)

def prep(corpus,smooth_f,  s,words_stat,lam, x):
    pr=0
    for w in corpus:
        pr+= np.log(smooth_f(s,words_stat[w],lam,x))
    return 2**((-1.0/len(corpus))*pr)


def main(args):
    # if (len(args)<6):
    #     print("wrong args")
    #     return
    output= [None]*30
    DEV = read_data_lines(args[1]);


    vocab = set(DEV)
    output[0] = sys.argv[1]
    output[1] = sys.argv[2]
    INPUT_WORD = sys.argv[3]
    output[2] = INPUT_WORD
    output[3] = sys.argv[4]
    lang_vocab_len = len(vocab)**2
    output[4] = lang_vocab_len
    output[5] = 1.0/lang_vocab_len
    dev_len = len(DEV)
    output[6]= dev_len
    train = DEV[0:int(math.ceil(dev_len*0.9))]
    validation = DEV[int(math.ceil(dev_len*0.9)):dev_len]
    count_T = defaultdict(int)
    count_V = defaultdict(int)
    for x in train:
        count_T[x]+=1
    for x in validation:
        count_V[x]+=1
    train_size = len(train)
    validation_size = len(validation)
    output[7] = validation_size
    output[8] = train_size
    train_voc_size =len(set(train))
    output[9] =train_voc_size
    freq_word = count_T[INPUT_WORD]
    output[10] = freq_word
    output[11]  = freq_word / len(train)
    freq_unseen = 0
    output[12]  = freq_unseen
    output[13] = linston(train_size,freq_word,0.1,train_voc_size)
    output[14] = linston(train_size,freq_unseen,0.1,train_voc_size)
    output[15] = prep(validation,linston,train_size,count_T,0.01,train_voc_size)
    output[16] = prep(validation,linston,train_size,count_T,0.10,train_voc_size)
    output[17] = prep(validation,linston,train_size,count_T,1.00,train_voc_size)
    # minimize lamda
    pr=[];
    # lam = [l/1000.0 for l in xrange(1, 1000,20)]
    # for l in lam:
    #    pr.append( prep(validation,linston,train_size,dev_words_stats,l,train_voc_size))
    # output[18]= lam[np.argmin(pr)]
    # output[19] = np.min(pr)

    #Held on
    train = DEV[0:int(math.ceil(dev_len*0.5))]
    held_on = DEV[int(math.ceil(dev_len*0.5))+1:dev_len-1]
    train_size = len(train)
    held_on_size = len(held_on)
    count_T= defaultdict(int)
    count_H = defaultdict(int)
    for x in train:
        count_T[x]+=1
    for x in held_on:
        count_H[x]+=1

    count_to_words_T=defaultdict(list)
    for x in count_T.items():
        count_to_words_T[x[1]].append(x[0])
    count_to_words_H=defaultdict(list)
    for x in count_H.items():
        count_to_words_H[x[1]].append(x[0])
    Tr = defaultdict(int)
    Nr = defaultdict(int)
    #print("train_size %s , sum : %s" ) % (train_size , sum([k*len(v) for k,v in count_to_words.items()]))
    for x in count_to_words_T.items():
        Nr[x[0]] = len(x[1])
        for w1 in x[1]:
            Tr[x[0]]+=count_H[w1]
            count_H.pop(w1) # zeoring to elemetns words that are not in class 0 in oder to count class 0 easier
    Nr[0] = len(vocab)**2
    Tr[0] = np.sum(count_H.values())
    #print("H: %s, sum: %s") %(len(held_on),sum(Tr.values()))

    Ph = {clas:1.0*cnt/(Nr[clas]*held_on_size) for  clas, cnt in Tr.items()}

    #freq_of_freqs = getfreq_of_freq(train,held_on)
    output[20] = train_size
    output[21] = held_on_size
    output[22] = Ph[count_T[INPUT_WORD]]
    output[23] = Ph[0]

    #test test
    TEST =read_data_lines(args[2]);
    test_len = len(TEST)
    output[24] = test_len

    print ("\n".join("output %s: %s " % (i+1,o) for i,o in enumerate(output)))






if __name__ == "__main__":
    main(sys.argv)

