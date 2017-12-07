# This file provides code which you may or may not find helpful.
# Use it if you want, or ignore it.
import random
from collections import defaultdict


def read_data_lines(file):
    corpus=[]
    with open(file) as f:
        for l in f:
            line = l.rstrip()
            if line and not line.startswith("<"):
                for w in line.split():
                    corpus.append(w)
    return corpus


def text_to_bigrams(text):
    return ["%s%s" % (c1,c2) for c1,c2 in zip(text,text[1:])]


