# -*- coding: utf-8 -*-
# Code by Saptarashmi Bandyopadhyay

import nltk
import sys
import os
from nltk.stem import PorterStemmer

porter = PorterStemmer()

PathToDataset = sys.argv[1]
PathToOutput = sys.argv[2]

# Factoring.
for split in ["train", "val", "test"]:

    # Creating a new directory if it doesn't exist.
    if not os.path.exists(PathToOutput + split):
        os.makedirs(PathToOutput + split)

    print("Processing files in the {} set".format(split))

    files = os.listdir(PathToDataset + split)
    # total_num = len(files)

    for num, filename in enumerate(files):
        f = open(PathToDataset + split + '/' + filename, 'r')
        of = open(PathToOutput + split + '/' + filename, 'w')

        for line in f:
            if "@highlight" not in line:
                text = line.encode('utf8')
                word = nltk.pos_tag(nltk.word_tokenize(text.decode('utf8')))
                for i in word:
                    stemmed = porter.stem(i[0])
                    oword = i[0] + ' | ' + stemmed + ' | ' + i[1] + ' '
                    of.write(oword)
