# Implement the naïve Bayes algorithm for text classification tasks. The version
# of naïve Bayes that you will implement is called the multinomial naïve Bayes
# (MNB). The details of this algorithm can be read from chapter 13 of the book
# "Introduction to Information Retrieval" by Manning et al. This chapter can be
# downloaded from: http://nlp.stanford.edu/IR-book/pdf/13bayes.pdf

# Read the introduction and sections 13.1 and 13.2 carefully. The MNB model is
# presented in Figure 13.2 Note that the algorithm uses add-one Laplace
# smoothing. Make sure that you do all the calculations in log-scale to avoid
# underflow as indicated in equation 13.4. To test your algorithm, you will use
# the 20 newsgroups dataset, which is available for download from here:
# http://qwone.com/~jason/20Newsgroups/

# You will use the "20 Newsgroups sorted by date" version. The direct link for
# this dataset is: http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz
# This dataset contains folders for training and test portions, with a sub-
# folder for different classes in each portion. For example, in the train
# portion, there is a sub-folder for computer graphics class, titled
# "comp.graphics" and a similar sub-folder exists in the test portion. To
# simplify storage and memory requirements, you can select any 5 classes out of
# these 20 to use for training and test portions. As always, you have to train
# your algorithm using the training portion and test it using the test portion.

import sys
import os
import math
import nltk
from nltk.corpus import stopwords
from collections import Counter

def dataCleanup(data):
    '''
    tokenize data and remove stopwords and numeric data
    :param data:
    :return:
    '''


    # push stopwords to a list
    stop = stopwords.words('english')
    # tokenize the documents
    words = nltk.word_tokenize(data.lower(), language='english')
    # data cleanup: remove stopwords, numbers, special chars
    cleaneddata = [word for word in words if word not in stop and not word.isdigit() and not word.isdecimal()
                           and not word.isnumeric() and word.isalpha()]
    return cleaneddata

def getFileContents(files_list):
    '''
    read contents from the set of file, append and return
    :param files_list: 
    :return: 
    '''

    # read all the train data files
    content = ""
    for f in files_list:
        with open(f, 'r') as content_file:
            for line in content_file:
                if 'Lines:' in line:
                    for line in content_file:
                        content = content + " " + line
    return content

def getAllVocabulary(traindata):
    '''
    extract vocabulary from the training data
    :param traindata:
    :return:
    '''

    # get all sub-directory names
    directory_list = list()
    for root, dirs, files in os.walk(traindata, topdown=False):
        for name in dirs:
            directory_list.append(name)
    class_name = directory_list

    # collect all files in train data sub dirs and each class label count
    files_list = list()
    eachclassdoccount = {}
    for class_file in class_name:
        for root, dirs, files in os.walk(os.path.join(traindata, class_file), topdown=False):
            count = 0
            for docname in files:
                count = count + 1
                files_list.append(os.path.join(traindata, class_file, docname))
            eachclassdoccount[class_file] = count

    print("The number of class labels present in training data", len(class_name))
    totaldoccount = len(files_list)
    content = getFileContents(files_list)
    tokenizeddata = dataCleanup(content)
    return totaldoccount, tokenizeddata, class_name, eachclassdoccount

def getConcatenatedText(fullpath):
    '''
    :param fullpath:
    :return:
    '''
    files_list = list()
    for root, dirs, files in os.walk(fullpath, topdown = False):
        for name in files:
            files_list.append(os.path.join(fullpath, name))
    content = getFileContents(files_list)
    return dataCleanup(content)


def getMatchedTokens(vocabulary, filepath):
    '''
    return tokens of test data matched with vocabulary
    :param vocabulary:
    :param filepath:
    :return:
    '''
    content = ""
    with open(filepath, 'r') as content_file:
        for line in content_file:
            if 'Lines:' in line:
                for line in content_file:
                    content = content + " " + line
    contentlist = dataCleanup(content)
    tokensmatch = list()
    for x in contentlist:
        if(vocabulary.__contains__(x)):
            tokensmatch.append(x)
    return tokensmatch

if __name__ == "__main__":

    # get train data and test data
    TrainingDataPath = sys.argv[1]
    TestingDataPath = sys.argv[2]
    doccount, vocabulary, classname, docsineachclass = getAllVocabulary(TrainingDataPath)

    print("-------------------Training starts-------------------------------")
    print("Total Vocabulary count: " + str(len(vocabulary)))
    print("Total Documents in training data: " + str(doccount))
    print("Class labels: ", classname)

    wordsineachdocument = list()
    priorineachclass = list()
    Term = {}
    conditionprob = {}
    for x in classname:
        print("Doc count for the class:"+ x +" is "+ str(docsineachclass[x]))
        priorineachclass.append(docsineachclass[x]/doccount)
        print("Prior of class:"+ x +" is "+ str(docsineachclass[x]/doccount))
        wordsineachdocument = getConcatenatedText(os.path.join(TrainingDataPath,x))
        counts = Counter(wordsineachdocument)
        for t in vocabulary:
            count = 0
            if (wordsineachdocument.__contains__(t)):
                for word in wordsineachdocument:
                    if (t == word):
                        count = count + 1
                    default = x
                    Term[default + "_" + t] = count
            else:
                Term[x + "_" + t] = count
        countofallterms = 0
        for t in vocabulary:
            countofallterms = countofallterms + Term[x + "_" + t]
        for t in vocabulary:
            conditionprob[x + "_" + t] = (Term[x + "_" + t] + 1) / (countofallterms + len(vocabulary))

    print("-------------------Testing starts--------------------------------")
    for x in classname:
        files_list = list()
        Path = os.path.join(TestingDataPath, x)
        for root, dirs, files in os.walk(Path, topdown=False):
            count = 0
            for name in files:
                count += 1
                files_list.append(os.path.join(Path, name))
    score = {}
    acc = 0
    totalclass = 0
    for file in files_list:
        Testtokens = getMatchedTokens(vocabulary, file)
        predictedclass = ""
        flag = 0
        for c in classname:
            actualclass = c
            countofclass = 0
            score[c] = math.log2(priorineachclass[countofclass])
            countofclass += 1
            for t in Testtokens:
                if(vocabulary.__contains__(t)):
                    score[c] = score[c]+ math.log2(conditionprob[c+"_"+t])
            if(flag == 0):
                maxscore = score[c]
                predictedclass = c
                flag = 1
            if(score[c] > maxscore):
                maxscore = score[c]
                predictedclass = c
            if (predictedclass == actualclass):
                acc += 1
            totalclass += 1
    accuracy = (acc/totalclass) * 100
    print("Accuracy obtained on the testing data: ", accuracy)










