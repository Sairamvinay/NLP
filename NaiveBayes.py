import sys
import getopt
import os
import math
import operator
from collections import defaultdict

class NaiveBayes:
    class TrainSplit:
        """
        Set of training and testing data
        """
        def __init__(self):
            self.train = []
            self.test = []

    class Document:
        """
        This class represents a document with a label. classifier is 'pos' or 'neg' while words is a list of strings.
        """
        def __init__(self):
            self.classifier = ''
            self.words = []

    def __init__(self):
        """
        Initialization of naive bayes
        """
        self.stopList = set(self.readFile('data/english.stop'))
        self.bestModel = False
        self.stopWordsFilter = False
        self.naiveBayesBool = False
        self.numFolds = 10
        self.doc_counts = {}    #the doc count dictionary, which has only two keys pos or neg, which maps to words
        self.Vocabulary = set() #all the different words; used for smoothing in classify()
        self.prior = {}         #count of all the pos and negative docs
        self.totDocs = 0    #the total number of documents for the prior P(c) probability
        self.ALPHA = 2    #the constant ALPHA for smoothing in binary naive bayes. Can be changed
        self.BETA = 7    #the constant ALPHA for smoothing in Best Model naive bayes. Can be changed
        self.LIMIT = 5  #the constant LIMIT for limiting the number of words to be evaluated on in my custom model (Feature Selection)
        self.actualWords = set()    #for my custom model (Feature Selection), only picks words with atleast LIMIT count in either of the classifier

        # TODO
        # Implement a multinomial naive bayes classifier and a naive bayes classifier with boolean features. The flag
        # naiveBayesBool is used to signal to your methods that boolean naive bayes should be used instead of the usual
        # algorithm that is driven on feature counts. Remember the boolean naive bayes relies on the presence and
        # absence of features instead of feature counts.

        # When the best model flag is true, use your new features and or heuristics that are best performing on the
        # training and test set.

        # If any one of the flags filter stop words, boolean naive bayes and best model flags are high, the other two
        # should be off. If you want to include stop word removal or binarization in your best performing model, you
        # will need to write the code accordingly.

    def sum_counts(self,c):
        #return the sum of the count of all the words in the given classifier
        sum_val = float(0)
        
        for w in self.doc_counts[c].keys():
            sum_val = sum_val + self.doc_counts[c][w]

        return sum_val

    def argmax(self,sums):

        argmax_c = ''
        max_sum = float('-inf')
        for key in sums:
            if sums[key] > max_sum:
                max_sum = sums[key]
                argmax_c = key

        return argmax_c

    def classify(self, words):
        """
        Classify a list of words and return a positive or negative sentiment
        """
        if self.stopWordsFilter:
            words = self.filterStopWords(words)

        # TODO
        # classify a list of words and return the 'pos' or 'neg' classification
        # Write code here
        sums = {}
        if self.naiveBayesBool:
            
            words = list(set(words))    #need to remove duplicates here also so that only original (single) copies of word prob are calculated
            prob_c = float(0.0)
            
            V = len(list(self.Vocabulary))

            for c in self.prior.keys():

                #reset the prob_c P(c) and the P(w|c) as prob_likelihood
                prob_c = math.log(float(self.prior[c])/self.totDocs)
                
                prob_likelihood = prob_c #begin with this as the formula goes P(c) * product of all wi in V P(wi|c)
                
                denoninator = float(self.sum_counts(c)) #returns the sum of all the counts of words of docs with c classifier

                for w in words:
                #boolean naive bayes formula used

                    prob_likelihood = prob_likelihood + math.log(self.ALPHA + self.doc_counts[c][w]) - math.log(denoninator + (float(V) * self.ALPHA))

                sums[c] = prob_likelihood           
                

        elif self.bestModel:

            #my model is to eliminate those words which have a count less than the limit value in the __init__ defined
            #i pick only those words from +ve and -ve classifications whose count is atleast the limit value in each of the states
            #i then use my binary Naive Bayes classifier on that document

            #NOTES: beta = 7: limit = 10 gives 83.95%,  limit = 5 gives 84%
            self.actualWords = set()    #reset to an empty set always before classifying
            for w in words:
                #Pick only those words which have a minimal count of LIMIT in either of the classifiers
                #else drop that word (ignore it, don't add)
                if self.doc_counts["pos"][w] > self.LIMIT or self.doc_counts["neg"][w] > self.LIMIT:
                    self.actualWords.add(w)
            

            prob_c = float(0.0)
            
            V = len(list(self.Vocabulary))

            for c in self.prior.keys():

                #reset the prob_c P(c) and the P(w|c) as prob_likelihood
                prob_c = math.log(float(self.prior[c])/self.totDocs)
                
                prob_likelihood = prob_c #begin with this as the formula goes P(c) * product of all wi in V P(wi|c)
                
                denoninator = float(self.sum_counts(c)) #returns the sum of all the counts of words of docs with c classifier

                for w in list(self.actualWords):
                #boolean naive bayes formula used

                    prob_likelihood = prob_likelihood + math.log(self.BETA + self.doc_counts[c][w]) - math.log(denoninator + (float(V) * self.BETA))

                
                sums[c] = prob_likelihood           
                



        else:

            #the general case with either no option flags or the -f filter flag on (neither Boolean nor Best Model)
            prob_c = float(0.0)
            V = len(list(self.Vocabulary))

            for c in self.prior.keys():

                #reset the prob_c P(c) and the P(w|c) as prob_likelihood
                prob_c = math.log(float(self.prior[c])/self.totDocs)
                
                prob_likelihood = prob_c #begin with this as the formula goes P(c) * product of all wi in V P(wi|c)
                
                denoninator = float(self.sum_counts(c)) #returns the sum of all the counts of words of docs with c classifier

                for w in words:
                    
                    #regular add one-smoothing used
                    prob_likelihood = prob_likelihood + math.log(1 + float(self.doc_counts[c][w])) - math.log(denoninator + V) 

                sums[c] = prob_likelihood


        return self.argmax(sums)

    def addDocument(self, classifier, words):
        """
        Train your model on a document with label classifier (pos or neg) and words (list of strings). You should
        store any structures for your classifier in the naive bayes class. This function will return nothing
        """
        # TODO
        # Train model on document with label classifiers and words
        # Write code here

        if classifier not in self.prior:
            self.prior[classifier] = 0


        if classifier not in self.doc_counts:
            self.doc_counts[classifier] = defaultdict(int)

        if self.naiveBayesBool:
            words = list(set(words))    #need to deal without duplicates (set removes duplicates for us) for binary naive bayes method
    
        self.prior[classifier] += 1  #increment the number of documents in the classifer
        self.totDocs += 1   #increment the total number of docs

        for w in words:
            
            
            self.doc_counts[classifier][w] += 1
            self.Vocabulary.add(w)

            


    def readFile(self, fileName):
        """
        Reads a file and segments.
        """
        contents = []
        f = open(fileName)
        for line in f:
            contents.append(line)
        f.close()
        str = '\n'.join(contents)
        result = str.split()
        return result

    def trainSplit(self, trainDir):
        """Takes in a trainDir, returns one TrainSplit with train set."""
        split = self.TrainSplit()
        posDocTrain = os.listdir('%s/pos/' % trainDir)
        negDocTrain = os.listdir('%s/neg/' % trainDir)
        for fileName in posDocTrain:
            doc = self.Document()
            doc.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
            doc.classifier = 'pos'
            split.train.append(doc)
        for fileName in negDocTrain:
            doc = self.Document()
            doc.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
            doc.classifier = 'neg'
            split.train.append(doc)
        return split

    def train(self, split):
        for doc in split.train:
            words = doc.words
            if self.stopWordsFilter:
                words = self.filterStopWords(words)
            self.addDocument(doc.classifier, words)

    def crossValidationSplits(self, trainDir):
        """Returns a lsit of TrainSplits corresponding to the cross validation splits."""
        splits = []
        posDocTrain = os.listdir('%s/pos/' % trainDir)
        negDocTrain = os.listdir('%s/neg/' % trainDir)
        # for fileName in trainFileNames:
        for fold in range(0, self.numFolds):
            split = self.TrainSplit()
            for fileName in posDocTrain:
                doc = self.Document()
                doc.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
                doc.classifier = 'pos'
                if fileName[2] == str(fold):
                    split.test.append(doc)
                else:
                    split.train.append(doc)
            for fileName in negDocTrain:
                doc = self.Document()
                doc.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
                doc.classifier = 'neg'
                if fileName[2] == str(fold):
                    split.test.append(doc)
                else:
                    split.train.append(doc)
            yield split

    def test(self, split):
        """Returns a list of labels for split.test."""
        labels = []
        for doc in split.test:
            words = doc.words
            if self.stopWordsFilter:
                words = self.filterStopWords(words)
            guess = self.classify(words)
            labels.append(guess)
        return labels

    def buildSplits(self, args):
        """
        Construct the training/test split
        """
        splits = []
        trainDir = args[0]
        if len(args) == 1:
            print '[INFO]\tOn %d-fold of CV with \t%s' % (self.numFolds, trainDir)

            posDocTrain = os.listdir('%s/pos/' % trainDir)
            negDocTrain = os.listdir('%s/neg/' % trainDir)
            for fold in range(0, self.numFolds):
                split = self.TrainSplit()
                for fileName in posDocTrain:
                    doc = self.Document()
                    doc.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
                    doc.classifier = 'pos'
                    if fileName[2] == str(fold):
                        split.test.append(doc)
                    else:
                        split.train.append(doc)
                for fileName in negDocTrain:
                    doc = self.Document()
                    doc.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
                    doc.classifier = 'neg'
                    if fileName[2] == str(fold):
                        split.test.append(doc)
                    else:
                        split.train.append(doc)
                splits.append(split)
        elif len(args) == 2:
            split = self.TrainSplit()
            testDir = args[1]
            print '[INFO]\tTraining on data set:\t%s testing on data set:\t%s' % (trainDir, testDir)
            posDocTrain = os.listdir('%s/pos/' % trainDir)
            negDocTrain = os.listdir('%s/neg/' % trainDir)
            for fileName in posDocTrain:
                doc = self.Document()
                doc.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
                doc.classifier = 'pos'
                split.train.append(doc)
            for fileName in negDocTrain:
                doc = self.Document()
                doc.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
                doc.classifier = 'neg'
                split.train.append(doc)

            posDocTest = os.listdir('%s/pos/' % testDir)
            negDocTest = os.listdir('%s/neg/' % testDir)
            for fileName in posDocTest:
                doc = self.Document()
                doc.words = self.readFile('%s/pos/%s' % (testDir, fileName))
                doc.classifier = 'pos'
                split.test.append(doc)
            for fileName in negDocTest:
                doc = self.Document()
                doc.words = self.readFile('%s/neg/%s' % (testDir, fileName))
                doc.classifier = 'neg'
                split.test.append(doc)
            splits.append(split)
        return splits

    def filterStopWords(self, words):
        """
        Stop word filter
        """
        removed = []
        for word in words:
            if not word in self.stopList and word.strip() != '':
                removed.append(word)
        return removed


def test10Fold(args, stopWordsFilter, naiveBayesBool, bestModel):
    nb = NaiveBayes()
    splits = nb.buildSplits(args)
    avgAccuracy = 0.0
    fold = 0
    for split in splits:
        classifier = NaiveBayes()
        classifier.stopWordsFilter = stopWordsFilter
        classifier.naiveBayesBool = naiveBayesBool
        classifier.bestModel = bestModel
        accuracy = 0.0
        for doc in split.train:
            words = doc.words
            classifier.addDocument(doc.classifier, words)

        for doc in split.test:
            words = doc.words
            guess = classifier.classify(words)
            if doc.classifier == guess:
                accuracy += 1.0

        accuracy = accuracy / len(split.test)
        avgAccuracy += accuracy
        print '[INFO]\tFold %d Accuracy: %f' % (fold, accuracy)
        fold += 1
    avgAccuracy = avgAccuracy / fold
    print '[INFO]\tAccuracy: %f' % avgAccuracy


def classifyFile(stopWordsFilter, naiveBayesBool, bestModel, trainDir, testFilePath):
    classifier = NaiveBayes()
    classifier.stopWordsFilter = stopWordsFilter
    classifier.naiveBayesBool = naiveBayesBool
    classifier.bestModel = bestModel
    trainSplit = classifier.trainSplit(trainDir)
    classifier.train(trainSplit)
    testFile = classifier.readFile(testFilePath)
    print classifier.classify(testFile)


def main():
    stopWordsFilter = False
    naiveBayesBool = False
    bestModel = False
    (options, args) = getopt.getopt(sys.argv[1:], 'fbm')
    if ('-f', '') in options:
        stopWordsFilter = True
    elif ('-b', '') in options:
        naiveBayesBool = True
    elif ('-m', '') in options:
        bestModel = True

    if len(args) == 2 and os.path.isfile(args[1]):
        classifyFile(stopWordsFilter, naiveBayesBool, bestModel, args[0], args[1])
    else:
        test10Fold(args, stopWordsFilter, naiveBayesBool, bestModel)


if __name__ == "__main__":
    main()
