import math, collections

#This is the Kneser Ney Model. I am implementing it as it gives me a good accuracy better than the backoff model
class CustomModel:

  CONSTANT = 10 ** -30  
  def __init__(self, corpus):
    """Initial custom language model and structures needed by this model"""
    self.unigramCounts = collections.defaultdict(lambda : 0)   #data structure for storing all unigrams in corpus
    self.bigramCounts = collections.defaultdict(lambda : 0)    #data structure for storing all bigrams in corpus
    self.prevWordCounts = collections.defaultdict(set) #data structure for storing list of the forward words of the key of prev_word present in train
    self.currWordCounts = collections.defaultdict(set) #data structure for storing list of the previous words of the key of curr_word present in train
    self.bigramforcurr = collections.defaultdict(list)
    self.total = 0  #N value = num of tokens
    self.Voc_count = 0  #V value = num of words in vocabulary
    self.discount = 0.75 #the discount(d) value in the model 0.75
    self.uniquebigramCounts = 0 #the non-repeated count of the number of bigrams with a given word as 2nd term
    self.train(corpus)


  def train(self, corpus):
    """ Takes a corpus and trains your language model.
    """  
    # TODO your code here

    for sentence in corpus.corpus:
        for datum in sentence.data:
            word = datum.word
            self.unigramCounts[word] += 1
            self.total += 1

    for sentence in corpus.corpus:
        for i in range(1,len(sentence)):

            word_curr = sentence.data[i].word
            word_prev = sentence.data[i-1].word
            key = word_curr + "|" + word_prev   #seperate key using | for bigram counts
            self.bigramCounts[key] += 1
    
    for sentence in corpus.corpus:
        for i in range(1,len(sentence)):
            word_curr = sentence.data[i].word
            word_prev = sentence.data[i-1].word
            self.prevWordCounts[word_prev].add(word_curr)   #add the current word (2nd word) to the dictionary of set for prevWords
            self.currWordCounts[word_curr].add(word_prev)   #add the previous word (1nd word) to the dictionary of set for currWords
            self.bigramforcurr[word_curr].append(word_prev) #the list of all prev word tokens (needed for the Pcont term)


    self.Voc_count = len(self.unigramCounts)
    for datum in sentence.data:
        word = datum.word
        self.uniquebigramCounts += len(self.currWordCounts[word])


    


  def score(self, sentence):
    """ With list of strings, return the log-probability of the sentence with language model. Use
        information generated from train.
    """
    # TODO your code here
    score = float(0)
    first_term = float(0)
    second_term = float(0)
    second_lambda = float(0)
    second_Pcont = float(0)
    for i in range(1,len(sentence)):
        word_curr = sentence[i]
        word_prev = sentence[i-1]
        key = word_curr + "|" + word_prev   #seperate key using | for bigram counts

        #the unigram count is first checked and dealt accordingly
        if self.unigramCounts[word_prev] == 0:
            first_term = float(0)
            second_lambda = float(0)

        else:
            first_term = max(self.bigramCounts[key] - self.discount,0.0) / self.unigramCounts[word_prev]
            second_lambda = self.discount * len(self.prevWordCounts[word_prev]) / self.unigramCounts[word_prev]


        second_Pcont = len(self.bigramforcurr[word_curr]) / float(self.uniquebigramCounts)  #in formula
        second_term = second_lambda * second_Pcont
        #if the Pkn = 0 , add the log of a really really small constant as it may help in including the factor v close to log(0) = -inf
        if (first_term + second_term == 0):
            score += math.log(CustomModel.CONSTANT)

        else:
            score += math.log(first_term + second_term)



    return score






