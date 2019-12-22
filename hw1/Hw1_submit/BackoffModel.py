import math, collections

class BackoffModel:
  
  #The backoff model
  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.unigramCounts = collections.defaultdict(lambda : 0)    #the unigram counts for the words in training corpus
    self.bigramCounts = collections.defaultdict(lambda : 0) #the bigram counts for the words in training corpus
    self.total = 0  #N value = num of tokens   
    self.Voc_count = 0  #V value = num of words in vocabulary
    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model. 
        Compute any counts or other corpus statistics in this function.
    """  
    # TODO your code here
    # Tip: To get words from the corpus, try
    #    for sentence in corpus.corpus:
    #       for datum in sentence.data:  
    #         word = datum.word

    #the unigram counts in corpus are found
    for sentence in corpus.corpus:
        for datum in sentence.data:
            word = datum.word
            self.unigramCounts[word] += 1
            self.total += 1

    #the bigram counts in the corpus are found
    for sentence in corpus.corpus:
        for i in range(1,len(sentence)):

            word_curr = sentence.data[i].word
            word_prev = sentence.data[i-1].word
            key = word_curr + "|" + word_prev   #seperate key using | for bigram counts
            self.bigramCounts[key] += 1


    self.Voc_count = len(self.unigramCounts)


  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the 
        sentence using your language model. Use whatever data you computed in train() here.
    """
    # TODO your code here

    #the notion is simple: if bigram is found in training: use regular bigram, else use smoothed unigram for the current word
    score = float(0)

    for i in range(1,len(sentence)):
        
        word_curr = sentence[i]
        word_prev = sentence[i-1]
        key = word_curr + "|" + word_prev
       
        #the equation for a bigram, find the key seperated using | symbol and then find ratio
        if self.bigramCounts[key] > 0:
            score += math.log(self.bigramCounts[key])   #the log of the bigram count for that particular bigram
            score -= math.log(self.unigramCounts[word_prev]) #the log of the unigram count of prev_word

        else:
            #the smoothed unigram formula
            score += math.log(self.unigramCounts[word_curr] + 1)    #the log of the count(token) + 1 for every token is added (multiplied in original)
            score -= math.log(self.total + self.Voc_count)  #the log of the (N + V) term which is subtracted (divided in original)


    return score


