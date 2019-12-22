import math, collections

class SmoothUnigramModel:

  
  #developing the smooth unigram model which evaluats P(w_i) = (C(w_i) + 1)/(N + V) where N is #tokens in corpus, V is #elem in vocab
  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.unigramCounts = collections.defaultdict(lambda: 0) #contains the count(word) for each word in training corpus and 0 for non_existent words as default
    self.total = 0  #the N value in the model
    self.Voc_Count = 0  #the V value in the model
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

    #take each token from the training corpus and add it to unigram counts
    #self.total counts the #token in the corpus (N value)
    for sentence in corpus.corpus:
        for datum in sentence.data:
            token = datum.word
            self.unigramCounts[token] = self.unigramCounts[token] + 1
            self.total += 1

    self.Voc_Count = len(self.unigramCounts)    #the V value, #elements in vocabulary

  

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the 
        sentence using your language model. Use whatever data you computed in train() here.
    """
    # TODO your code here
    #P(w_i) = (C(w_i) + 1)/(N + V) is calculated using log

    score = float(0)
    for token in sentence:
        score += math.log(self.unigramCounts[token] + 1)    #the log of the count(token) + 1 for every token is added (multiplied in original)
        score -= math.log(self.total + self.Voc_Count)  #the log of the (N + V) term which is subtracted (divided in original)


    return score
