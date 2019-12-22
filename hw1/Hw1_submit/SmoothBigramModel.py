import math, collections

class SmoothBigramModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.unigramCounts = collections.defaultdict(lambda : 0) #contains the count(word) for each word in training corpus and 0 for non_existent words as default
    self.bigramCounts = collections.defaultdict(lambda : 0) #contains the bigram count(word_i|word_i-1) for each word in training corpus and 0 for non_existent words as default
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

    '''
        Each word in the training corpus is found and then added to the unigram count first
    '''
    for sentence in corpus.corpus:
        for datum in sentence.data:
            word = datum.word
            self.unigramCounts[word] += 1   #need this for the denominator expression for this model
            self.total += 1

    #the bigrams are counts by a special key (seperated by | symbol).
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

    #the eqn is P = [C(wn - 1,wn)+1] / [C(wn - 1) +V]
    #the logs are accordingly taken
    
    score = float(0)
    for i in range(1,len(sentence)):
        word_curr = sentence[i]
        word_prev = sentence[i - 1]
        key = word_curr + "|" + word_prev   #seperate key using | for getting bigram counts
        score += math.log(1 + self.bigramCounts[key])
        score -= math.log(self.unigramCounts[word_prev] + self.Voc_count)


    return score




