# Code for maximum likelihood estimation of a bigram HMM from 
# column-formatted training data.

# Usage:  train_hmm.py tags text > hmm-file

# The training data should consist of one line per sequence, with
# states or symbols separated by whitespace and no trailing whitespace.
# The initial and final states should not be mentioned; they are 
# implied.  
# The output format is the HMM file format as described in viterbi.pl.

import sys,re
from itertools import izip
from collections import defaultdict

NUM_TRAIN = 5000
TAG_FILE=sys.argv[1]
TOKEN_FILE=sys.argv[2]

vocab={}
OOV_WORD="OOV"
INIT_STATE="init"
FINAL_STATE="final"
STATES = set()
emissions={}
transitions={}
transitionsTotal=defaultdict(int)
emissionsTotal=defaultdict(int)


with open(TAG_FILE) as tagFile, open(TOKEN_FILE) as tokenFile:
	
	for tagString, tokenString in izip(tagFile, tokenFile):

		tags=re.split("\s+", tagString.rstrip())
		tokens=re.split("\s+", tokenString.rstrip())
		pairs=zip(tags, tokens)

		prevtag=INIT_STATE
		prev2tag = INIT_STATE

		for (tag, token) in pairs:

			# this block is a little trick to help with out-of-vocabulary (OOV)
			# words.  the first time we see *any* word token, we pretend it
			# is an OOV.  this lets our model decide the rate at which new
			# words of each POS-type should be expected (e.g., high for nouns,
			# low for determiners).
			STATES.add((prev2tag,prevtag))
			
			if token not in vocab:
				vocab[token]=1
				token=OOV_WORD

			if tag not in emissions:
				emissions[tag]=defaultdict(int)


			# increment the emission observation
			emissions[tag][token]+=1
			emissionsTotal[tag]+=1


			key = (prev2tag,prevtag,tag)
			if key not in transitions:
				transitions[key] = 1

			else:
				transitions[key] += 1


			tot_key = (prev2tag,prevtag)
			transitionsTotal[tot_key]+=1
			prev2tag = prevtag
			prevtag = tag



		key_check = (prev2tag,prevtag,FINAL_STATE)
		if key_check not in transitions:
			transitions[key_check] = 1

		else:
			transitions[key_check] += 1

		tot_key_check = (prev2tag,prevtag)
		STATES.add(tot_key_check)
		transitionsTotal[tot_key_check] += 1



for tag in emissions:
	for token in emissions[tag]:
		print "emit %s %s %s " % (tag, token, float(emissions[tag][token]) / emissionsTotal[tag])

V = len(list(STATES))
for key_trans in transitions.keys():
	total_key_trans = key_trans[:2]
	prev2tag = key_trans[0]
	prevtag = key_trans[1]
	tag = key_trans[2]
	print "trans %s %s %s %s" % (prev2tag, prevtag, tag, float (transitions[key_trans] + 1) / (transitionsTotal[total_key_trans] + V))






