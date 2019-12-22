# Noah A. Smith
# 2/21/08
# Runs the Viterbi algorithm (no tricks other than logmath!), given an
# HMM, on sentences, and outputs the best state path.

# Usage:  viterbi.py hmm-file < text > tags
import sys,re
from collections import defaultdict
from math import log,exp

HMM_FILE = sys.argv[1]
OOV_WORD="OOV"
INIT_STATE="init"
FINAL_STATE="final"
STATES = set()
VOCAB = set()
CONST = log(10**-300)

def get_emit_transmit(hmm_lines):
	emitProb = {}
	transProb = {}
	for line in hmm_lines:
		line.strip('\n')
		items = line.split(' ')
		type_op = items[0]
		tag = items[1]
		prob = float(items[3])
		if type_op == "trans":
			next_tag = items[2]
			if tag not in transProb:
				transProb[tag] = defaultdict(float)


			transProb[tag][next_tag] = log(prob)
			STATES.add(tag)
			STATES.add(next_tag)

		else:
			token = items[2]
			if tag not in emitProb:
				emitProb[tag] = defaultdict(float)

			emitProb[tag][token] = log(prob)
			STATES.add(tag)
			VOCAB.add(token)


	return transProb,emitProb




def calc_viterbi(A,B,sentence):
	viterbi = {}
	backpointer = {}
	N = len(sentence)
	sentence = [""] + sentence
	
	for i in range(N+1):
		viterbi[i] = {}
		backpointer[i] = {}
	
	viterbi[0][INIT_STATE] = 0.0
	score = float(0)
	
	for i in range(1,N+1):
		if sentence[i] not in VOCAB:
			sentence[i] = OOV_WORD

		for state_curr in STATES:
			for state_prev in STATES:
				if state_prev == FINAL_STATE or state_curr == FINAL_STATE:
					continue

				
				if ((A[state_prev][state_curr] != 0.0) and (B[state_curr][sentence[i]] != 0.0) and (state_prev in viterbi[i-1])):
					score = viterbi[i-1][state_prev] + A[state_prev][state_curr] + B[state_curr][sentence[i]]
					if ((state_curr not in viterbi[i]) or (score > viterbi[i][state_curr])):
						viterbi[i][state_curr] = score
						backpointer[i][state_curr] = state_prev



	
	tag_sequence = []
	found = False
	max_score = CONST
	max_tag = str()
	for state in STATES:
		if state == FINAL_STATE or state == INIT_STATE:
			continue

		if (A[state][FINAL_STATE] != 0.0 and (state in viterbi[N])):
			score = viterbi[N][state] + A[state][FINAL_STATE]
			if ((not found) or score > max_score):
				max_score = score
				max_tag = state
				found = True

	if not found:
		return ""

	else:

		for i in range(N,0,-1):
			tag_sequence = [max_tag] + tag_sequence
			max_tag = backpointer[i][max_tag]

		return " ".join(tag_sequence)





def main():

	
	hmm_file = open(HMM_FILE)
	hmm_lines = hmm_file.readlines() #list of all lines inside the hmm file
	text_lines = sys.stdin.readlines() #reads in the file from the text file needed to tag (test text data set)
	transProb,emitProb = get_emit_transmit(hmm_lines) #A,B probability
	N = len(transProb)	#N is #states
	
	for line in text_lines:
		line = text_lines[0]	
		tags = calc_viterbi(transProb,emitProb,re.split("\s+", line.rstrip()))
		print tags



main()



