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
PREV2STATES = set()
PREVSTATES = set()
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
		if type_op == "trans":
			prev2tag = items[1]
			prevtag = items[2]
			tag = items[3]
			prob = float(items[4])
			key = (prev2tag,prevtag,tag)
			transProb[key] = log(prob)
			PREV2STATES.add(prev2tag)
			PREVSTATES.add(prevtag)
			if tag != FINAL_STATE:
				STATES.add(tag)
			
			

		else:
			token = items[2]
			tag = items[1]
			prob = float(items[3])
			if tag not in emitProb:
				emitProb[tag] = defaultdict(float)

			emitProb[tag][token] = log(prob)
			STATES.add(tag)
			VOCAB.add(token)


	return transProb,emitProb




def calc_viterbi(A,B,sentence):
	viterbi = {}
	backpointer = {}
	all_states = {} #K set in the algorithm in slides
	n = len(sentence)
	all_states[-2] = set([INIT_STATE])
	all_states[-1] = set([INIT_STATE])
	for i in range(0,n):
		all_states[i] = STATES



	score = float()
	viterbi[-1] = {}
	inits = (INIT_STATE,INIT_STATE)
	viterbi[-1][inits] = 0.0 #stores only logs , hence log(1) = 0 is stored
	for k in range(n):
		viterbi[k] = {}
		backpointer[k] = {}
		if sentence[k] not in VOCAB:
			sentence[k] = OOV_WORD

		for v in all_states[k]:
			for u in all_states[k-1]:
				max_score = float("-inf")
				argmax = ''
				
				for w in all_states[k-2]:
					word = sentence[k]
					key_A = (w,u,v)
					if (key_A not in A or A[key_A] == 0.0):
						A[key_A] = CONST

					if (B[v][word] == 0.0):
						B[v][word] = CONST

					score = viterbi[k-1][(w,u)] + A[key_A] + B[v][word]
					if score > max_score:
						max_score = score
						argmax = w
						
					
				viterbi[k][(u,v)] = max_score
				backpointer[k][(u,v)] = argmax


	#need to do the final step and then finish off and check
	new_max_score = float("-inf")
	new_score = float()
	argmax_u = ''
	argmax_v = ''

	for v in all_states[n-1]:
		for u in all_states[n-2]:
			key_A = (u,v,FINAL_STATE)
			if (key_A not in A or A[key_A] == 0.0):
				A[key_A] = CONST

			new_score = viterbi[n-1][(u,v)] + A[key_A]
			if new_score > new_max_score:
				new_max_score = new_score
				argmax_u = u
				argmax_v = v





	tag_sequence = [str] * n
	tag_sequence[n-1] = argmax_v
	tag_sequence[n-2] = argmax_u
	
	for i in range(n-3,-1,-1):
		
		tag_first = tag_sequence[i+1]
		tag_second = tag_sequence[i+2]
		tag_sequence[i] = backpointer[i+2][(tag_first,tag_second)]
		
	
	return ' '.join(tag_sequence)



def main():

	
	hmm_file = open(HMM_FILE)
	hmm_lines = hmm_file.readlines() #list of all lines inside the hmm file
	text_lines = sys.stdin.readlines() #reads in the file from the text file needed to tag (test text data set)
	transProb,emitProb = get_emit_transmit(hmm_lines) #A,B probability
	
	for line in text_lines:
		tags = calc_viterbi(transProb,emitProb,re.split("\s+", line.rstrip()))
		print tags



main()


