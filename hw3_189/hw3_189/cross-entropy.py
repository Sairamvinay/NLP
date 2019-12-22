import math
import sys

FNAME = sys.argv[1]

File = open(FNAME)
CONST = math.log(10**-300)
Lines = File.readlines()
N = len(Lines)
score = 0
for line in Lines:
	items = line.split()
	if len(items) == 1: #Check for failures
		score -= CONST
		continue
	

	score -= math.log(float(items[1]),2)
	

score /= N

print "\nThe cross-entropy score of the grammar file %s is %f\n" %(FNAME,score)
