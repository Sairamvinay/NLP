import matplotlib.pyplot as plt
x = [5000 * i for i in range(1,9)]
y_error_word = [0.10826,0.08144,0.07029,0.06558,0.06127,0.05718,0.05599,0.05409]
y_error_sent = [0.8,0.74941,0.72706,0.70353,0.69,0.66765,0.65941,0.65588]

plt.plot(x,y_error_sent)

plt.xlabel('Training Data Set Size')
plt.ylabel('Error rate for sentences')

plt.title('Error rate for sentences tested on ptb.22.* test set Graph')
plt.show()
plt.close('all')
