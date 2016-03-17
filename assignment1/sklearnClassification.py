import numpy as np  
from six.moves import cPickle as pickle  
from sklearn.linear_model import LogisticRegression   
import sys

train_num = int(sys.argv[1])
test_num = int(sys.argv[2])
  
with open('notMNIST.pickle','rb') as f:
    dataset = pickle.load(f)

train_dataset = dataset['train_dataset']
train_labels = dataset['train_labels']
test_dataset = dataset['test_dataset']
test_labels = dataset['test_labels']

train50 = train_dataset[:train_num]
label50 = train_labels[:train_num]

train50 = train50.reshape((train_num,784))
classifier2 = LogisticRegression()
classifier2.fit(train50, label50)

test100 = test_dataset[:test_num]
testlabel100 = test_labels[:test_num]

test100 = test100.reshape(test_num,784)
count = 0
for i in range(0,test_num-1):
    try:
        y = classifier2.predict(test100[i])
        if y[0] == test_labels[i]:
	    count += 1
    except ValueError:
	print 'valueerror'
print('Accuracy: '+str(count*100/test_num) + '%') 
