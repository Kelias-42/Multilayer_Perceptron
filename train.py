import sys
import numpy as np
import pandas as pd
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt

from neural_network_multilayer import neural_network


def get_data():
	try:
		f = open(sys.argv[1], 'r')
		data = f.readlines()
		f.close()
		return data
	except IndexError:
		sys.exit("usage: python describe.py [your_dataset].csv")
	except IOError:
		sys.exit("could not read data file")
	except:
		sys.exit("Error")
	
def normalize(data):
	for column in range(len(data[0])):
		lowest = np.amin(data[:,column])
		highest = np.amax(data[:,column])
		for i in range(len(data[:,0])):
			data[i][column] = (data[i][column] - lowest) / (lowest + highest)
			if data[i][column] == 1:
				data[i][column] == 0.99
			elif data[i][column] == 0:
				data[i][column] == 0.01

def	softmax(x):
	return np.exp(x) / np.sum(np.exp(x), axis=0)

def preprocess_data():

	data = get_data()
	# formatting training data
	for i in range(len(data)):
		if (data[i][-1] == '\n'):
			data[i] = data[i][0:len(data[i])-1]
	for i in range(len(data)):
		data[i] = data[i].split(',')
	for i in range(len(data)):
		data[i].pop(0)
	for i in range(len(data)):
		data[i][0] = 0 if data[i][0] == 'M' else 1
	data = np.array([np.array(sub) for sub in data]).astype(float)
	normalize(data[:,1:])
	return data

def train(data):

	# model training
	train_data = data[0:500]
	for e in range(epochs):
		for elm in train_data:
			expected_output = [0.01] * 2
			expected_output[int(elm[0])] = 0.99
			n_net.train(elm[1:], expected_output)

	# model testing		
	test_data = data[501:]
	correct = 0
	for i in range(len(test_data)):
		prediction = n_net.predict(test_data[i][1:]).tolist()
		if int(test_data[i][0]) == prediction.index(max(prediction)):
			correct += 1
		#prediction = softmax(prediction)
		#print(prediction)

	print("Acc is", correct/len(test_data))

input_neurons = 30
output_neurons = 2
learning_rate = 0.2
n_hidden_layers = 4
n_hidden_neurons = 4
epochs = 3

if __name__ == "__main__":
	n_net = neural_network(input_neurons, output_neurons, learning_rate, n_hidden_layers, n_hidden_neurons)
	data = preprocess_data()
	#train(data)
	print(n_net.predict(data[0][1:]))
	