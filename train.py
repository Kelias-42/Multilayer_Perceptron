import sys
import numpy as np
import pandas as pd
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
from math import log
from progress.bar import FillingSquaresBar

from neural_network_multilayer import neural_network


def get_data(i):
	try:
		print(sys.argv[i])
		f = open(sys.argv[i], 'r')
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

def preprocess_data(data):
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
	train_data = data
	for e in range(epochs):
		with FillingSquaresBar('Processing epoch {}/{}'.format(e+1, epochs), max=len(train_data)) as bar:
			for elm in train_data:
				expected_output = [0.01] * 2
				expected_output[int(elm[0])] = 0.99
				n_net.train(elm[1:], expected_output)
				bar.next()
			bar.finish()
		acc_test, acc_train = test()
		print('Acc_train: {:.4f} Acc_test: {:.4f}\n'.format(acc_train, acc_test))

def binary_cross_entropy(result, predictions):
	n = len(result)
	return -1/n * sum(result[i][0] * log(predictions[i][1][0]) + (1 - result[i][0]) * log(1 - predictions[i][1][0]) for i in range(n)) 

def test():
	# model evaluation on testing data		
	predictions = []
	test_data = data
	correct = 0
	for i in range(len(test_data)):
		prediction = n_net.predict(test_data[i][1:]).tolist()
		predictions.append(prediction)
		if int(test_data[i][0]) == prediction.index(max(prediction)):
			correct += 1
		#prediction = softmax(prediction)
		#print(prediction)
	acc_test = correct/len(test_data)
	print(binary_cross_entropy(test_data, predictions))

	# model evaluation on training data
	predictions = []
	train_data = data_test
	correct = 0
	for i in range(len(train_data)):
		prediction = n_net.predict(train_data[i][1:]).tolist()
		predictions.append(prediction)
		if int(train_data[i][0]) == prediction.index(max(prediction)):
			correct += 1
	acc_train = correct/len(train_data)
	print(binary_cross_entropy(train_data, predictions))
	return acc_test, acc_train 

input_neurons = 30
output_neurons = 2
learning_rate = 0.02
n_hidden_layers = 1
n_hidden_neurons = 42
epochs = 50

if __name__ == "__main__":
	n_net = neural_network(input_neurons, output_neurons, learning_rate, n_hidden_layers, n_hidden_neurons)
	data = preprocess_data(get_data(1))
	data_test = preprocess_data(get_data(2))
	train(data)
	#print(n_net.predict(data[0][1:]))
	