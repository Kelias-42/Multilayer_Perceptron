import sys
import numpy as np
import pandas as pd
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt

from neural_network import neural_network

states={"M": 1,"B": 2}

states_rev = {value: key for key, value in states.items()}

def get_data():
	try:
		data_path = sys.argv[1]
		return (pd.read_csv(data_path))
	except IndexError:
		sys.exit("usage: python describe.py [your_dataset].csv")
	except IOError:
		sys.exit("could not read data file")
	except:
		sys.exit("Error")
	
def preprocess_data():
	data = get_data()
	columns = [f"p{i}" for i in range(1, 32)]
	columns.insert(1, "state")
	data.columns = columns
	data["state"].replace(states, inplace=True)
	data = data.select_dtypes('number')
	data["state"].replace(states_rev, inplace=True)
	for i in [1, 10, 11, 13, 16, 20, 21, 31]:
		data.drop(f"p{i}", axis=1, inplace=True)

def process_mnist():
	f = open("resources/mnist_train_100.csv", 'r')
	data = f.readlines()
	f.close()
	f = open("resources/mnist_test_10.csv", 'r')
	test_data = f.readlines()
	f.close()
	for i in range(len(data)):
		if (data[i][-1] == '\n'):
			data[i] = data[i][0:len(data[i])-1]
	for i in range(len(test_data)):
		if (test_data[i][-1] == '\n'):
			test_data[i] = test_data[i][0:len(test_data[i])-1]
	for i in range(len(data)):
		data[i] = data[i].split(',')
		for j in range(1, len(data[i])):
			data[i][j] = (float(data[i][j]) / 255 * 0.99) + 0.01
	for i in range(len(test_data)):
		test_data[i] = test_data[i].split(',')
		for j in range(1, len(test_data[i])):
			test_data[i][j] = (float(test_data[i][j]) / 255 * 0.99) + 0.01
	for e in range(epochs):
		for elm in data:
			expected_output = [0.01] * 10
			expected_output[int(elm[0])] = 0.99
			n_net.train(elm[1:], expected_output)
	correct = 0
	for i in range(10):
		prediction = n_net.predict(test_data[i][1:]).tolist()
		if int(test_data[i][0]) == prediction.index(max(prediction)):
			correct += 1
	print("got", correct, "correct answers out of 10")

input_neurons = 784
output_neurons = 10
learning_rate = 0.2
n_hidden_layers = 1
n_hidden_neurons = 100
epochs = 20

if __name__ == "__main__":
	n_net = neural_network(input_neurons, output_neurons, learning_rate, n_hidden_layers, n_hidden_neurons)
	process_mnist()