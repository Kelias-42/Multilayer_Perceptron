import math
import numpy as np

class neural_network:

	def __init__(self, in_neurons, out_neurons, learning_rate, n_hidden_layers, n_hidden_neurons):
		self.n_in = in_neurons
		self.n_out = out_neurons
		self.lr = learning_rate
		self.n_hl = n_hidden_layers
		self.n_hn = n_hidden_neurons
		self.wih = np.random.normal(0, 1/math.sqrt(self.n_hn), (self.n_hn, self.n_in))
		self.who = np.random.normal(0, 1/math.sqrt(self.n_out), (self.n_out, self.n_hn))
		#self.wih = np.random.normal(0.0, pow(self.n_hn, -0.5), (self.n_hn, self.n_in))
		#self.who = np.random.normal(0.0, pow(self.n_out, -0.5), (self.n_out, self.n_hn))
		self.activation_function = lambda x: 1 / (1 + np.exp(-x))
		#self.softmax = lambda x: np.exp(x) / np.sum(np.exp(x), axis=0)

	def train(self, inputs_list, targets_list):
		# convert inputs and targets lists to a transposed 2D array
		inputs = np.array(inputs_list, ndmin=2).T
		targets = np.array(targets_list, ndmin=2).T
		
		# signal from input layer to out of hidden layer
		hidden_inputs = np.dot(self.wih, inputs)
		hidden_outputs = self.activation_function(hidden_inputs)
		
		# signal that goes out of output layer
		final_inputs = np.dot(self.who, hidden_outputs)
		final_outputs = self.activation_function(final_inputs)

		# compute prediction and errors
		output_errors = targets - final_outputs
		hidden_errors = np.dot(self.who.T, output_errors)
		
		# update the weights for the links between the hidden and output layers
		self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
		# update the weights for the links between the input and hidden layers
		self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))

	def predict(self, inputs_list):
		# convert inputs list to a transposed 2D array
		inputs = np.array(inputs_list, ndmin=2).T
		
		# signal from input layer to out of hidden layer
		hidden_inputs = np.dot(self.wih, inputs)
		hidden_outputs = self.activation_function(hidden_inputs)
		
		# signal that goes out of output layer
		final_inputs = np.dot(self.who, hidden_outputs)
		final_outputs = self.activation_function(final_inputs)
		#final_outputs = self.softmax(final_inputs)

		return final_outputs

