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
		self.activation_function = lambda x: 1 / (1 + np.exp(-x))

	def train():
		pass

	def predict(self, inputs_list):
		# convert inputs list to a transposed 2d array
		inputs = np.array(inputs_list, ndmin=2).T
		
		# signal from input layer to out of hidden layer
		hidden_inputs = np.dot(self.wih, inputs)
		hidden_outputs = self.activation_function(hidden_inputs)
		
		# signal that goes out of output layer
		final_inputs = np.dot(self.who, hidden_outputs)
		final_outputs = self.activation_function(final_inputs)

		return final_outputs

