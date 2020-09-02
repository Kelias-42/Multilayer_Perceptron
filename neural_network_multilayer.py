import math
import numpy as np

class neural_network:

	def __init__(self, in_neurons, out_neurons, learning_rate, n_hidden_layers, n_hidden_neurons):
		self.n_in = in_neurons
		self.n_out = out_neurons
		self.lr = learning_rate
		self.n_hl = n_hidden_layers
		self.n_hn = n_hidden_neurons

		#self.wih = np.random.normal(0, 1 / math.sqrt(self.n_hn), (self.n_hn, self.n_in))
		#self.who = np.random.normal(0, 1 / math.sqrt(self.n_out), (self.n_out, self.n_hn))
		
		self.weights = [np.random.normal(0, 1 / math.sqrt(self.n_hn), (self.n_hn, self.n_in))]
		for	i in range(self.n_hl - 1):
			self.weights.append(np.random.normal(0, 1 / math.sqrt(self.n_hn), (self.n_hn, self.n_hn)))
		self.weights.append(np.random.normal(0, 1 / math.sqrt(self.n_out), (self.n_out, self.n_hn)))

		self.activation_function = lambda x: 1 / (1 + np.exp(-x))

	def train(self, inputs_list, targets_list):
		# convert inputs and targets lists to a transposed 2D array
		inputs = np.array(inputs_list, ndmin=2).T
		targets = np.array(targets_list, ndmin=2).T
		signal = []


		for layers in self.weights:
			pass
		# signal from input layer to out of hidden layer
		hidden_inputs = np.dot(self.wih, inputs)
		hidden_outputs = self.activation_function(hidden_inputs)

		# signal inside hidden layers
		hidden_signals = []
		hidden_signals_out = []
		for i in range(len(self.whls)):
			hidden_signals.append(np.dot(self.whls[i], hidden_outputs if i == 0 else hidden_signals_out[i - 1]))
			hidden_signals_out.append(self.activation_function(hidden_signals[i]))
		
		# signal that goes out of output layer
		#final_inputs = np.dot(self.who, hidden_outputs)
		final_inputs = np.dot(self.who, hidden_signals_out[-1])
		final_outputs = self.activation_function(final_inputs)

		# compute prediction and errors
		output_errors = targets - final_outputs
		hidden_errors = np.dot(self.who.T, output_errors)
		
		# update the weights for the links between the hidden and output layers
		self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), hidden_outputs.T)
		# update the weights for the links between the input and hidden layers
		self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), inputs.T)

	def predict(self, inputs_list):
		# convert inputs list to a transposed 2D array
		inputs = np.array(inputs_list, ndmin=2).T
		
		signal = inputs
		for layer in self.weights:
			signal = self.activation_function(np.dot(layer, signal))

		return signal

