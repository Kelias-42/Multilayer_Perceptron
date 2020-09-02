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
		self.softmax = lambda x: np.exp(x) / np.sum(np.exp(x), axis=0)

	def train(self, inputs_list, targets_list):
		# convert inputs and targets lists to a transposed 2D array
		inputs = np.array(inputs_list, ndmin=2).T
		targets = np.array(targets_list, ndmin=2).T
		signal = []


		# signal from input layer to out of hidden layer
		#hidden_inputs = np.dot(self.wih, inputs)
		#hidden_outputs = self.activation_function(hidden_inputs)

		# signal inside hidden layers
		hidden_signals = []
		hidden_signals_out = []
		for i in range(len(self.weights)):
			hidden_signals.append(np.dot(self.weights[i], inputs if i == 0 else hidden_signals_out[i - 1]))
			hidden_signals_out.append(self.activation_function(hidden_signals[i]))
		hidden_signals_out.insert(0, inputs)
		# signal that goes out of output layer
		#final_inputs = np.dot(self.who, hidden_outputs)
		#final_inputs = np.dot(self.who, hidden_signals_out[-1])
		#final_outputs = self.activation_function(final_inputs)

		
		output_errors = targets - hidden_signals_out[-1]
		for i in range(1, len(self.weights) + 1):
			error = output_errors if i == 1 else np.dot(self.weights[-(i-1)].T, error)
			self.weights[-i] += self.lr * np.dot((error * hidden_signals_out[-i] * (1.0 - hidden_signals_out[-i])), hidden_signals_out[-(i+1)].T)
			#print('oui')
		# compute prediction and errors
		#hidden_errors = np.dot(self.who.T, output_errors)
		
		# update the weights for the links between the hidden and output layers
		#self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), hidden_outputs.T)
		# update the weights for the links between the input and hidden layers
		#self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), inputs.T)

	def predict(self, inputs_list):
		# convert inputs list to a transposed 2D array
		inputs = np.array(inputs_list, ndmin=2).T
		
		signal = inputs
		for layer in self.weights:
			if np.array_equal(layer, self.weights[len(self.weights) - 1]):
				signal = self.softmax(np.dot(layer, signal))
			else:
				signal = self.activation_function(np.dot(layer, signal))
		return signal

