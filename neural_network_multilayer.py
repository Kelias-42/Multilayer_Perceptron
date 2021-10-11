import math
import numpy as np
import pandas as pd

class neural_network:

	def __init__(self, in_neuron_count, out_neuron_count, learning_rate, hidden_layer_count, hidden_neuron_count):
		self.in_neuron_count = in_neuron_count
		self.out_neuron_count = out_neuron_count
		self.learning_rate = learning_rate
		self.hidden_layer_count = hidden_layer_count
		self.hidden_neuron_count = hidden_neuron_count

		self.weights = [np.random.normal(0, 1 / math.sqrt(self.hidden_neuron_count), (self.hidden_neuron_count, self.in_neuron_count))]
		for	_ in range(self.hidden_layer_count - 1):
			self.weights.append(np.random.normal(0, 1 / math.sqrt(self.hidden_neuron_count), (self.hidden_neuron_count, self.hidden_neuron_count)))
		self.weights.append(np.random.normal(0, 1 / math.sqrt(self.out_neuron_count), (self.out_neuron_count, self.hidden_neuron_count)))

		self.activation_function = lambda x : 1 / (1 + np.exp(-x))
		self.softmax = lambda x : np.exp(x) / np.sum(np.exp(x), axis=0)

	def train(self, inputs_list, targets_list):
		# convert inputs and targets lists to a transposed 2D array
		inputs = np.array(inputs_list, ndmin=2).T
		targets = np.array(targets_list, ndmin=2).T

		# signal inside hidden layers
		hidden_signals = []
		hidden_signals_out = []
		for i in range(len(self.weights)):
			hidden_signals.append(np.dot(self.weights[i], inputs if i == 0 else hidden_signals_out[i - 1]))
			hidden_signals_out.append(self.activation_function(hidden_signals[i]))
		hidden_signals_out.insert(0, inputs)
		
		# compute and propagate error
		output_errors = targets - hidden_signals_out[-1]
		for i in range(1, len(self.weights) + 1):
			error = output_errors if i == 1 else np.dot(self.weights[-(i - 1)].T, error)
			self.weights[-i] += self.learning_rate * np.dot((error * hidden_signals_out[-i] * (1.0 - hidden_signals_out[-i])), hidden_signals_out[-(i+1)].T)

	def predict(self, inputs_list):
		# convert inputs list to a transposed 2D array
		inputs = np.array(inputs_list, ndmin=2).T
		
		signal = inputs
		for i, layer in enumerate(self.weights):
			if i == len(self.weights) - 1:
				signal = self.softmax(np.dot(layer, signal))
			else:
				signal = self.activation_function(np.dot(layer, signal))
		return signal