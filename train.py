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
	#print(data)
	#plt.show()	

input_neurons = 3
output_neurons = 3
learning_rate = 0.01
n_hidden_layers = 2
n_hidden_neurons = 3

if __name__ == "__main__":
	n_net = neural_network(input_neurons, output_neurons, learning_rate, n_hidden_layers, n_hidden_neurons)
