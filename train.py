import sys
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

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
	
if __name__ == "__main__":
	data = get_data()
	columns = [f"p{i}" for i in range(1, 32)]
	columns.insert(1, "state")
	data.columns = columns
	data["state"].replace(states, inplace=True)
	data = data.select_dtypes('number')
	data["state"].replace(states_rev, inplace=True)
	for i in range(1,20):
		data.drop(f"p{i}", axis=1, inplace=True)
	print(data)
	sb.pairplot(data, hue='state', markers='.')
	plt.show()