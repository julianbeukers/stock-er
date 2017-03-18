# In this program, I attempt to build three different models that
# predict the prices of Apple Stock
# then plot them all on a graph to compare the results

# The steps would be:
# 1. Install Dependencies
# 2. Collect Dataset
# 3. Write Script
# 4. Analyze Graph

# Step 1: The four dependencies include:
# pip install csv - To read data from the stock prices
# pip install numpy - To perform calculations
# pip install scikit-learn - build a predictive model
# pip install matplotlib - plot datapoints on the model to analyze

# Step 2: Collecting dataset (Apple stocks from the past 30 days)
# Go to finance.google.com
# Look up NASDAQ:AAPL
# Select "Historical prices"
# Select "Download to spreadsheet"

# Step 3: Write Script
import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

# matplotlib is a graphical library therefore it will depend
# on a graphical backend
# If matplotlib does not plot a graph on the machine for some reason
# use the switch_backend option and try out a few different possible backends
# For this example, I use TkAgg, which is an Agg rendering to a Tk canvas (requires TkInter)
# For further reference: http://matplotlib.org/faq/usage_faq.html#gtk-and-cairo
plt.switch_backend('TkAgg')  

# Initialize two empty lists
dates = []
prices = []

def get_data(filename):
	'''
	Reads data from a file (aapl.csv) and adds data to
	the lists dates and prices
	'''
	with open(filename, 'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
		next(csvFileReader)	# skipping column names
		for row in csvFileReader:
			dates.append(int(row[0].split('-')[0]))
			prices.append(float(row[1]))

			print ('dates = ', dates)
			print ('prices =', prices)

	return

get_data("aapl.csv")
