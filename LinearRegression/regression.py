from numpy import *
from pylab import *
import matplotlib.pyplot as plt

def loadDataSet():
	x = []
	y = []

	with open("..\..\DataSet\REGRESSION\\2-dimension.txt") as fd:
		line = fd.readline()
		while line is not "":
			tmp = []
			line = line.strip()
			data_split = line.split("\t")

			x.append(float(data_split[1]))
			y.append(float(data_split[2]))

			line = fd.readline()

	length = len(x)

	x = array(x)
	y = array(y)

	x.shape = (length,1)
	y.shape = (length,1)

	# print(data_mat)
	return x,y

def plotData(x,y):
	scatter(x,y)

	show()

def standRegres(x,y):
	x_x_t = x * x.T

	print(linalg.det(x_x_t))
	print(linalg.det(x_x_t) == 0.0)

	# singular
	if linalg.det(x_x_t) == 0.0:
		print("This matrix is singular,cannot do inverse")
		return
	# not singular
	else:
		ws = x_x_t.I * (x*y.T)

	return ws

x,y = loadDataSet()
plotData(x,y)
# w = standRegres(x,y)