#! /usr/bin/python3

import matplotlib.pyplot as plt 
import numpy as np 

class FitnessData: 
    def __init__(self):
        self.fitnesses = []
    def save(self, filename):
        with open(filename,'a') as file:
            for f in self.fitnesses[-1][:-1]:
                file.write(str(f) + ",")
            file.write(str(self.fitnesses[-1][-1]))
            file.write("\n")
    def add_fitnesses(self,fit):
        self.fitnesses.append(fit)

    def avg(self):
        return [np.average(fits) for fits in self.fitnesses]

    def median(self):
        return [np.median(fits) for fits in self.fitnesses]
    def percentile(self,perc):
        return [np.percentile(fits,perc) for fits in self.fitnesses]

class Plotter:
	def __init__(self):
		self.fig = plt.figure()
		self.ax = self.fig.add_subplot()	
	def plot(self,fitness_data):	
		self.ax.clear()
		self.ax.plot(fitness_data.avg(), color = 'black')
		xs =[]
		for v in range(len(fitness_data.percentile(0))):
			xs.append(v)
		self.ax.fill_between(xs,fitness_data.percentile(0),fitness_data.percentile(100), color = 'black', alpha = 0.1)
		self.ax.fill_between(xs,fitness_data.percentile(25),fitness_data.percentile(75), color = 'black', alpha = 0.3)
		plt.pause(0.001)
		plt.ion()
