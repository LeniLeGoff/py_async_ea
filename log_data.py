#! /usr/bin/python3

import matplotlib.pyplot as plt 
import numpy as np 

class Data: 
    def __init__(self,name="data"):
        self.data = []
        self.name = name
    def save(self, filename):
        with open(filename,'a') as file:
            for f in self.data[-1][:-1]:
                file.write(str(f) + ",")
            file.write(str(self.data[-1][-1]))
            file.write("\n")

    def load(self,filname):
        print("To be implemented")
        #TODO
        pass

    def add_data(self,fit):
        self.data.append(fit)

    def avg(self):
        return [np.average(d) for d in self.data]

    def median(self):
        return [np.median(d) for d in self.data]
    def percentile(self,perc):
        return [np.percentile(d,perc) for d in self.data]

class Plotter:
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot()    
    def plot(self,data):    
        self.ax.clear()
        self.ax.set_title(data.name)
        self.ax.plot(data.avg(), color = 'black')
        xs =[]
        for v in range(len(data.percentile(0))):
            xs.append(v)
        self.ax.fill_between(xs,data.percentile(0),data.percentile(100), color = 'black', alpha = 0.1)
        self.ax.fill_between(xs,data.percentile(25),data.percentile(75), color = 'black', alpha = 0.3)
        plt.pause(0.001)
        plt.ion()
