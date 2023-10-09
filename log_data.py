#! /usr/bin/python3

import matplotlib.pyplot as plt 
import numpy as np 
from datetime import datetime
import os

class Data: 
    def __init__(self,name="data"):
        self.data = []
        self.name = name
    def save(self, filename):
        if len(self.data) == 0:
            return
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

    def depop(self):
        if len(self.data) != 0:
            del self.data[0]

    def avg(self):
        return [np.average(d) for d in self.data]

    def median(self):
        return [np.median(d) for d in self.data]

    def percentile(self,perc):
        return [np.percentile(d,perc) for d in self.data]

def create_log_folder(log_folder,name):
    time = datetime.today()
    rd_nb = int(os.getrandom(3,flags=os.GRND_RANDOM).hex(),16)
    foldername = name + "_" + str(time.year)  \
                      + "_" + str(time.month) \
                      + "_" + str(time.day)   \
                      + "_" + str(time.hour)  \
                      + "_" + str(time.minute) \
                      + "_" + str(time.second) \
                      + "_" + str(time.microsecond) \
                      + "_" + str(rd_nb)
    if not os.path.exists(log_folder + "/" + foldername):
        os.makedirs(log_folder + "/" + foldername)
    return foldername

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
