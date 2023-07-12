#! /usr/bin/python3

import asynch_ea as asynch
import random as rd
import time
import numpy

from deap import base
from deap import creator
from deap import tools
from deap import benchmarks

def evaluate(ind):
	time.sleep(rd.randint(0,10))
	return benchmarks.rastrigin(ind)

def elitist_select(pop,size):
	sort_pop = pop
	sort_pop.sort(reverse=True,key=lambda p: p.fitness.values[0])
	return sort_pop[:size]

ind_size = 5

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", rd.random)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_float, n=ind_size)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1,indpb=1)	
toolbox.register("death_select", elitist_select)
toolbox.register("parents_select", tools.selTournament, tournsize=3)
toolbox.register("eval", evaluate)
toolbox.register("generate",asynch.generate)

if __name__ == '__main__':
	stats = tools.Statistics(key=lambda ind: ind.fitness.values)
	stats.register("avg", numpy.mean)
	stats.register("std", numpy.std)
	stats.register("min", numpy.min)
	stats.register("max", numpy.max)
	asynch_ea = asynch.AsynchEA(100,sync=0)
	pop = asynch_ea.init(toolbox)
	print("init finish")
	for i in range(1000):
		pop = asynch_ea.step(toolbox)
		print(stats.compile(pop))