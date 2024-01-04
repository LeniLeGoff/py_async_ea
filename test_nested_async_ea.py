#! /usr/bin/python3

import asynch_ea as asynch
import random as rd
import time
import numpy
import sys

from scoop import futures

from deap import base
from deap import creator
from deap import tools
from deap import benchmarks
from deap import cma

def evaluate(ind):
	#time.sleep(rd.random()*2)
	return benchmarks.rastrigin(ind) 

def map(func,list_):
    workers = []
    for l in list_:
        worker = futures.submit(func,l)
        workers.append(worker)
    return [completed.result() for completed in futures.as_completed(workers)]


def cmaes_evaluate(individual):
    toolbox = base.Toolbox()
    strategy = cma.Strategy(centroid=[5.0]*ind_size, sigma=5.0, lambda_=5*ind_size) 
    toolbox.register("generate", strategy.generate, creator.CMAInd)
    toolbox.register("update", strategy.update)
    toolbox.register("cma_eval",benchmarks.rastrigin)

    #toolbox.register("map",pool.map)
    pop = toolbox.generate()
    #pop = [ind + i for i in pop]
    fitnesses = toolbox.map(toolbox.cma_eval, pop)
    for ind, fit in zip(pop, fitnesses):
        print(fit,flush=True)
        ind.fitness.values = fit
    toolbox.update(pop)
    individual.fitness.values = min([ind.fitness.values for ind in pop])
    return individual

def elitist_select(pop,size):
    sort_pop = pop
    sort_pop.sort(reverse=True,key=lambda p: p.fitness.values[0])
    return sort_pop[:size]
if __name__ == '__main__':
    ind_size = 5
    asynch_ea = asynch.AsynchEA(10,int(sys.argv[1]))

    creator.create("CMAFit", base.Fitness, weights=(-1.0,))
    creator.create("CMAInd", list, fitness=creator.CMAFit)

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
    toolbox.register("eval", cmaes_evaluate)
    toolbox.register("generate",asynch.generate)
    def extra(toolbox,pop,iter):#no extra step
        pass
    toolbox.register("extra",extra)


    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    pop = asynch_ea.init(toolbox)
    print("init finish")
    for i in range(100):
        pop, new_ind = asynch_ea.step(toolbox)
        print(stats.compile(pop))
