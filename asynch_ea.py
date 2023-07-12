#! /usr/bin/python3

import random as rd

from scoop import futures
from deap import algorithms

def generate(parents,toolbox,size):
        ''' generate takes a list already evaluated parents and the size of the offspring to generate'''

        #select the individuals from the offspring will be produced
        offspring = toolbox.parents_select(parents, size)
    
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        offspring = algorithms.varAnd(offspring, toolbox, 0.1, 0.2)

        return offspring

class AsynchEA:
    def __init__(self,pop_size,sync=0):
        self.pop_size = pop_size
        self.nbr_ind_to_wait = int(pop_size)*sync
        if sync == 0:
            self.nbr_ind_to_wait = 1
        print("number ind to wait:", self.nbr_ind_to_wait)
        self.parents = []
        self.pop = []
        self.workers = []
        self.fitnesses = []
        self.iteration = 0

    def remove(self,select):
        dead = select(self.parents,len(self.parents)-self.pop_size)
        for ind in dead:
            self.parents.remove(ind) 

    def worker_callback(self,worker):
        for w in self.workers:
            if w[0] == worker:
                self.fitnesses.append([worker.result(),w[1]])
                self.workers.remove(w)

    def asynch_map(self,eval):
        for ind in self.pop:
            is_new_ind = True
            for w in self.workers:
                if w[1] == ind:
                    is_new_ind = False
                    break
            if is_new_ind:
                worker = futures.submit(eval,ind)
                worker.add_done_callback(self.worker_callback)
                self.workers.append([worker,ind])

    def update(self,eval):
        # Evaluate the individuals with asynch map. Evaluate as to return a ref to the ind at the end
        self.asynch_map(eval)
        jobs = [w[0] for w in self.workers]
        nb_completed = 0
        for completed in futures.as_completed(jobs):
            nb_completed+=1
            if nb_completed == self.nbr_ind_to_wait:
                break
        print(len(self.fitnesses))
        if len(self.fitnesses) >= self.nbr_ind_to_wait:
            for fit in self.fitnesses:
                for ind in self.pop:
                   if ind == fit[1]:
                        ind.fitness.values = fit[0]
            self.fitnesses = []

    
        new_parents = [ind for ind in self.pop if ind.fitness.valid]
        for ind in new_parents:
            self.pop.remove(ind)

        return new_parents

    def init(self,toolbox):
        #initialisation
        self.pop = toolbox.population(self.pop_size)
        print(len(self.pop))
        while len(self.parents) < self.pop_size:
            new_par = self.update(toolbox.eval)
            self.parents = self.parents + new_par
        assert(len(self.pop) == 0)
        self.pop = toolbox.generate(self.parents,toolbox,self.pop_size)
        toolbox.extra(toolbox,self.parents,self.iteration)
        return self.parents

    def step(self,toolbox):
        #update - evaluation
        new_par = self.update(toolbox.eval)
        self.parents = self.parents + new_par

        #survival
        self.remove(toolbox.death_select)
        
        #selection - mutation - crossover
        offspring = toolbox.generate(self.parents,toolbox,self.pop_size - len(self.pop))
        self.pop = self.pop + offspring
        
        self.iteration += 1

        toolbox.extra(toolbox,self.parents,self.iteration)

        return self.parents

