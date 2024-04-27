#! /usr/bin/python3
import sys
from nestable_pool import NestablePool
from deap import algorithms
from exception import LogExceptions
import builtins
from typing import TypeVar, Callable

TIndividual = TypeVar("TIndividual")

def custom_print(*objects):
    string = "" 
    for o in objects:
        string += str(o) + " "
    builtins.print(string,flush=True)

def generate(parents,toolbox,size):
        ''' generate takes a list already evaluated parents and the size of the offspring to generate'''

        #select the individuals from the offspring will be produced
        selected_parents = toolbox.parents_select(parents, size)
    
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, selected_parents))

        # Apply crossover and mutation on the offspring
        offspring = algorithms.varAnd(offspring, toolbox, 0.1, 0.2)

        return offspring

class AsynchEA:
    def __init__(self,pop_size,nb_workers,sync=0) -> None:
        self.pop_size = pop_size
        self.nbr_ind_to_wait = int(pop_size)*sync
        if sync == 0:
            self.nbr_ind_to_wait = 1
        custom_print("number ind to wait:", self.nbr_ind_to_wait)
        self.iteration = 0
        self.max_workers = nb_workers
        self.pool = NestablePool(processes=nb_workers)#,maxtasksperchild=100)
        self.in_evaluation: list[TIndividual] = []
        self.parents: list[TIndividual] = []
        self.pop: list[TIndividual] = []
        self.evaluated_ind: list[TIndividual] = []
        self.workers_failed = False

    def remove(self, select: Callable) -> None:
        dead = select(self.parents,len(self.parents)-self.pop_size)
        for ind in dead:
            self.parents.remove(ind) 
            del ind

    def worker_callback(self, results: TIndividual) -> None:
        if results is None:
            self.workers_failed = True
        else:
            self.evaluated_ind.append(results)
            to_remove = [elem for elem in self.in_evaluation if elem.uuid == results.uuid]
            for individual in to_remove:
                self.in_evaluation.remove(individual)

    def asynch_map(self, eval: Callable) -> None:
        for ind in self.pop:
            is_new_ind = False if ind in self.in_evaluation or ind in self.evaluated_ind else True
            if is_new_ind and len(self.in_evaluation) < self.max_workers:
                #print("ind",ind.index,"send to evaluation")
                self.in_evaluation.append(ind)
                self.pool.apply_async(LogExceptions(eval),(ind,),callback=self.worker_callback)
            if len(self.in_evaluation) >= self.max_workers:
                break

    #sequential execution. Use only for debugging
    def seq_map(self, eval: Callable) -> None:
        for ind in self.pop:
            results = eval(ind)
            self.evaluated_ind.append(results)
            
    def update(self, eval: Callable) -> list[TIndividual]:
        # Evaluate the individuals with asynch map. Evaluate as to return a ref to the ind at the end
        #self.seq_map(eval)
        self.asynch_map(eval)
        if self.workers_failed:
            self.terminate()
            sys.exit("Exiting because of workers crash")

        if len(self.evaluated_ind) >= self.nbr_ind_to_wait:
            custom_print("number individual evaluated",len(self.evaluated_ind))
            for e_ind in self.evaluated_ind:
                for i, candidate in enumerate(self.pop):
                    if candidate.uuid == e_ind.uuid:
                        self.pop[i] = e_ind
                        break
            del self.evaluated_ind
            self.evaluated_ind = []
    
        new_parents = [ind for ind in self.pop if ind.fitness.valid]
        if len(new_parents) > 0:
            custom_print("new_parents", [ind.index for ind in new_parents])
        for ind in new_parents:
            self.pop.remove(ind)
        return new_parents

    def init(self,toolbox) -> list[TIndividual]:
        #initialisation
        self.pop = toolbox.population(self.pop_size)
        while len(self.parents) < self.pop_size:
            new_par = self.update(toolbox.eval)
            self.parents = self.parents + new_par
            if len(new_par) > 0:
                custom_print("init progress:", float(len(self.parents)) / float(self.pop_size) * 100, "%")
        assert len(self.pop) == 0, "Error: population contains robots still."
        self.pop = toolbox.generate(self.parents,toolbox,self.pop_size)
        custom_print(len(self.pop))
        toolbox.extra(toolbox,self.parents,self.iteration)
        return self.parents

    def step(self,toolbox) -> tuple[list[TIndividual], list[TIndividual]]:
        #update - evaluation
        new_par = self.update(toolbox.eval)
        if len(new_par) > 0:
            self.parents = self.parents + new_par

            #survival
            self.remove(toolbox.death_select)
        
            #selection - mutation - crossover
            offspring = toolbox.generate(self.parents,toolbox,self.pop_size - len(self.pop))
            self.pop = self.pop + offspring
        
            self.iteration += 1

            toolbox.extra(toolbox,self.parents,self.iteration)

        return self.parents, new_par

    def terminate(self) -> None:
        self.pool.terminate()
        self.pool.join()
