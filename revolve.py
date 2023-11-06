import os
import sys
import configparser as cp
import random as rd
import numpy as np

import revolve as rev
import asynch_ea as asynch
from deap import base,tools


def select(pop,size):
    sort_pop = pop
    sort_pop.sort(key=lambda p: p.fitness.values[0])
    return sort_pop[:size]

def generate(parents,toolbox,size):
    print("tournament")
    selected_parents = toolbox.parent_select(parents, size)

    # deep copy of selected population
    offspring = list(map(toolbox.clone, selected_parents))
    for o in offspring:
        toolbox.mutate(o)
        o.index= rev.Individual.static_index
        o.nbr_eval = 0
        rev.Individual.static_index+=1
        # TODO only reset fitness to zero when mutation changes individual
        # Implement DEAP built in functionality
        o.fitness = rev.Fitness()

    return offspring

def no_extra():
    pass

if __name__ == '__main__':    
    config = cp.ConfigParser()
    max_workers = 0
    if(len(sys.argv) == 3):
        config.read(sys.argv[1])
        max_workers = int(sys.argv[2])
    else:
        config.read("modular_2d_walker.cfg")
        max_workers = int(sys.argv[1])

    config["experiment"]["max_workers"] = str(max_workers)
    
    log_folder = config["experiment"]["log_folder"]
    exp_name = config["experiment"]["name"]
    #foldername = ld.create_log_folder(log_folder,exp_name)


    select_type = config["experiment"]["select_type"]
    elitist_survival = config["experiment"].getboolean("elitist_survival")

    #define seed
    seed = int(os.getrandom(5,flags=os.GRND_RANDOM).hex(),16)
    rd.seed(a=seed)
    config["experiment"]["seed"] = str(seed)

    no_learning = config["controller"].getboolean("no_learning")

    archive=[]


    toolbox = base.Toolbox()
    toolbox.register("individual", rev.Individual.random,config=config)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("eval", rev.Evaluator.evaluate,config=config)
    toolbox.register("mutate", rev.Individual.mutate, \
                            morph_mutation_rate=float(config["morphology"]["mut_rate"]),\
                            morph_sigma=float(config["morphology"]["sigma"]),\
                            ctrl_mutation_rate=float(config["controller"]["mut_rate"]),\
                            ctrl_sigma=float(config["controller"]["sigma"]), \
                            config=config)
    toolbox.register("parent_select",tools.selTournament,tournsize=int(config["morphology"]["tournament_size"]))
    toolbox.register("death_select", select)
    toolbox.register("generate",generate)
    toolbox.register("extra",no_extra)

    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)


   # with open(log_folder + "/" + foldername + "/config.cfg",'w') as configfile :
   #    config.write(configfile)

    evaluations_budget = int(config["experiment"]["evaluations_budget"])
    
    asynch_ea = asynch.AsynchEA(int(config["morphology"]["pop_size"]),max_workers,sync=float(config["morphology"]["synch"]))
    pop = asynch_ea.init(toolbox)
    print("init finish, running for", evaluations_budget, "evaluations")
    nbr_eval = 0
    for ind in pop:
        nbr_eval += ind.nbr_eval
    while nbr_eval < evaluations_budget:
        pop, new_inds = asynch_ea.step(toolbox)

        if len(new_inds) > 0:
            new_idx = [ind.index for ind in new_inds]
            for ind in new_inds:
                nbr_eval += ind.nbr_eval
            print("fitness - ",stats.compile(pop))
            print("nbr eval",nbr_eval)
            print("progress :",float(nbr_eval)/float(evaluations_budget)*100,"%")

    asynch_ea.terminate()
    print("EA has terminated normaly")
