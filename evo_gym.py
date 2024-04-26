#! /usr/bin/python3
import sys
import os
# import pickle
import numpy as np
import configparser as cp
import random as rd
import gym
# import multiprocessing as mp
import evogym.envs
from evo_gym import individual as eg_ind

#sys.path.append("task")
#from evogym import individual

from functools import partial

import log_data as ld
import asynch_ea as asynch
from asynch_ea import print


from deap import base,tools


fitness_data = ld.Data("fitness")
ind_index_data = ld.Data("indexes")
novelty_data = ld.Data("novelty")
#learning_trials = ld.Data("learning_trials")
learning_delta = ld.Data("learning_delta")
plot_fit = ld.Plotter()
plot_ld = ld.Plotter()

def evaluate(individual, config):
    evaluation_steps = int(config["simulation"]["evaluation_steps"])
    interval = int(config["simulation"]["render_interval"])
    headless = config["simulation"].getboolean("headless")
    env_length = int(config["simulation"]["env_length"])

    env = gym.make('Walker-v0', body=individual.structure)
    env.reset()
    it = 0
    for i in range(evaluation_steps):   
        action = env.action_space.sample() - 1 
        ob, reward, done, info = env.step(action)
        if not headless:
            env.render()

        if done:
            individual.fitness.values = [reward]
            env.reset()
            env.close()
            break
    individual.nbr_eval+=1
    return individual


def elitist_select(pop,size):
    sort_pop = pop
    sort_pop.sort(key=lambda p: p.fitness.values[0])
    return sort_pop[:size]

def age_select(pop,size):
    sort_pop = pop
    sort_pop.sort(key=lambda p: p.age)
    return sort_pop[:size]

def generate(parents,toolbox,size):
    print("tournament")
    selected_parents = toolbox.parent_select(parents, size)

    # deep copy of selected population
    offspring = list(map(toolbox.clone, selected_parents))
    for o in offspring:
        toolbox.mutate(o)
        o.nbr_eval = 0
        # TODO only reset fitness to zero when mutation changes individual
        # Implement DEAP built in functionality
        o.fitness = eg_ind.Fitness()
    return offspring

def update_data(toolbox,population,gen,log_folder,config,plot=False,save=False):
    fitness_values = [ind.fitness.values[0] for ind in population]
    fitness_data.add_data(fitness_values)
    indexes = [ind.index for ind in population]
    ind_index_data.add_data(indexes)
    goal_select = config["experiment"].getboolean("goal_select")
    if goal_select == False:
        novelty_scores = [ind.novelty.values[0] for ind in population]
        novelty_data.add_data(novelty_scores)
    learning_deltas = [ind.learning_delta for ind in population]
    learning_delta.add_data(learning_deltas)
    if save:
        n_gens=int(config["experiment"]["checkpoint_frequency"])
        fitness_data.save(log_folder + "/fitnesses")
        fitness_data.depop()
        ind_index_data.save(log_folder + "/indexes")
        ind_index_data.depop()
        if not config["controller"].getboolean("no_learning"):
            learning_delta.save(log_folder + "/learning_delta")
            learning_delta.depop()
        if goal_select == False:
            novelty_data.save(log_folder + "/novelty")
            novelty_data.depop()
        # if(gen%n_gens == 0):
        #     pickle.dump(population,open(log_folder + "/pop_" + str(gen), "wb"))
        #     if not config["controller"].getboolean("no_learning"):
        #         eg_ind.save_learning_ctrl_log(population,gen,log_folder)
        #         eg_ind.save_learning_ctrl_pop(population,gen,log_folder)



if __name__ == '__main__':    
    config = cp.ConfigParser()
    max_workers = 0
    if(len(sys.argv) == 3):
        config.read(sys.argv[1])
        max_workers = int(sys.argv[2])
    else:
        config.read("config/evo_gym.cfg")
        max_workers = int(sys.argv[1])

    config["experiment"]["max_workers"] = str(max_workers)
    
    log_folder = config["experiment"]["log_folder"]
    exp_name = config["experiment"]["name"]
    foldername = ld.create_log_folder(log_folder,exp_name)


    # goal_select = config["experiment"].getboolean("goal_select")
    elitist_survival = config["experiment"].getboolean("elitist_survival")

    #define seed
    seed = int(os.getrandom(5,flags=os.GRND_RANDOM).hex(),16)
    rd.seed(a=seed)
    config["experiment"]["seed"] = str(seed)

    no_learning = config["controller"].getboolean("no_learning")

    archive=[]

    toolbox = base.Toolbox()

    toolbox.register("individual", eg_ind.Individual.random,config=config)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("eval", evaluate,config=config)
   

    toolbox.register("mutate", eg_ind.Individual.mutate, mutation_rate=float(config["morphology"]["mut_rate"]),num_attempts=int(config["morphology"]["num_attempts"]))
    # if goal_select: #Do a goal-based selection
    toolbox.register("parent_select",tools.selTournament,tournsize=int(config["morphology"]["tournament_size"]))

    if elitist_survival: #Do an elitist survival: remove the worst individual in term of fitness
        toolbox.register("death_select", elitist_select)
    else: #Do an age based survival: remove the oldest individual
        toolbox.register("death_select", age_select)
    toolbox.register("generate",generate)
    def extra(toolbox,pop,iter):#no extra step
        pass
    toolbox.register("extra",extra)
    #toolbox.register("extra",update_data,log_folder=log_folder + "/" + foldername,config=config,plot=bool(config["experiment"].getboolean("plot_prog")),save=config["experiment"].getboolean("save_logs"))


    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    # if goal_select == False:
    #     stats_nov = tools.Statistics(key=lambda ind: ind.novelty.values)
    #     stats_nov.register("avg", np.mean)
    #     stats_nov.register("std", np.std)
    #     stats_nov.register("min", np.min)
    #     stats_nov.register("max", np.max)

    with open(log_folder + "/" + foldername + "/config.cfg",'w') as configfile :
        config.write(configfile)

    evaluations_budget = int(config["experiment"]["evaluations_budget"])
    
    print("setup finished")

    asynch_ea = asynch.AsynchEA(int(config["morphology"]["pop_size"]),max_workers,sync=float(config["morphology"]["synch"]))
    pop = asynch_ea.init(toolbox)
    print("init finish, running for", evaluations_budget, "evaluations")
    nbr_eval = 0
    for ind in pop:
        nbr_eval += ind.nbr_eval
    while nbr_eval < evaluations_budget:
        pop, new_inds = asynch_ea.step(toolbox)
        if len(new_inds) > 0:
            for ind in new_inds:
                nbr_eval += ind.nbr_eval
            print("fitness - ",stats.compile(pop))
            print("progress :",float(nbr_eval)/float(evaluations_budget)*100,"%")

    asynch_ea.terminate()
    print("EA has terminated normaly")