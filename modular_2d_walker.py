#! /usr/bin/python3
import sys
import os
import time
import pickle
import numpy as np
import configparser as cp
import random as rd
import gym
import multiprocessing as mp

from functools import partial

from exception import LogExceptions
import log_data as ld
import asynch_ea as asynch
from asynch_ea import print
import ea_simple as ea

import tools.novelty as nov
from modular_2d import individual as mod_ind

from deap import base,tools


time_data = ld.Data("time_data")
fitness_data = ld.Data("fitness")
ind_index_data = ld.Data("indexes")
novelty_data = ld.Data("novelty")
learning_trials = ld.Data("learning_trials")
learning_delta = ld.Data("learning_delta")
morph_norm = ld.Data("morph_norms")
plot_fit = ld.Plotter()
plot_ld = ld.Plotter()

env = None
def getEnv():
    global env
    if env is None:
        #env = M2D.Modular2D()
       #OpenAI code to register and call gym environment.
        env = gym.make("Modular2DLocomotion-v0")
    return env

def evaluate(individual, config):
    evaluation_steps = int(config["simulation"]["evaluation_steps"])
    interval = int(config["simulation"]["render_interval"])
    headless = config["simulation"].getboolean("headless")
    env_length = int(config["simulation"]["env_length"])

    env = getEnv()
    if config["controller"].getboolean("no_learning"):
        individual.create_tree(config)
    env.seed(int(config["experiment"]["seed"]))
    env.reset(tree=individual.tree, module_list=individual.tree.moduleList)
    it = 0
    for i in range(evaluation_steps):
        it+=1
        if it % interval == 0 or it == 1:
            if not headless:
                env.render()

        action = [1,1,1,1] #not used here
        observation, reward, done, info  = env.step(action)
        if reward< -10:
            break
        elif reward > env_length:
            reward += (evaluation_steps-i)/evaluation_steps
            individual.fitness.values = [reward]
            break
        if reward > 0:
            individual.fitness.values = [reward]
    individual.nbr_eval += 1
    if config["controller"].getboolean("no_learning"):
        return individual
    return individual.fitness.values

def identity(a):
    return a

def learning_loop(individual,config):
    individual.create_tree(config)
    toolbox = base.Toolbox()
    toolbox.register("individual", mod_ind.Individual.init_for_controller_opti,individual=individual,config=config)
    toolbox.register("population", ea.seeded_init_repeat,list,toolbox.individual,[individual])
    toolbox.register("evaluate", LogExceptions(evaluate),config=config)
    toolbox.register("mutate", mod_ind.Individual.mutate_controller, mutation_rate = float(config["controller"]["mut_rate"]),mut_sigma = float(config["controller"]["sigma"]))
    toolbox.register("select",tools.selBest)
    pool = mp.Pool(processes=int(config["controller"]["pop_size"]))
    toolbox.register("map",pool.map)
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("max",np.max)
    stats.register("min",np.min)
    stats.register("fitness",identity)
    pop = toolbox.population(int(config["controller"]["pop_size"]))
    target_fit = float(config["controller"]["target_fit"])
    if(target_fit == -1):
        target_fit = None
    target_delta = float(config["controller"]["target_delta"])
    if target_delta == -1:
        target_delta = None
    pop, log, seed_fitness, best_ind = ea.steady_state_ea(pop,toolbox,cxpb=0,mutpb=1,ngen=int(config["controller"]["nbr_gen"]),stats=stats,verbose=False,min_fit=6,target_fit=target_fit,target_delta=target_delta)
    individual.genome = best_ind.genome
    individual.ctrl_log = log
    individual.ctrl_pop = [ind.get_controller_genome() for ind in pop]
    # print("pop",[ind.get_controller_genome() for ind in pop])
    individual.learning_delta.values = best_ind.fitness.values[0] - seed_fitness,
    individual.fitness = best_ind.fitness
    individual.nbr_eval = sum(log.select("nevals"))
    pool.terminate()
    pool.join()
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
        o.index= mod_ind.Individual.static_index
        o.nbr_eval = 0
        mod_ind.Individual.static_index+=1
        # TODO only reset fitness to zero when mutation changes individual
        # Implement DEAP built in functionality
        o.fitness = mod_ind.Fitness()
    return offspring

def update_data(toolbox,population,gen,log_folder,config,plot=False,save=False):
    time_data.add_data([time.time()])
    fitness_values = [ind.fitness.values[0] for ind in population]
    fitness_data.add_data(fitness_values)
    indexes = [ind.index for ind in population]
    ind_index_data.add_data(indexes)
    select_type = config["experiment"]["select_type"]
    if select_type == "novelty":
        novelty_scores = [ind.novelty.values[0] for ind in population]
        novelty_data.add_data(novelty_scores)
    if not config["controller"].getboolean("no_learning"):
        learning_delta.add_data([ind.learning_delta.values[0] for ind in population])
        learning_trials.add_data( [ind.nbr_eval for ind in population])
    morph_norm.add_data([ind.tree.norm() for ind in population])
    if plot:
        plot_fit.plot(fitness_data)
        plot_ld.plot(learning_delta)
    if save:
        time_data.save(log_folder + "/time_data")
        time_data.depop()
        n_gens=int(config["experiment"]["checkpoint_frequency"])
        fitness_data.save(log_folder + "/fitnesses")
        fitness_data.depop()
        ind_index_data.save(log_folder + "/indexes")
        ind_index_data.depop()
        morph_norm.save(log_folder + "/morph_norms")
        morph_norm.depop()
        if not config["controller"].getboolean("no_learning"):
            learning_delta.save(log_folder + "/learning_delta")
            learning_delta.depop()
            learning_trials.save(log_folder + "/learning_trials")
            learning_trials.depop()
        if select_type == "novelty":
            novelty_data.save(log_folder + "/novelty")
            novelty_data.depop()
        if(gen%n_gens == 0):
            pickle.dump(population,open(log_folder + "/pop_" + str(gen), "wb"))
            if not config["controller"].getboolean("no_learning"):
                mod_ind.save_learning_ctrl_log(population,gen,log_folder)
                mod_ind.save_learning_ctrl_pop(population,gen,log_folder)

def compute_novelty_scores(population,archive,config):
    for ind in population:
        if ind.novelty.valid:
            continue
        dist = nov.distances_parallel(population,archive,partial(mod_ind.morphological_distance,ind2=ind),cores=int(config["experiment"]["max_workers"]))
        ind.novelty.values = nov.sparsness(dist),
    for ind in population:
        archive = nov.update_archive(ind,ind.novelty.values[0],archive,novelty_thr=float(config["novelty"]["nov_thres"]),adding_prob=float(config["novelty"]["adding_prob"]),arch_size=int(config["novelty"]["arch_max_size"]))

def novelty_select(parents,size,archive,config):
    compute_novelty_scores(parents,archive,config)
    return tools.selTournament(parents,size,int(config["morphology"]["tournament_size"]),fit_attr="novelty")



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
    foldername = ld.create_log_folder(log_folder,exp_name)


    select_type = config["experiment"]["select_type"]
    elitist_survival = config["experiment"].getboolean("elitist_survival")

    #define seed
    seed = int(os.getrandom(5,flags=os.GRND_RANDOM).hex(),16)
    rd.seed(a=seed)
    config["experiment"]["seed"] = str(seed)

    no_learning = config["controller"].getboolean("no_learning")

    archive=[]


    toolbox = base.Toolbox()
    toolbox.register("individual", mod_ind.Individual.random,config=config)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    if no_learning:
        toolbox.register("eval", evaluate,config=config)
        toolbox.register("mutate", mod_ind.Individual.mutate, \
                            morph_mutation_rate=float(config["morphology"]["mut_rate"]),\
                            morph_sigma=float(config["morphology"]["sigma"]),\
                            ctrl_mutation_rate=float(config["controller"]["mut_rate"]),\
                            ctrl_sigma=float(config["controller"]["sigma"]), \
                            config=config)
    else:
        toolbox.register("eval", learning_loop,config=config)
        toolbox.register("mutate", mod_ind.Individual.mutate_morphology,\
                            mutation_rate=float(config["morphology"]["mut_rate"]),\
                            mut_sigma=float(config["morphology"]["sigma"]))


    if select_type == "goal": #Do a goal-based selection
        toolbox.register("parent_select",tools.selTournament,tournsize=int(config["morphology"]["tournament_size"]))
    elif select_type == "novelty": #Do a novelty selection.
        toolbox.register("parent_select",novelty_select, archive=archive ,config=config)
    elif select_type == "delta":
        toolbox.register("parent_select",tools.selTournament,tournsize=int(config["morphology"]["tournament_size"]),fit_attr="learning_delta")
    if elitist_survival: #Do an elitist survival: remove the worst individual in term of fitness
        toolbox.register("death_select", elitist_select)
    else: #Do an age based survival: remove the oldest individual
        toolbox.register("death_select", age_select)
    toolbox.register("generate",generate)
    toolbox.register("extra",update_data,log_folder=log_folder + "/" + foldername,config=config,plot=bool(config["experiment"].getboolean("plot_prog")),save=config["experiment"].getboolean("save_logs"))


    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    if select_type == "novelty":
        stats_nov = tools.Statistics(key=lambda ind: ind.novelty.values)
        stats_nov.register("avg", np.mean)
        stats_nov.register("std", np.std)
        stats_nov.register("min", np.min)
        stats_nov.register("max", np.max)

    with open(log_folder + "/" + foldername + "/config.cfg",'w') as configfile :
        config.write(configfile)

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
            for ind in new_inds:
                nbr_eval += ind.nbr_eval
            print("fitness - ",stats.compile(pop))
            if select_type == "novelty":
                print("novelty - ",stats_nov.compile(pop),"archive size :", len(archive))
            print("progress :",float(nbr_eval)/float(evaluations_budget)*100,"%")

    asynch_ea.terminate()
    print("EA has terminated normaly")
