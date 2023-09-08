#! /usr/bin/python3
#! /usr/bin/python3
import sys
import pickle
from modular_2d import individual as mod_ind
import configparser as cp
from deap import base,tools
import numpy as np
import ea_simple as ea
import gym
import time
import multiprocessing as mp

def identity(a):
    return a

def load_population(filename):
    with open(filename,"rb") as file:
        population = pickle.load(file)
    return population

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
    headless = True #config["simulation"].getboolean("headless")
    env_length = int(config["simulation"]["env_length"])

    env = getEnv()
    #print(individual.get_controller_genome())
    env.seed(int(config["experiment"]["seed"]))
    env.reset(tree=individual.tree, module_list=individual.genome.moduleList)
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

        time.sleep(0.001)

    print("reward",individual.fitness.values)

    individual.nbr_eval += 1
    return individual.fitness.values

if __name__ == '__main__':

    if len(sys.argv) != 4:
        exit(1)
        
    archived_pop_file = sys.argv[1]

    config = cp.ConfigParser()
    config.read(sys.argv[2])

    ind_index = int(sys.argv[3])

    population = load_population(archived_pop_file)
    base_ind = population[ind_index]
    del population
    base_ind.create_tree(config)

    #print("base controller",base_ind.get_controller_genome())

    toolbox = base.Toolbox()
    toolbox.register("individual", mod_ind.Individual.init_for_controller_opti,individual=base_ind,config=config)
    toolbox.register("population", ea.seeded_init_repeat,list,toolbox.individual,[base_ind])
    toolbox.register("evaluate", evaluate,config=config)
    toolbox.register("mutate", mod_ind.Individual.mutate_controller, mutation_rate = float(config["controller"]["mut_rate"]),mut_sigma = float(config["controller"]["sigma"]))
    toolbox.register("select",tools.selBest)
    pool = mp.Pool()
    toolbox.register("map",pool.map)
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("max",np.max)
    stats.register("min",np.min)
    stats.register("fitness",identity)
    hof = tools.HallOfFame(1)
    pop = toolbox.population(int(config["controller"]["pop_size"]))
    pop, log = ea.steady_state_ea(pop,toolbox,cxpb=0,mutpb=1,ngen=int(config["controller"]["nbr_gen"]),stats=stats,halloffame=hof,verbose=True)
  
