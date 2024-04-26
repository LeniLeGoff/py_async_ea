#! /usr/bin/python3
import sys
import pickle
from modular_2d import individual as mod_ind
import configparser as cp
import numpy as np
import ea_simple as ea
import gym
import time

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
    headless = False #config["simulation"].getboolean("headless")
    env_length = int(config["simulation"]["env_length"])

    env = getEnv()

    env.seed(0)
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

        time.sleep(0.1)

    individual.nbr_eval += 1
    return individual.fitness.values

if __name__ == '__main__':

    if len(sys.argv) != 5:
        print("usage:")
        print("\targ1 pickled population archive path")
        print("\targ2 config file")
        print("\targ3 individual index")
        print("\targ4 controller index")
        exit(1)
        
    archived_pop_file = sys.argv[1]

    config = cp.ConfigParser()
    config.read(sys.argv[2])

    ind_index = int(sys.argv[3])
    ctrl_index = int(sys.argv[4])

    population = load_population(archived_pop_file)
    base_ind = population[ind_index]
    del population
    base_ind.create_tree(config)
    base_ind.set_controller_parameters(base_ind.ctrl_pop[ctrl_index])
    print(base_ind.ctrl_pop[ctrl_index])
    print("best fitness",np.argmax(base_ind.ctrl_log.select("fitness")[-1]),max(base_ind.ctrl_log.select("fitness")[-1]))

    print("controller parameters:",base_ind.get_controller_genome())

    print("Final task performance:",evaluate(base_ind,config))

    
