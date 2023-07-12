#! /usr/bin/python3
import sys
import copy
import pickle
import numpy as np
import gym
import configparser as cp

import log_fitnesses as lf
import asynch_ea as asynch
import novelty as nov
sys.path.append("tasks/ModularER_2D")
from Encodings import Network_Encoding as cppn
from Encodings import LSystem as lsystem
from DataAnalysis import DataAnalysis as da
import Tree as tr

import gym_rem2D as rem
from gym_rem2D.morph import SimpleModule, CircularModule2D

from deap import creator,base,tools,algorithms

env = None
def getEnv():
    global env
    if env is None:
        #env = M2D.Modular2D()
        # OpenAI code to register and call gym environment.
        env = gym.make("Modular2DLocomotion-v0")
    return env


def get_module_list():
    module_list = []
    for i in range(4):
        module_list.append(SimpleModule.Standard2D())
    for i in range(4):
        module_list.append(CircularModule2D.Circular2D())
    return module_list


fitness_data = lf.FitnessData()
plotter = lf.Plotter()

class Fitness(base.Fitness):
    def __init__(self):
        self.weights=[1.0]

class Individual:
    def __init__(self):
        self.genome = None
        self.novelty = Fitness()
        self.fitness = Fitness()
        self.age = 0
        self.config = None

    @staticmethod
    def clone(individual):
        self = copy.deepcopy(individual)
        return self

    @staticmethod
    def init_for_controller_opti(individual):
        self = Individual()
        self = Individual.clone(individual)
        self.random_controller()
        return self

    @staticmethod
    def random(config):
        # creates a random individual based on the encoding type
        self = Individual()
        self.config = config
        moduleList = get_module_list()
        self.genome = cppn.NN_enc(moduleList,self.config)
        
        self.tree_depth = int(self.config["morphology"]["max_depth"])
        self.genome.create(self.tree_depth)
        return self

    def random_controller(self):
        for mod in self.genome.moduleList:
            mod.mutate_controller(0.5,0.5)

    def mutate_morphology(morph_mutation_rate,mutation_rate,mut_sigma,self):
        self.genome.mutate(morph_mutation_rate,mutation_rate,mut_sigma)
        self.age+=1 #increase age when mutated because it is an offspring

    #To correspond to DEAP API and be able to use EASimple for the controller mutation
    @staticmethod
    def mutate_controller(ind,mutation_rate,mut_sigma):
        for mod in ind.genome.moduleList:
            mod.mutate_controller(mutation_rate,mut_sigma)
        return ind,
       

    def mutate(morph_mutation_rate,mutation_rate,mut_sigma,self):
        self.mutate_morphology(morph_mutation_rate,mutation_rate,mut_sigma)
        self, = Individual.mutate_controller(self,mutation_rate,mut_sigma)



def evaluate(individual, config):
    tree_depth = int(config["morphology"]["max_depth"])
    evaluation_steps = int(config["simulation"]["evaluation_steps"])
    interval = int(config["simulation"]["render_interval"])
    headless = bool(config["simulation"]["headless"])
    env_length = int(config["simulation"]["env_length"])

    env = getEnv()
    if tree_depth is None:
        try:
           tree_depth = individual.tree_depth
        except:
            raise Exception("Tree depth not defined in evaluation")
    tree = individual.genome.create(tree_depth)
    tree.create_children_lists()
    env.seed(4)
    env.reset(tree=tree, module_list=individual.genome.moduleList)
    it = 0
    for i in range(evaluation_steps):
        it+=1
        if it % interval == 0 or it == 1:
            if not headless:
                env.render()

        action = np.ones_like(env.action_space.sample())
        observation, reward, done, info  = env.step(action)
        if reward< -10:
            break
        elif reward > env_length:
            reward += (evaluation_steps-i)/evaluation_steps
            individual.fitness.values = [reward]
            break
        if reward > 0:
            individual.fitness.values = [reward]
    return individual.fitness.values

def learning_loop(individual,config):
    toolbox = base.Toolbox()
    toolbox.register("individual", Individual.init_for_controller_opti,individual=individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate,config=config)
    toolbox.register("mutate", Individual.mutate_controller, mutation_rate = float(config["controller"]["mut_rate"]),mut_sigma = float(config["controller"]["sigma"]))
    toolbox.register("select",tools.selTournament, tournsize = int(config["controller"]["tournament_size"]))
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    hof = tools.HallOfFame(1)
    pop = toolbox.population(int(config["controller"]["pop_size"]))
    algorithms.eaSimple(pop,toolbox,cxpb=0,mutpb=1,ngen=int(config["controller"]["nbr_gen"]),stats=stats,halloffame=hof,verbose=False)
    individual.genome = hof[0].genome
    return hof[0].fitness.values

def elitist_select(pop,size):
    sort_pop = pop
    sort_pop.sort(key=lambda p: p.fitness.values[0])
    return sort_pop[:size]

def age_select(pop,size):
    sort_pop = pop
    sort_pop.sort(key=lambda p: p.age)
    return sort_pop[:size]

def generate(parents,toolbox,size):
    offspring = toolbox.parent_select(parents, size)

    # deep copy of selected population
    offspring = list(map(toolbox.clone, offspring))
    for o in offspring:
        toolbox.mutate(o)
        # TODO only reset fitness to zero when mutation changes individual
        # Implement DEAP built in functionality
        o.fitness = Fitness()
    return offspring

def update_fitness_data(toolbox,population,gen,plot=False,save=False):
    fitness_values = [ind.fitness.values[0] for ind in population]
    fitness_data.add_fitnesses(fitness_values)
    if plot:
        plotter.plot(fitness_data)
    if save:
        n_gens=1
        if(gen%n_gens == 0):
            fitness_data.save("logs/fitnesses")
            pickle.dump(population,open("logs/pop_" + str(gen), "wb"))

def morphological_distance(ind1,ind2):
    tree1 = ind1.genome.create(ind1.tree_depth)
    tree1.create_children_lists()
    tree2 = ind2.genome.create(ind2.tree_depth)
    tree2.create_children_lists()
    return tr.Tree.distance(tree1,tree2)

def compute_novelty_scores(population,archive,config):
    for ind in population:
        dist = nov.distances(ind,population,archive,morphological_distance)
        ind.novelty.values = nov.sparsness(dist),
    for ind in population:
        archive = nov.update_archive(ind,ind.novelty.values[0],archive,novelty_thr=float(config["novelty"]["nov_thres"]),adding_prob=float(config["novelty"]["adding_prob"]),arch_size=int(config["novelty"]["arch_max_size"]))

def novelty_select(parents,size,archive,tournsize = 4):
    compute_novelty_scores(parents,archive)
    return tools.selTournament(parents,size,tournsize,fit_attr="novelty")

if __name__ == '__main__':
    config = cp.ConfigParser()
    if(len(sys.argv) == 2):
        config.read(sys.argv[1])
    else:
        config.read("modular_2d_walker.cfg")

    archive=[]

    toolbox = base.Toolbox()

    toolbox.register("individual", Individual.random,config=config)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("eval", learning_loop,config=config)
    toolbox.register("mutate", Individual.mutate_morphology, float(config["morphology"]["mut_rate"]),float(config["morphology"]["sigma"]))
    toolbox.register("parent_select",novelty_select, archive=archive ,tournsize = int(config["morphology"]["tournament_size"]))
    toolbox.register("death_select", elitist_select)
    toolbox.register("generate",generate)
    toolbox.register("extra",update_fitness_data,plot=bool(config["experiment"]["plot_prog"]),save=bool(config["experiment"]["save_logs"]))


    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    stats_nov = tools.Statistics(key=lambda ind: ind.novelty.values)
    stats_nov.register("avg", np.mean)
    stats_nov.register("std", np.std)
    stats_nov.register("min", np.min)
    stats_nov.register("max", np.max)


    asynch_ea = asynch.AsynchEA(int(config["morphology"]["pop_size"]),sync=float(config["morphology"]["synch"]))
    pop = asynch_ea.init(toolbox)
    print("init finish")
    for i in range(int(config["morphology"]["nbr_gen"])):
        pop = asynch_ea.step(toolbox)
        print("fitness - ",stats.compile(pop))
        print("novelty - ",stats_nov.compile(pop),"archive size :", len(archive))