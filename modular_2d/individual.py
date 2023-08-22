import sys
import os
import copy
import pickle

from deap import base

sys.path.append("tasks/ModularER_2D")
from Encodings import Network_Encoding as cppn
from Encodings import LSystem as lsystem
import Tree as tr
import gym_rem2D as rem
from gym_rem2D.morph import simple_module, circular_module

def get_module_list():
    module_list = []
    for i in range(4):
        module_list.append(simple_module.Standard2D())
    for i in range(4):
        module_list.append(circular_module.Circular2D())
    return module_list

class Fitness(base.Fitness):
    def __init__(self):
        self.weights=[1.0]

class Individual:
    static_index = 0
    def __init__(self):
        self.genome = None
        self.novelty = Fitness()
        self.fitness = Fitness()
        self.age = 0
        self.config = None
        self.ctrl_log = None
        self.ctrl_pop = None
        self.learning_delta = 0
        self.nbr_eval = 0
        self.index = 0

    def __eq__(self, other):
        return self.index == other

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
        self.index = Individual.static_index
        Individual.static_index+=1
        self.config = config
        moduleList = get_module_list()
        self.genome = cppn.NN_enc(moduleList,self.config)
        
        self.tree_depth = int(self.config["morphology"]["max_depth"])
        self.genome.create(self.tree_depth)
        return self

    def random_controller(self):
        for mod in self.genome.moduleList:
            mod.mutate_controller(0.5,0.5)

    @staticmethod
    def mutate_morphology(ind,mutation_rate,mut_sigma):
        ind.genome.mutate(mutation_rate,mut_sigma)
        ind.age+=1 #increase age when mutated because it is an offspring
        return ind,

    #To correspond to DEAP API and be able to use EASimple for the controller mutation
    @staticmethod
    def mutate_controller(ind,mutation_rate,mut_sigma):
        for mod in ind.genome.moduleList:
            mod.mutate_controller(mutation_rate,mut_sigma)
        return ind,
       
    def get_controller_genome(self):
        return [[mod.controller.amplitude,mod.controller.phase,mod.controller.frequency,mod.controller.offset] for mod in self.genome.moduleList] 

    def mutate(morph_mutation_rate,mutation_rate,mut_sigma,self):
        self, = Individual.mutate_morphology(self,morph_mutation_rate,mutation_rate,mut_sigma)
        self, = Individual.mutate_controller(self,mutation_rate,mut_sigma)

    def ctrl_log_to_string(self):
        ctrl_fits = self.ctrl_log.select("fitness")
        str_fit = str()
        for fit in ctrl_fits:
            str_fit += str(fit[0][0])
            for f in fit[0:]:
                str_fit += "," + str(f[0])
            str_fit += "\n"
        return str_fit

def morphological_distance(ind1,ind2):
    tree1 = ind1.genome.create(ind1.tree_depth)
    tree1.create_children_lists()
    tree2 = ind2.genome.create(ind2.tree_depth)
    tree2.create_children_lists()
    return tr.Tree.distance(tree1,tree2)

def save_learning_ctrl_log(pop,gen,log_folder):
    foldername = log_folder + "/controller_logs_" + str(gen)
    if not os.path.exists(foldername):
        os.makedirs(foldername)

    for ind,index in zip(pop,range(len(pop))):
        with open(foldername + "/ctrl_log_" + str(index),'w') as file:
            file.write(ind.ctrl_log_to_string())


def save_learning_ctrl_pop(pop,gen,log_folder):
    foldername = log_folder + "/controller_logs_" + str(gen)
    if not os.path.exists(foldername):
        os.makedirs(foldername)

    for ind,index in zip(pop,range(len(pop))):
        pickle.dump(ind.ctrl_pop,open(foldername + "/ctrl_pop_" + str(index), "wb"))



