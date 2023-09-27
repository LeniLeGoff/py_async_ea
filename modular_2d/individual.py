import sys
import os
import copy
import pickle

from deap import base

sys.path.append("tasks/ModularER_2D")
from Encodings import Network_Encoding as cppn
from Encodings import LSystem as lsystem
from Encodings import Direct_Encoding as de
import Tree as tr
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
        #self.genome = None
        self.novelty = Fitness()
        self.fitness = Fitness()
        self.age = 0
        self.config = None
        self.ctrl_log = None
        self.ctrl_pop = None
        self.learning_delta = 0
        self.nbr_eval = 0
        self.index = 0
        self.tree = None

    def __eq__(self, other):
        return self.index == other

    @staticmethod
    def clone(individual):
        self = copy.deepcopy(individual)
        return self

    @staticmethod
    def init_for_controller_opti(individual,config):
        self = Individual()
        self = Individual.clone(individual)
        self.create_tree(config)
        self.fitness = Fitness()
        self.random_controller(config)
        return self

    @staticmethod
    def random(config):
        # creates a random individual based on the encoding type
        self = Individual()
        self.index = Individual.static_index
        Individual.static_index+=1
        self.config = config
        moduleList = get_module_list()
        encoding = config["morphology"]["encoding"]
        if encoding == "cppn":
            self.genome = cppn.NN_enc(moduleList,config)
        elif encoding == "lsystem":
            self.genome = lsystem.LSystem(moduleList,config)
        elif encoding == "direct":
            self.genome = de.DirectEncoding(moduleList,config)
        
        self.tree_depth = int(self.config["morphology"]["max_depth"])
        self.genome.create(self.tree_depth)
        self.create_tree(config)
        self.random_controller(config)
        return self

    def random_controller(self,config):
        for node in self.tree.nodes:
            node.controller.mutate(1,0.1,self.tree.moduleList[node.type].angle)

    def create_tree(self,config):
        tree_depth = int(config["morphology"]["max_depth"])
        self.tree = self.genome.create(tree_depth)
        self.tree.create_children_lists()

    @staticmethod
    def mutate_morphology(ind,mutation_rate,mut_sigma):
        ind.genome.mutate(mutation_rate,mut_sigma)
        ind.age+=1 #increase age when mutated because it is an offspring
        return ind,

    #To correspond to DEAP API and be able to use EASimple for the controller mutation
    @staticmethod
    def mutate_controller(ind,mutation_rate,mut_sigma):
        for node in ind.tree.nodes:
            node.controller.mutate(mutation_rate,mut_sigma,ind.tree.moduleList[node.type].angle)
        return ind,
       
    def get_controller_genome(self):
        return [[node.controller.amplitude,node.controller.phase,node.controller.frequency,node.controller.offset] for node in self.tree.nodes] 

    @staticmethod
    def mutate(ind,morph_mutation_rate,morph_sigma,ctrl_mutation_rate,ctrl_sigma,config):
        ind.mutate_morphology(ind,morph_mutation_rate,morph_sigma)
        ind.create_tree(config)
        ind.mutate_controller(ind,ctrl_mutation_rate,ctrl_sigma)
        return ind,

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



