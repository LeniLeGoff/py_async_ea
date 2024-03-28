import sys
import os
import copy
import pickle

from deap import base
import numpy as np

sys.path.append("tasks")
from evogym import is_connected, has_actuator, get_full_connectivity, draw, get_uniform, EvoWorld, EvoSim, EvoViewer, sample_robot


class Fitness(base.Fitness):
    def __init__(self):
        self.weights=[1.0]

class Individual:
    static_index = 0
    def __init__(self,robot_shape=(5,5)):
        self.genome = None
        self.novelty = Fitness()
        self.fitness = Fitness()
        self.age = 0
        self.config = None
        self.ctrl_log = None
        self.ctrl_pop = None
        self.learning_delta = Fitness()
        self.nbr_eval = 0
        self.index = 0
        self.shape = robot_shape
        self.structure = np.zeros(robot_shape)
        self.connections = get_full_connectivity(self.structure)


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
        self.structure, self.connections = sample_robot((5, 5))
        return self

    

    def mutate(self,mutation_rate,num_attempts):
        pd = get_uniform(5)  
        pd[0] = 0.6 #it is 3X more likely for a cell to become empty
        child = Individual()
        child.clone(self)
        structure = child.structure
        connections = child.connections
        # iterate until valid robot found
        for n in range(num_attempts):
            # for every cell there is mutation_rate% chance of mutation
            for i in range(structure.shape[0]):
                for j in range(structure.shape[1]):
                    mutation = [mutation_rate, 1-mutation_rate]
                    if draw(mutation) == 0: # mutation
                        structure[i][j] = draw(pd)
        
            if is_connected(structure) and has_actuator(structure):
                child.structure = structure
                child.connections = connections
                return child,

        # no valid robot found after num_attempts
        return None,