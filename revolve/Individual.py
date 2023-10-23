"""Individual class."""

from dataclasses import dataclass
from genotype import Genotype
from __future__ import annotations

from dataclasses import dataclass

import multineat
import numpy as np
import copy

from revolve2.ci_group.genotypes.cppnwin.modular_robot import BrainGenotypeCpg
from revolve2.ci_group.genotypes.cppnwin.modular_robot.v1 import BodyGenotypeV1
from revolve2.modular_robot import ModularRobot

# This is not clean!!! Basically, each experiment



class Fitness(base.Fitness):
    def __init__(self):
        self.weights=[1.0]

class Individual:
    static_index = 0
    _innov_db_body = multineat.InnovationDatabase()

    def __init__(self):
        #self.genome = None
        self.novelty = Fitness()
        self.fitness = Fitness()
        self.age = 0
        self.config = None
        self.ctrl_log = None
        self.ctrl_pop = None
        self.learning_delta = Fitness()
        self.nbr_eval = 0
        self.index = 0
        self.tree = None
        self.genotype = Genotype()
        self._innov_db_brain = multineat.InnovationDatabase()

    @staticmethod
    def clone(individual):
        self = copy.deepcopy(individual)
        return self

    @staticmethod
    def random(config):
        self = Individual()
        self.genotype = Genotype()
        self.genotype = self.genotype.random(self._innov_db_body,self._innov_db_brain)
        return self

     def random_controller(self,config):
         self.genotype = self.genotype.random(
            self._innov_db_bod,self._innov_db_brain, body=False, brain=True)

    @staticmethod
    def mutate_morphology(ind,mutation_rate,mut_sigma):
        ind.genotype = ind.genotype.mutate(
           ind._innov_db_bod,ind._innov_db_brain, body=True, brain=False)
        ind.age+=1 #increase age when mutated because it is an offspring
        self._innov_db_brain = multineat.InnovationDatabase()
        return ind,

    @staticmethod
    def mutate_controller(ind,mutation_rate,mut_sigma):
        ind.genotype = ind.genotype.mutate(
           ind._innov_db_bod,ind._innov_db_brain, body=False, brain=True)
        return ind,

    @staticmethod
    def mutate(ind,morph_mutation_rate,morph_sigma,ctrl_mutation_rate,ctrl_sigma,config):
        ind.mutate_morphology(ind,morph_mutation_rate,morph_sigma)
        ind.mutate_controller(ind,ctrl_mutation_rate,ctrl_sigma)
        return ind,

class Genotype(BodyGenotypeV1, BrainGenotypeCpg):

    def random(
        self,
        innov_db_body: multineat.InnovationDatabase,
        innov_db_brain: multineat.InnovationDatabase,
        body=True,
        brain=True,
        rng: np.random.Generator=np.random.default_rng(None),
        ) -> Genotype:
        if body:
            body = cls.random_body(innov_db_body, rng)
        else:
            body = self

        if brain:
            brain = cls.random_brain(innov_db_brain, rng)
        else:
            brain = self

        return Genotype(body=body.body, brain=brain.brain)

    def mutate(
        self,
        innov_db_body: multineat.InnovationDatabase,
        innov_db_brain: multineat.InnovationDatabase,
        body=True,
        brain=True,
        rng: np.random.Generator=np.random.default_rng(None),
        ) -> Genotype:
        if body:
            body = self.mutate_body(innov_db_body, rng)
        else:
            body = else

        if brain:
            brain = self.mutate_brain(innov_db_brain, rng)
        else:
            brain = self

        return Genotype(body=body.body, brain=brain.brain)

    @classmethod
    def crossover(
        cls,
        parent1: Genotype,
        parent2: Genotype,
        body=True,
        brain=True,
        rng: np.random.Generator=np.random.default_rng(None),
    ) -> Genotype:
        """
        Perform crossover between two genotypes.

        :param parent1: The first genotype.
        :param parent2: The second genotype.
        :param rng: Random number generator.
        :returns: A newly created genotype.
        """
        if body:
            body = cls.crossover_body(parent1, parent2, rng)
        else:
            body = self

        if brain:
            brain = cls.crossover_brain(parent1, parent2, rng)
        else:
            brain = self

        return Genotype(body=body.body, brain=brain.brain)

    def develop(self) -> ModularRobot:
        """
        Develop the genotype into a modular robot.

        :returns: The created robot.
        """
        body = self.develop_body()
        brain = self.develop_brain(body=body)
        return ModularRobot(body=body, brain=brain)

# @dataclass
# class Individual:
#     """An individual in a population."""
#
#     genotype: Genotype
#     fitness: float
