"""Individual class."""
from __future__ import annotations
from deap import base
import multineat
import numpy as np
import copy
from ._genotype import Genotype
from dataclasses import dataclass, field
from revolve2.ci_group.genotypes.cppnwin.modular_robot import BrainGenotypeCpg
from revolve2.ci_group.genotypes.cppnwin.modular_robot.v1 import BodyGenotypeV1
from revolve2.modular_robot import ModularRobot
from revolve2.experimentation.rng import make_rng_time_seed
class Fitness(base.Fitness):
    def __init__(self):
        self.weights=[1.0]

@dataclass
class Individual:
    """A Revolve2 Individual"""
    """Counter Variables"""
    static_index: int = field(default=0)
    index: int = field(default=0)
    nbr_eval: int = field(default=0)

    """Individual specific values."""
    age: int = field(default=0)
    genotype: Genotype = field(default_factory=lambda: Genotype())

    """Evaluation Measures."""
    novelty: Fitness = field(default_factory=lambda: Fitness())
    fitness: Fitness = field(default_factory=lambda: Fitness())
    learning_delta: Fitness = field(default_factory=lambda: Fitness())

    """Auxiliary varibables."""
    rng: np.random.Generator = field(default_factory=lambda: make_rng_time_seed())
    #TODO: This should not be part of the individual.
    _innov_db_body: multineat.InnovationDatabase = field(default_factory=lambda: multineat.InnovationDatabase())
    _innov_db_brain: multineat.InnovationDatabase = field(default_factory=lambda: multineat.InnovationDatabase())

    """Not sure yet what thoese are."""
    ctrl_log = None
    ctrl_pop = None
    tree = None

    @staticmethod
    def clone(individual: Individual) -> Individual:
        """
        Clone the Individual.

        :param individual: The individual to clone.
        :returns: The clone.
        """
        return copy.deepcopy(individual)


    def random(self) -> Individual:
        """
        Generate a random individual.

        :returns: The individual.
        """
        ind = Individual()
        ind.genotype = Genotype().random(self._innov_db_body, self._innov_db_brain, rng=make_rng_time_seed())
        return ind

    def random_controller(self) -> None:
        """Initialize a random brain for the current individual."""
        self.genotype.brain = self.genotype.random_brain(self._innov_db_brain, self.rng)

    @staticmethod
    def mutate_morphology(individual: Individual) -> Individual:
        """
        Mutate the body of an individual.

        :param individual: The individual.
        """
        individual.genotype = individual.genotype.mutate_body(individual._innov_db_body, individual.rng)
        individual.age += 1 #increase age when mutated because it is an offspring
        return individual

    @staticmethod
    def mutate_controller(individual: Individual) -> Individual:
        """
        Mutate the controller of an individual.

        :param individual: The individual.
        :returns: A Individual with mutated controller.
        """
        individual.genotype.brain = individual.genotype.mutate_brain(individual._innov_db_brain, individual.rng)
        return individual

    @staticmethod
    def mutate(individual: Individual) -> Individual:
        """
        Mutate all aspects of an Individual.

        :param individual: The individual to mutate.
        :returns: The mutated individual.
        """
        individual.genotype = individual.genotype.mutate(individual._innov_db_body, individual._innov_db_brain, individual.rng)
        return individual
