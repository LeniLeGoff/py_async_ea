"""Individual class."""
from __future__ import annotations
from deap import base
import multineat
import numpy as np
import copy
from ._genotype import Genotype
from dataclasses import dataclass, field

from revolve2.experimentation.rng import make_rng_time_seed
from revolve2.ci_group.genotypes.cppnwin.modular_robot import BrainGenotypeCpg
from uuid import uuid1, UUID
class Metric(base.Fitness):
    """A metric class based on deap Fitness."""
    def __init__(self) -> None:
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
    genotype: Genotype | None = field(default=None)

    """Evaluation Measures."""
    novelty: Metric = field(default_factory=lambda: Metric())
    fitness: Metric = field(default_factory=lambda: Metric())
    learning_delta: Metric = field(default_factory=lambda: Metric())

    """Auxiliary varibables."""
    rng: np.random.Generator = field(default_factory=lambda: make_rng_time_seed())
    uuid: UUID = field(default_factory=lambda: uuid1())

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
        individual = copy.deepcopy(individual)
        individual.uuid = uuid1()
        return individual

    @staticmethod
    def random(dbs: tuple[multineat.InnovationDatabase, multineat.InnovationDatabase]) -> Individual:
        """
        Generate a random individual.

        :param dbs: The innovation databases.
        :returns: The individual.
        """
        individual = Individual()
        individual.genotype = Genotype.random(*dbs, rng=make_rng_time_seed())
        return individual

    def random_controller(self, innov_db_brain: multineat.InnovationDatabase) -> None:
        """
        Initialize a random brain for the current individual.

        :param innov_db_brain: The innovation db for the brain.
        """
        self.genotype.brain = BrainGenotypeCpg.random_brain(innov_db_brain, self.rng).brain

    @staticmethod
    def init_for_controller_opti(individual: Individual, innov_db_brain: multineat.InnovationDatabase) -> Individual:
        """
        Initialize a new individual based on a existing one for controller optimization.

        :param individual: The individual to clone.
        :param innov_db_brain: The innovation db for the brain.
        :returns: A new individual with same body but random controller.
        """
        individual = Individual.clone(individual)
        individual.fitness = Metric()
        individual.random_controller(innov_db_brain)
        return individual

    @staticmethod
    def mutate_morphology(individual: Individual, innov_db_body: multineat.InnovationDatabase) -> Individual:
        """
        Mutate the body of an individual.

        :param individual: The individual.
        :param innov_db_body: The innovation db for the body.
        """
        individual.genotype.body = individual.genotype.mutate_body(innov_db_body, individual.rng).body
        individual.age += 1 #increase age when mutated because it is an offspring
        return individual

    @staticmethod
    def mutate_controller(individual: Individual, innov_db_brain: multineat.InnovationDatabase) -> Individual:
        """
        Mutate the controller of an individual.

        :param individual: The individual.
        :param innov_db_brain: The innovation db for the brain.
        :returns: An Individual with mutated controller.
        """
        individual.genotype.brain = individual.genotype.mutate_brain(innov_db_brain, individual.rng).brain
        return individual

    @staticmethod
    def mutate(individual: Individual, dbs: tuple[multineat.InnovationDatabase, multineat.InnovationDatabase]) -> Individual:
        """
        Mutate all aspects of an Individual.

        :param individual: The individual to mutate.
        :param dbs: The innovation databases.
        :returns: The mutated individual.
        """
        individual.genotype = individual.genotype.mutate(*dbs, individual.rng)
        return individual

    def get_controller_genome(self):
        return self.genotype.brain

