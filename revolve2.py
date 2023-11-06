from __future__ import annotations
from dataclasses import dataclass

import multineat
import numpy as np

from revolve2.ci_group.genotypes.cppnwin.modular_robot import BrainGenotypeCpg
from revolve2.ci_group.genotypes.cppnwin.modular_robot.v1 import BodyGenotypeV1
from revolve2.modular_robot import ModularRobot

from dataclasses import dataclass

from genotype import Genotype

from revolve2.experimentation.rng import make_rng_time_seed


class revolveWrapper(objecy):

    def __init__(self):
        self._innov_db_body = multineat.InnovationDatabase()
        self._innov_db_brain = multineat.InnovationDatabase()
        self._rng = make_rng_time_seed()

    def get_random(self):
        ind = Individual()
        ind.genotype = Genotype.random(self._innov_db_body, self._innov_db_brain, self._rng)
        ind.fitness = None

    def do_mutate(self, ind):
        gen = ind.genotype.mutate(self._innov_db_body, self._innov_db_brain, self._rng)
        ind.genotype = gen

    def do_crossover(self, ind1, ind2):
        gen = ind1.genotype.crossover(ind1, ind2, self._rng)
        ind = Individual()
        ind.genotype = gen
        ind.fitness = None

    def evaluate(self, ind):
        evaluator = Evaluator(headless=True, num_simulators=1)
        list_of_fitness = evaluator.evaluate([ind.genotype.develop])
        return list_of_fitness[0]


@dataclass
class Individual:
    """An individual in a population."""

    genotype: Genotype
    fitness: float


@dataclass
class Genotype(BodyGenotypeV1, BrainGenotypeCpg):
    """A genotype for a body and brain using CPPN."""

    @classmethod
    def random(
        cls,
        innov_db_body: multineat.InnovationDatabase,
        innov_db_brain: multineat.InnovationDatabase,
        rng: np.random.Generator,
    ) -> Genotype:
        """
        Create a random genotype.

        :param innov_db_body: Multineat innovation database for the body. See Multineat library.
        :param innov_db_brain: Multineat innovation database for the brain. See Multineat library.
        :param rng: Random number generator.
        :returns: The created genotype.
        """
        body = cls.random_body(innov_db_body, rng)
        brain = cls.random_brain(innov_db_brain, rng)

        return Genotype(body=body.body, brain=brain.brain)

    def mutate(
        self,
        innov_db_body: multineat.InnovationDatabase,
        innov_db_brain: multineat.InnovationDatabase,
        rng: np.random.Generator,
    ) -> Genotype:
        """
        Mutate this genotype.

        This genotype will not be changed; a mutated copy will be returned.

        :param innov_db_body: Multineat innovation database for the body. See Multineat library.
        :param innov_db_brain: Multineat innovation database for the brain. See Multineat library.
        :param rng: Random number generator.
        :returns: A mutated copy of the provided genotype.
        """
        body = self.mutate_body(innov_db_body, rng)
        brain = self.mutate_brain(innov_db_brain, rng)

        return Genotype(body=body.body, brain=brain.brain)

    @classmethod
    def crossover(
        cls,
        parent1: Genotype,
        parent2: Genotype,
        rng: np.random.Generator,
    ) -> Genotype:
        """
        Perform crossover between two genotypes.

        :param parent1: The first genotype.
        :param parent2: The second genotype.
        :param rng: Random number generator.
        :returns: A newly created genotype.
        """
        body = cls.crossover_body(parent1, parent2, rng)
        brain = cls.crossover_brain(parent1, parent2, rng)

        return Genotype(body=body.body, brain=brain.brain)

    def develop(self) -> ModularRobot:
        """
        Develop the genotype into a modular robot.

        :returns: The created robot.
        """
        body = self.develop_body()
        brain = self.develop_brain(body=body)
        return ModularRobot(body=body, brain=brain)

from revolve2.ci_group import fitness_functions, terrains
from revolve2.ci_group.simulation import make_standard_batch_parameters
from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot_simulation import (
    ModularRobotScene,
    Terrain,
    simulate_scenes,
)
from revolve2.simulators.mujoco_simulator import LocalSimulator


class Evaluator:
    """Provides evaluation of robots."""

    _simulator: LocalSimulator
    _terrain: Terrain

    def __init__(
        self,
        headless: bool,
        num_simulators: int,
    ) -> None:
        """
        Initialize this object.

        :param headless: `headless` parameter for the physics simulator.
        :param num_simulators: `num_simulators` parameter for the physics simulator.
        """
        self._simulator = LocalSimulator(
            headless=headless, num_simulators=num_simulators
        )
        self._terrain = terrains.flat()

    def evaluate(
        self,
        robots: list[ModularRobot],
    ) -> list[float]:
        """
        Evaluate multiple robots.

        Fitness is the distance traveled on the xy plane.

        :param robots: The robots to simulate.
        :returns: Fitnesses of the robots.
        """
        # Create the scenes.
        scenes = []
        for robot in robots:
            scene = ModularRobotScene(terrain=self._terrain)
            scene.add_robot(robot)
            scenes.append(scene)

        # Simulate all scenes.
        scene_states = simulate_scenes(
            simulator=self._simulator,
            batch_parameters=make_standard_batch_parameters(),
            scenes=scenes,
        )

        # Calculate the xy displacements.
        xy_displacements = [
            fitness_functions.xy_displacement(
                states[0].get_modular_robot_simulation_state(robot),
                states[-1].get_modular_robot_simulation_state(robot),
            )
            for robot, states in zip(robots, scene_states)
        ]

        return xy_displacements
