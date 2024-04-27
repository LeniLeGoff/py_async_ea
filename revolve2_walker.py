#! /usr/bin/python3
import os
import logging
import time
import pickle
import numpy as np
import configparser as cp
import random
import math
import multiprocessing as mp
from functools import partial
from exception import LogExceptions
import log_data as ld
import asynch_ea as asynch
import ea_simple as ea
import tools.novelty as nov
from deap import base, tools
from revolve import Individual, Metric, save_learning_ctrl_pop, save_learning_ctrl_log, morphological_distance

from revolve2.ci_group.terrains import flat
from revolve2.experimentation.logging import setup_logging
from revolve2.ci_group.fitness_functions import xy_displacement
from revolve2.ci_group.simulation_parameters import make_standard_batch_parameters
from revolve2.simulators.mujoco_simulator._simulation_state_impl import SimulationStateImpl
from revolve2.simulators.mujoco_simulator._control_interface_impl import ControlInterfaceImpl
from revolve2.simulators.mujoco_simulator._scene_to_model import scene_to_model
from revolve2.modular_robot_simulation import ModularRobotScene, SceneSimulationState
from revolve2.simulation.scene import SimulationState
import mujoco
import multineat

time_data = ld.Data("time_data")
fitness_data = ld.Data("fitness")
parents_index_data = ld.Data("parent_indexes")
eval_index_data = ld.Data("evaluted_indexes")
novelty_data = ld.Data("novelty")
learning_trials = ld.Data("learning_trials")
learning_delta = ld.Data("learning_delta")
#morph_norm = ld.Data("morph_norms")
plot_fit = ld.Plotter()
plot_ld = ld.Plotter()

"""Revolve2 objects."""
batch_parameters = make_standard_batch_parameters()
(innov_db_body, innov_db_brain) = dbs = (multineat.InnovationDatabase(), multineat.InnovationDatabase())



def evaluate(individual: Individual, config: dict) -> list[float]:
    """
    Evaluate an individual in locomotion.

    :param individual: The individual.
    :param config: The experiments config.
    :returns: The fitness of the individual (as a list).
    """
    robot = individual.genotype.develop()
    scene = ModularRobotScene(terrain=flat())
    scene.add_robot(robot)


    scene, mr_mapping = scene.to_simulation_scene()
    control_step = batch_parameters.control_frequency
    sample_step = batch_parameters.sampling_frequency

    model, mapping = scene_to_model(scene, batch_parameters.simulation_timestep, cast_shadows=False, fast_sim=True)
    data = mujoco.MjData(model)

    # The measured states of the simulation
    simulation_states: list[SimulationState] = []

    mujoco.mj_forward(model, data)

    if sample_step is not None:
        simulation_states.append(
            SimulationStateImpl(data=data, abstraction_to_mujoco_mapping=mapping, camera_views={})
        )

    control_interface = ControlInterfaceImpl(
        data=data, abstraction_to_mujoco_mapping=mapping
    )

    for _ in range(int(config["simulation"]["evaluation_steps"])):
        simulation_state = SimulationStateImpl(
            data=data, abstraction_to_mujoco_mapping=mapping, camera_views={}
        )
        scene.handler.handle(simulation_state, control_interface, control_step)
        mujoco.mj_step(model, data)

    # Sample one final time.
    if sample_step is not None:
        simulation_states.append(
            SimulationStateImpl(
                data=data, abstraction_to_mujoco_mapping=mapping, camera_views={}
            )
        )

    states = [SceneSimulationState(simulation_state, mr_mapping) for simulation_state in simulation_states]
    individual.nbr_eval += 1
    individual.fitness.values = [
        xy_displacement(
            states[0].get_modular_robot_simulation_state(robot),
            states[-1].get_modular_robot_simulation_state(robot),
        )
    ]
    if config["controller"].getboolean("no_learning"):
        return individual
    return individual.fitness.values


def learning_loop(individual: Individual, config: dict) -> Individual:
    """
    Integrate learningn into evolution.

    :param individual: The individual forced to learn.
    :param config: The configuration.
    :returns: A genius.
    """
    """Register Toolbox operations."""
    toolbox = base.Toolbox()
    toolbox.register("individual", Individual.init_for_controller_opti, individual=individual, innov_db_brain=innov_db_brain)
    toolbox.register("population", ea.seeded_init_repeat, list, toolbox.individual, [individual])
    toolbox.register("evaluate", LogExceptions(evaluate), config=config)
    toolbox.register("mutate", Individual.mutate_controller, innov_db_brain=innov_db_brain)
    toolbox.register("select", tools.selBest)
    pool = mp.Pool(processes=int(config["controller"]["pop_size"]))
    toolbox.register("map", pool.map)
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    stats.register("min", np.min)
    stats.register("fitness", lambda a: a)
    population = toolbox.population(int(config["controller"]["pop_size"]))

    target_fit = None if (val := float(config["controller"]["target_fit"])) == -1 else val
    target_delta = None if (val := float(config["controller"]["target_delta"])) == -1 else val

    population, log, seed_fitness, individual = ea.steady_state_ea(
        population,
        toolbox,
        cxpb=0,
        mutpb=1,
        ngen=int(config["controller"]["nbr_gen"]),
        stats=stats,
        verbose=False,
        min_fit=6,
        target_fit=target_fit,
        target_delta=target_delta,
    )

    individual.ctrl_log = log
    individual.ctrl_population = [individual.get_controller_genome() for individual in population]

    individual.learning_delta.values = individual.fitness.values[0] - seed_fitness,
    individual.nbr_eval = 1 if log is None else sum(log.select("nevals"))

    pool.terminate()
    pool.join()
    return individual

"""Im not touching this."""
def elitist_select(pop, size):
    sort_pop = pop
    sort_pop.sort(key=lambda p: p.fitness.values[0])
    return sort_pop[:size]


def age_select(pop, size):
    sort_pop = pop
    sort_pop.sort(key=lambda p: p.age)
    return sort_pop[:size]


def generate(parents, toolbox, size):
    logging.info("Tournament started!")
    selected_parents = toolbox.parent_select(parents, size)

    # deep copy of selected population
    offspring = list(map(toolbox.clone, selected_parents))
    for o in offspring:
        toolbox.mutate(o)
        o.index = Individual.static_index
        o.nbr_eval = 0
        o.fitness = Metric()
        Individual.static_index += 1
    return offspring


def update_data(toolbox, population, gen, log_folder, config, plot=False, save=False):
    time_data.add_data([time.time()])
    fitness_values = [ind.fitness.values[0] for ind in population]
    fitness_data.add_data(fitness_values)
    indexes = [ind.index for ind in population]
    parents_index_data.add_data(indexes)
    select_type = config["experiment"]["select_type"]
    if select_type == "novelty":
        novelty_scores = [ind.novelty.values[0] for ind in population]
        novelty_data.add_data(novelty_scores)
    if not config["controller"].getboolean("no_learning"):
        learning_delta.add_data([ind.learning_delta.values[0] for ind in population])
        learning_trials.add_data([ind.nbr_eval for ind in population])
    #morph_norm.add_data([ind.tree.norm() for ind in population])
    if plot:
        plot_fit.plot(fitness_data)
        plot_ld.plot(learning_delta)
    if save:
        time_data.save(log_folder + "/time_data")
        time_data.depop()
        n_gens = int(config["experiment"]["checkpoint_frequency"])
        fitness_data.save(log_folder + "/fitnesses")
        fitness_data.depop()
        parents_index_data.save(log_folder + "/parent_indexes")
        parents_index_data.depop()
        eval_index_data.save(log_folder + "/new_ind_indexes")
        eval_index_data.depop()
        #morph_norm.save(log_folder + "/morph_norms")
        #morph_norm.depop()
        if not config["controller"].getboolean("no_learning"):
            learning_delta.save(log_folder + "/learning_delta")
            learning_delta.depop()
            learning_trials.save(log_folder + "/learning_trials")
            learning_trials.depop()
        if select_type == "novelty":
            novelty_data.save(log_folder + "/novelty")
            novelty_data.depop()
        if (gen % n_gens == 0):
            pickle.dump(population, open(log_folder + "/pop_" + str(gen), "wb"))
            if not config["controller"].getboolean("no_learning"):
                save_learning_ctrl_log(population, gen, log_folder)
                save_learning_ctrl_pop(population, gen, log_folder)


def compute_novelty_scores(population, archive, config):
    for individual in population:
        if individual.novelty.valid:
            continue
        dist = nov.distances_parallel(population, archive, partial(morphological_distance, individual2=individual),
                                      cores=int(config["experiment"]["max_workers"]))
        individual.novelty.values = nov.sparsness(dist),
    for individual in population:
        archive = nov.update_archive(individual, individual.novelty.values[0], archive,
                                     novelty_thr=float(config["novelty"]["nov_thres"]),
                                     adding_prob=float(config["novelty"]["adding_prob"]),
                                     arch_size=int(config["novelty"]["arch_max_size"]))


def novelty_select(parents, size, archive, config):
    compute_novelty_scores(parents, archive, config)
    return tools.selTournament(parents, size, int(config["morphology"]["tournament_size"]), fit_attr="novelty")

def main():
    setup_logging(file_name="log.txt")

    config = cp.ConfigParser()
    config.read("revolve2_walker.cfg")
    max_workers = 6

    config["experiment"]["max_workers"] = str(max_workers)

    log_folder = config["experiment"]["log_folder"]
    foldername = ld.create_log_folder(log_folder, config["experiment"]["name"])

    select_type = config["experiment"]["select_type"]

    # define seed
    seed = int(os.getrandom(5, flags=os.GRND_RANDOM).hex(), 16)
    random.seed(a=seed)
    config["experiment"]["seed"] = str(seed)

    archive = []

    toolbox = base.Toolbox()
    toolbox.register("individual", Individual.random, dbs=dbs)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    if config["controller"].getboolean("no_learning"):
        toolbox.register("eval", evaluate, config=config)
        toolbox.register("mutate", Individual.mutate,dbs=dbs)
    else:
        toolbox.register("eval", learning_loop, config=config)
        toolbox.register("mutate", Individual.mutate_morphology,innov_db_body=innov_db_body)
    match select_type:
        case "goal":
            toolbox.register("parent_select", tools.selTournament, tournsize=int(config["morphology"]["tournament_size"]))
        case "novelty":
            toolbox.register("parent_select", novelty_select, archive=archive, config=config)
        case "delta":
            toolbox.register("parent_select", tools.selTournament,
                             tournsize=int(config["morphology"]["tournament_size"]), fit_attr="learning_delta")
    if config["experiment"].getboolean("elitist_survival"):
        """Do an elitist survival: remove the worst individual in term of fitness."""
        toolbox.register("death_select", elitist_select)
    else:
        """Do an age based survival: remove the oldest individual."""
        toolbox.register("death_select", age_select)

    toolbox.register("generate", generate)
    toolbox.register("extra", update_data, log_folder=f"{log_folder}/{foldername}", config=config,
                     plot=bool(config["experiment"].getboolean("plot_prog")),
                     save=config["experiment"].getboolean("save_logs"))

    stats = tools.Statistics(key=lambda individual: individual.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    if select_type == "novelty":
        stats_nov = tools.Statistics(key=lambda individual: individual.novelty.values)
        stats_nov.register("avg", np.mean)
        stats_nov.register("std", np.std)
        stats_nov.register("min", np.min)
        stats_nov.register("max", np.max)

    with open(f"{log_folder}/{foldername}/config.cfg", 'w') as configfile:
        config.write(configfile)

    evaluations_budget = int(config["experiment"]["evaluations_budget"])

    asynch_ea = asynch.AsynchEA(int(config["morphology"]["pop_size"]), max_workers,
                                sync=float(config["morphology"]["synch"]))
    pop = asynch_ea.init(toolbox)
    logging.info(f"Initialization finished, running: {evaluations_budget} evaluations")
    nbr_eval = 0

    logging.info("----------------")
    logging.info("Start experiment")
    for ind in pop:
        nbr_eval += ind.nbr_eval
    while nbr_eval < evaluations_budget:
        pop, new_inds = asynch_ea.step(toolbox)
        if len(new_inds) > 0:
            new_idx = [ind.index for ind in new_inds]
            eval_index_data.add_data(new_idx)
            for ind in new_inds:
                nbr_eval += ind.nbr_eval
            logging.info(f"Fitness - {stats.compile(pop)}")
            if select_type == "novelty":
                logging.info(f"Novelty - {stats_nov.compile(pop)}archive size :{len(archive)}")
            logging.info(f"Number of Evaluations: {nbr_eval}", )
            logging.info(f"Current Progress : {float(nbr_eval) / float(evaluations_budget):.1%}")

    asynch_ea.terminate()
    logging.info("EA has terminated normally")

if __name__ == '__main__':
    main()
