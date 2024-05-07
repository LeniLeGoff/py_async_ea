import os
import pickle
from ._individual import Individual
def save_learning_ctrl_log(population ,generation: int, log_folder: str) -> None:
    """
    Save the learning log.

    :param population: The population.
    :param generation: The generation index.
    :param log_folder: The folder to save the log to.
    """
    foldername = f"{log_folder}/controller_logs_{generation}"
    if not os.path.exists(foldername):
        os.makedirs(foldername)

    for index, individual, in enumerate(population):
        if individual.ctrl_log is None:
            continue
        with open(f"{foldername}/ctrl_log_{index}",'w') as file:
            file.write(individual.ctrl_log_to_string())

def save_learning_ctrl_pop(population ,generation: int, log_folder: str) -> None:
    """
    Save the population log.

    :param population: The population.
    :param generation: The generation index.
    :param log_folder: The folder to save the log to.
    """
    foldername = f"{log_folder}/controller_logs_{generation}"
    if not os.path.exists(foldername):
        os.makedirs(foldername)

    for index, ind in enumerate(population):
        if ind.ctrl_log is None:
            continue
        pickle.dump(ind.ctrl_pop,open(f"{foldername}/ctrl_pop_{index}", "wb"))

def morphological_distance(individual1: Individual, individual2: Individual) -> float:
    """
    Calculate morphological distance between individuals.

    :param individual1: The first individual.
    :param individual2: The second individual.
    :returns: The tree-edit distance.
    """
    raise NotImplemented("This is not defined yet.")
