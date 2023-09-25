#! /usr/bin/python3
import sys
import pickle
import configparser as cp


def identity(a):
    return a

def load_population(filename):
    with open(filename,"rb") as file:
        population = pickle.load(file)
    return population


if __name__ == '__main__':

    if len(sys.argv) != 3:
        exit(1)
        
    archived_pop_file = sys.argv[1]

    config = cp.ConfigParser()
    config.read(sys.argv[2])

    norms = []

    population = load_population(archived_pop_file)
    for ind in population:
        ind.create_tree(config)
        norms.append(ind.tree.norm())

    print("norms: ", norms)
    