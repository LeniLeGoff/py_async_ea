#! /usr/bin/python3
import sys
import os
import pickle
import configparser as cp


def identity(a):
    return a

def load_population(filename):
    with open(filename,"rb") as file:
        population = pickle.load(file)
    return population


if __name__ == '__main__':

    if len(sys.argv) != 2:
        exit(1)
    
    log_folder = sys.argv[1]
    config = cp.ConfigParser()
    config.read(log_folder + "/config.cfg")  
    ofile = open(log_folder + "/morph_norms","w+")
    lines = []
    for filename in os.listdir(log_folder):
        if(filename.split("_")[0] != "pop"):
            continue
        print(filename)
        archived_pop_file = log_folder + "/" + filename
        norms = ""
        population = load_population(archived_pop_file)
        for ind in population[:-1]:
            ind.create_tree(config)
            norms += str(ind.tree.norm()) + ","
        population[-1].create_tree(config)
        norms += str(population[-1].tree.norm()) + "\n"
        lines.append(norms)

    ofile.writelines(lines)
    ofile.close()
    