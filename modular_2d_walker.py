#! /usr/bin/python3
import sys
import pickle
import numpy as np
import gym
import configparser as cp
import random as rd

import log_data as ld
import asynch_ea as asynch
import novelty as nov

sys.path.append("modular_2d")
from modular_2d import individual as mod_ind


from deap import creator,base,tools,algorithms

env = None
def getEnv():
    global env
    if env is None:
        #env = M2D.Modular2D()
        # OpenAI code to register and call gym environment.
        env = gym.make("Modular2DLocomotion-v0")
    return env



fitness_data = ld.Data("fitness")
novelty_data = ld.Data("novelty")
learning_trials = ld.Data("learning_trials")
learning_delta = ld.Data("learning_delta")
plot_fit = ld.Plotter()
plot_ld = ld.Plotter()

def evaluate(individual, config):
    tree_depth = int(config["morphology"]["max_depth"])
    evaluation_steps = int(config["simulation"]["evaluation_steps"])
    interval = int(config["simulation"]["render_interval"])
    headless = bool(config["simulation"]["headless"])
    env_length = int(config["simulation"]["env_length"])

    env = getEnv()
    if tree_depth is None:
        try:
           tree_depth = individual.tree_depth
        except:
            raise Exception("Tree depth not defined in evaluation")
    tree = individual.genome.create(tree_depth)
    tree.create_children_lists()
    env.seed(int(config["experiment"]["seed"]))
    env.reset(tree=tree, module_list=individual.genome.moduleList)
    it = 0
    for i in range(evaluation_steps):
        it+=1
        if it % interval == 0 or it == 1:
            if not headless:
                env.render()

        action = np.ones_like(env.action_space.sample())
        observation, reward, done, info  = env.step(action)
        if reward< -10:
            break
        elif reward > env_length:
            reward += (evaluation_steps-i)/evaluation_steps
            individual.fitness.values = [reward]
            break
        if reward > 0:
            individual.fitness.values = [reward]
    return individual.fitness.values

def identity(a):
    return a

def learning_loop(individual,config):
    toolbox = base.Toolbox()
    toolbox.register("individual", mod_ind.Individual.init_for_controller_opti,individual=individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate,config=config)
    toolbox.register("mutate", mod_ind.Individual.mutate_controller, mutation_rate = float(config["controller"]["mut_rate"]),mut_sigma = float(config["controller"]["sigma"]))
    toolbox.register("select",tools.selTournament, tournsize = int(config["controller"]["tournament_size"]))
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("max",np.max)
    stats.register("min",np.min)
    stats.register("fitness",identity)
    hof = tools.HallOfFame(1)
    pop = toolbox.population(int(config["controller"]["pop_size"]))
    pop, log = algorithms.eaSimple(pop,toolbox,cxpb=0,mutpb=1,ngen=int(config["controller"]["nbr_gen"]),stats=stats,halloffame=hof,verbose=False)
    individual.genome = hof[0].genome
    individual.ctrl_log = log
    individual.ctrl_pop = pop
    individual.learning_delta = hof[0].fitness.values[0] - log.select("min")[0]
    individual.fitness = hof[0].fitness
    return individual

def elitist_select(pop,size):
    sort_pop = pop
    sort_pop.sort(key=lambda p: p.fitness.values[0])
    return sort_pop[:size]

def age_select(pop,size):
    sort_pop = pop
    sort_pop.sort(key=lambda p: p.age)
    return sort_pop[:size]

def generate(parents,toolbox,size):
    offspring = toolbox.parent_select(parents, size)

    # deep copy of selected population
    offspring = list(map(toolbox.clone, offspring))
    for o in offspring:
        toolbox.mutate(o)
        o.index=mod_ind.Individual.static_index
        mod_ind.Individual.static_index+=1
        # TODO only reset fitness to zero when mutation changes individual
        # Implement DEAP built in functionality
        o.fitness = mod_ind.Fitness()
    return offspring

def update_data(toolbox,population,gen,log_folder,config,plot=False,save=False):
    fitness_values = [ind.fitness.values[0] for ind in population]
    fitness_data.add_data(fitness_values)
    goal_select = config["experiment"].getboolean("goal_select")
    if goal_select == False:
        novelty_scores = [ind.novelty.values[0] for ind in population]
        novelty_data.add_data(novelty_scores)
    learning_deltas = [ind.learning_delta for ind in population]
    learning_delta.add_data(learning_deltas)
    if plot:
        plot_fit.plot(fitness_data)
        plot_ld.plot(learning_delta)
    if save:
        n_gens=int(config["experiment"]["checkpoint_frequency"])
        fitness_data.save(log_folder + "/fitnesses")
        learning_delta.save(log_folder + "/learning_delta")
        if goal_select == False:
            novelty_data.save(log_folder + "/novelty")
        if(gen%n_gens == 0):
            pickle.dump(population,open(log_folder + "/pop_" + str(gen), "wb"))
            mod_ind.save_learning_ctrl_log(population,gen,log_folder)
            mod_ind.save_learning_ctrl_pop(population,gen,log_folder)


def compute_novelty_scores(population,archive,config):
    for ind in population:
        dist = nov.distances(ind,population,archive,mod_ind.morphological_distance)
        ind.novelty.values = nov.sparsness(dist),
    for ind in population:
        archive = nov.update_archive(ind,ind.novelty.values[0],archive,novelty_thr=float(config["novelty"]["nov_thres"]),adding_prob=float(config["novelty"]["adding_prob"]),arch_size=int(config["novelty"]["arch_max_size"]))

def novelty_select(parents,size,archive,config):
    compute_novelty_scores(parents,archive,config)
    return tools.selTournament(parents,size,int(config["morphology"]["tournament_size"]),fit_attr="novelty")

if __name__ == '__main__':
    config = cp.ConfigParser()
    if(len(sys.argv) == 2):
        config.read(sys.argv[1])
    else:
        config.read("modular_2d_walker.cfg")

    log_folder = config["experiment"]["log_folder"]
    exp_name = config["experiment"]["name"]
    foldername = ld.create_log_folder(log_folder,exp_name)


    goal_select = config["experiment"].getboolean("goal_select")
    elitist_survival = config["experiment"].getboolean("elitist_survival")

    #define seed
    seed = int(os.getrandom(5,flags=os.GRND_RANDOM).hex(),16)
    rd.seed(a=seed)
    config["experiment"]["seed"] = str(seed)

    archive=[]

    toolbox = base.Toolbox()

    toolbox.register("individual", mod_ind.Individual.random,config=config)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("eval", learning_loop,config=config)
    toolbox.register("mutate", mod_ind.Individual.mutate_morphology, mutation_rate=float(config["morphology"]["mut_rate"]),mut_sigma=float(config["morphology"]["sigma"]))
    print()
    if goal_select: #Do a goal-based selection
        toolbox.register("parent_select",tools.selTournament,tournsize=int(config["morphology"]["tournament_size"]))
    else: #Do a novelty selection.
        toolbox.register("parent_select",novelty_select, archive=archive ,config=config)
    if elitist_survival: #Do an elitist survival: remove the worst individual in term of fitness
        toolbox.register("death_select", elitist_select)
    else: #Do an age based survival: remove the oldest individual
        toolbox.register("death_select", age_select)
    toolbox.register("generate",generate)
    toolbox.register("extra",update_data,log_folder=log_folder + "/" + foldername,config=config,plot=bool(config["experiment"].getboolean("plot_prog")),save=config["experiment"].getboolean("save_logs"))


    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    if goal_select == False:
        stats_nov = tools.Statistics(key=lambda ind: ind.novelty.values)
        stats_nov.register("avg", np.mean)
        stats_nov.register("std", np.std)
        stats_nov.register("min", np.min)
        stats_nov.register("max", np.max)


    asynch_ea = asynch.AsynchEA(int(config["morphology"]["pop_size"]),sync=float(config["morphology"]["synch"]))
    pop = asynch_ea.init(toolbox)
    print("init finish")
    for i in range(int(config["morphology"]["nbr_gen"])):
        pop = asynch_ea.step(toolbox)
        print("fitness - ",stats.compile(pop))
        if goal_select == False:
            print("novelty - ",stats_nov.compile(pop),"archive size :", len(archive))