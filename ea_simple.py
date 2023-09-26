from deap import tools, algorithms
import numpy as np

def seeded_init_repeat(container,func,seed,n):
    return  seed + container(func() for _ in range(n-1))

def update_best(best_ind,pop):
    for ind in pop:
        if(ind.fitness.values[0] > best_ind.fitness.values[0]):
            best_ind = ind
    return best_ind

def steady_state_ea(population, toolbox, cxpb, mutpb, ngen, stats=None,
                     verbose=__debug__,min_fit=0,target_fit=10000):#todo change min_fit and target_fit for minimisation case

    
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    #First evaluate the morphology with the random controller. If the fitness is lower than the min fitness do not apply the optimisation.
    population[0].fitness.values = toolbox.evaluate(population[0])
    seed_fitness = population[0].fitness.values[0]
    best_ind = population[0]

    if(population[0].fitness.values[0] <= min_fit):
        return population, logbook, seed_fitness, best_ind


    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        
    var = np.var([fit for fit in fitnesses])
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    best_ind = update_best(best_ind,population)
    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)
    if var < 0.0001 or best_ind.fitness.values[0] >= target_fit:
        return population, logbook, seed_fitness, best_ind
    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        half_length = int(len(population)/2)
        offspring = toolbox.select(population, half_length)

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)


        var = np.var([fit for fit in fitnesses])
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update best inidividual ever seen
        best_ind = update_best(best_ind,offspring)

        # Replace the current population by the offspring
        population.sort(key=lambda ind:ind.fitness.values[0],reverse=True)
        del population[half_length:]
        population = population + offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream,var)

        if var < 0.0001 or best_ind.fitness.values[0] >= target_fit:
            break

    return population, logbook, seed_fitness, best_ind