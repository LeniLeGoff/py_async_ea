from deap import tools, algorithms
import numpy as np

def seeded_init_repeat(container,func,seed,n):
    return  seed + container(func() for _ in range(n-1))

def steady_state_ea(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__):
    
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    var = np.var([fit for fit in fitnesses])
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    seed_fitness = population[0].fitness.values[0]

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)
    if var < 0.0001:
        return population, logbook, seed_fitness
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

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population.sort(key=lambda ind:ind.fitness.values[0],reverse=True)
        del population[half_length:]
        population = population + offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream,var)

        if var < 0.0001:
            break

    return population, logbook, seed_fitness