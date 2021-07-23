# IMPORTING MODULES
import numpy as np
import matplotlib.pyplot as plt
from ypstruct import structure


# OBJECTIVE FUNCTION
def objective(value):
    return sum(value ** 2)


# PROBLEM STATEMENT
problem = structure()
problem.costfunc = objective
problem.nvar = 8
problem.varmin = -10
problem.varmax = 10


# PARAMETER STATEMENT
params = structure()
params.maxit = 50
params.npop = 100
params.pc = 1
params.gamma = 0.1
params.sigma = 0.1
params.mu = 0.1


# IMPLEMENTING GENETIC ALGORITHM
def run_ga(problem, params):

    # EXTRACTION INFORMATION
    # PROBLEM STATEMENT
    costfunc = problem.costfunc
    nvar = problem.nvar
    varmin = problem.varmin
    varmax = problem.varmax

    # PARAMETER STATEMENT
    max_iteration = params.maxit
    population_number = params.npop
    proportion = params.pc
    nc = int(np.round(proportion * population_number / 2) * 2)
    gamma = params.gamma
    sigma = params.sigma
    mutation = params.mu

################################ INITIALIZATION PHASE ################################

    # EMPTY INDIVIDUAL
    empty_individual = structure()
    empty_individual.position = None
    empty_individual.cost = None

    # BEST SOLUTION
    best_solution = empty_individual.deepcopy()
    best_solution.cost = np.inf

    # INITIALIZE POPULATION
    population = empty_individual.repeat(population_number)
    for value in range(0, population_number):
        population[value].position = np.random.uniform(varmin, varmax, nvar)
        population[value].cost = costfunc(population[value].position)

    # BEST COST VALUE
    best_cost = np.empty(max_iteration)

############################ CROSSOVER,MUTATION,SELECTION PHASE ######################

    # MIAN LOOP
    for iterations in range(0, max_iteration):
        popc = []
        for _ in range(nc // 2):
            q = np.random.permutation(population_number)
            p1 = population[q[0]]
            p2 = population[q[1]]

            # CROSSOVER
            offspring_1, offspring_2 = crossover(p1, p2, gamma)

            # mutate
            offspring_1 = mutate(offspring_1, mutation, sigma)
            offspring_2 = mutate(offspring_2, mutation, sigma)

            # APPLY BOUNDS
            apply_bound(offspring_1, varmin, varmax)
            apply_bound(offspring_2, varmin, varmax)

            # EVALUATE FIRST OFFSPRING
            offspring_1.cost = costfunc(offspring_1.position)
            if offspring_1.cost < best_solution.cost:
                best_solution = offspring_1.deepcopy()

            # EVALUATE SECOND OFFSPRING
            offspring_2.cost = costfunc(offspring_2.position)
            if offspring_2.cost < best_solution.cost:
                best_solution = offspring_2.deepcopy()

            # ADDING OFFSPRING TO NEW POPULATION
            popc.append(offspring_1)
            popc.append(offspring_2)

############################## MERGE,SORT,SELECTION PHASE ###########################

        population += popc
        population = sorted(population, key=lambda x: x.cost, reverse=True)
        population = population[0:population_number]

        # STORING BEST COST VALUE
        best_cost[iterations] = best_solution.cost

        # SHOWING RESULTS
        # print(f"Iteration: {iterations} Best Cost: {best_cost[iterations]}")

    # OUTPUT RESULT
    output = structure()
    # output.pop = population
    # output.best_solution = best_solution
    output.best_cost = best_cost
    return output


# CROSSOVER FUNCTION
def crossover(p1, p2, gamma):
    c1 = p1.deepcopy()
    c2 = p2.deepcopy()
    alpha = np.random.uniform(-gamma, 1+gamma, *c1.position.shape)
    c1.position = alpha * p1.position + (1-alpha) * p2.position
    c2.position = alpha * p2.position + (1-alpha) * p1.position
    return c1, c2


# MUTATION FUNCTION
def mutate(x, mu, sigma):
    y = x.deepcopy()
    flag = np.random.rand(*x.position.shape) <= mu
    ind = np.argwhere(flag)
    y.position[ind] += sigma * np.random.rand(*ind.shape)
    return y


# APPLYING BOUNDS
def apply_bound(x, varmin, varmax):
    x.position = np.maximum(x.position, varmin)
    x.position = np.minimum(x.position, varmax)
