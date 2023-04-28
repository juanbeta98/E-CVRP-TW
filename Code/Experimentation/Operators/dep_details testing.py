from numpy.random import seed, choice, randint, random
from time import time
import sys
import pickle
import matplotlib.pyplot as plt

path: str = '/Users/juanbeta/My Drive/Research/Energy/E-CVRP-TW/Code/' ##### CHANGE WHEN NECESSARY!!!
#path: str = 'C:/Users/jm.betancourt/Documents/Research/Energy/E-CVRP-TW/Code/' ##### CHANGE WHEN NECESSARY!!!

sys.path.insert(0,path)
from E_CVRP_TW import  E_CVRP_TW, Constructive, Experiment, Genetic, Reparator, Feasibility

'''
General parameters
'''
start: float = time()

rd_seed: int = 0
seed(rd_seed)

verbose = True
saving = True

'''
Environment
'''
env: E_CVRP_TW = E_CVRP_TW(path)


'''
Constructive heuristic
'''
training_ind_prop = 0.1
RCL_criterion:str = 'Exo-Hybrid'

constructive_verbose = True

constructive:Constructive = Constructive()


'''
Genetic algorithm
'''
Population_size:int = 15
training_ind:int = int(round(Population_size * training_ind_prop,0))
Elite_size:int = int(Population_size * 0.5)

crossover_rate:float = 0.5
mutation_rate:float = 0.5

genetic: Genetic = Genetic(Population_size, Elite_size, crossover_rate, mutation_rate)

Operator:str = 'two opt'

'''
Repair operators
'''
repair_op: Reparator = Reparator()


'''
Feasibility operators
'''
feas_op: Feasibility = Feasibility()

'''
EXPERIMENTATION
Variable convention:
- Details: List of tuples (individual), where the tuple (distances, times) are discriminated per route
- best_individual: list with (individual, distance, time, details)
'''
lab: Experiment = Experiment()
colors: list = ['blue', 'red', 'black', 'purple', 'green', 'orange']

testing_times = {'s':0.5, 'm':3, 'l':7}


'''
Instance testing
'''
test_bed = [env.sizes['l'][0]]
#test_bed = ['c202C10.txt', 'c103_21.txt']


for instance in test_bed:
    # Saving performance 
    Results = dict()
    Incumbents = list()
    ploting_Times = list()

    # Setting runnign times depending on instance size
    if instance in env.sizes['s']:  max_time = 60 * testing_times['s']
    elif instance in env.sizes['m']:  max_time = 60 * testing_times['m']
    else:   max_time = 60 * testing_times['l']
   
    # Constructive
    g_start = time()
    env.load_data(instance)
    env.generate_parameters()
    constructive.reset(env)

    # Printing progress
    if verbose: 
        print(f'\n\n########################################################################')
        print(f'                 Instance {instance} / {Operator} ################')
        print(f'########################################################################')
        print(f'- size: {len(list(env.C.keys()))}')
        print(f'- bkFO: {env.bkFO[instance]}')
        print(f'- bkEV: {env.bkEV[instance]}')

    # Population generation
    Population, Distances, Times, Details, incumbent, best_individual, RCL_alpha = genetic.generate_population(env, constructive, training_ind, 
                                                                                                    g_start, instance, constructive_verbose)
    

    for i, individual in enumerate(Population):
        dep_t, dep_q = Details[i][2]
        feasible, details  = feas_op.individual_check(env, individual)

        if not feasible:
            print('FUCKED')
        if dep_q != details[2][2][1]:
            print(f'dep_q: individuo{i}')
        elif dep_t != details[2][2][0]:
            print(f'dep_t: individuo{i}')
        elif not feasible:
            print('FUCKED')
        




    