from numpy.random import seed, choice
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

'''
Environment
'''
env: E_CVRP_TW = E_CVRP_TW(path)


'''
Constructive heuristic
'''
training_ind_prop = 0.5
RCL_criterion:str = 'Exo-Hybrid'

constructive:Constructive = Constructive()


'''
Genetic algorithm
'''
Population_size:int = 1000
training_ind:int = Population_size * 0.5
Elite_size:int = int(Population_size * 0.05)

crossover_rate:float = 0.5
mutation_rate:float = 0.5

genetic: Genetic = Genetic(Population_size, Elite_size, crossover_rate, mutation_rate)


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

testing_times = {'s':0.5, 'm':2, 'l':6}

'''
Instance testing
'''
test_bed = env.sizes['l']

for instance in test_bed:
    # Saving performance 
    Results = {}
    Incumbents = []
    Times = []

    # Setting runnign times depending on instance size
    if instance in env.sizes['s']:  max_time = testing_times['s']
    elif instance in env.sizes['m']:  max_time = testing_times['m']
    else:   max_time = testing_times['l']
   
    # Constructive
    g_start = time()
    env.load_data(instance)
    env.generate_parameters()
    constructive.reset(env)

    # Printing results
    if verbose: 
        print(f'\n\n########################################################################')
        print(f'################ Instance {instance} / {Operator} ################')
        print(f'########################################################################')
        print(f'- size: {len(list(env.C.keys()))}')
        print(f'- bkFO: {env.bkFO[instance]}')
        print(f'- bkEV: {env.bkEV[instance]}')

    Population, Distances, Times, Details, incumbent, best_individual = genetic.generate_population(env, constructive, training_ind, start, instance, True)

    ### Print performance
    if verbose: 
        print('\n')
        print(f'########## Performance ##########')
        print(f'total running time: {round(time() - start,2)}')
        print(f'incumbent: {round(incumbent,2)}')
        print(f'gap: {round(lab.compute_gap(env, instance, incumbent)*100,2)}%')
        print(f'time to find: {round(Results["time to find"],2)}')
        #print(f'best solution: {best_individual}')
        print('\n')

    ### Save performance
    a_file = open(path + f'Experimentation/Operators/RCL criterion/{RCL_criterion}/results_{instance}', "wb")
    pickle.dump(Results, a_file)
    a_file.close()




