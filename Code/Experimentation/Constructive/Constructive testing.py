#%%
from numpy.random import seed
from time import time
import sys
import pickle
import matplotlib.pyplot as plt

path: str = '/Users/juanbeta/My Drive/Research/Energy/E-CVRP-TW/Code/' ##### CHANGE WHEN NECESSARY!!!
#path: str = 'C:/Users/jm.betancourt/Documents/Research/Energy/E-CVRP-TW/Code/' ##### CHANGE WHEN NECESSARY!!!

sys.path.insert(0,path)
from E_CVRP_TW import  E_CVRP_TW, Constructive, Experiment

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
RCL_list: list[float] = [0.15, 0.25, 0.35, 0.5]
training_prop = 0.4
constructive: Constructive = Constructive()
RCL_alpha = RCL_list[1]

'''
EXPERIMENTATION
Variable convention:
- Details: List of tuples (individual), where the tuple (distances, times) are discriminated per route
- best_individual: list with (individual, distance, time, details)
'''
lab: Experiment = Experiment()
colors: list = ['blue', 'red', 'black', 'purple', 'green', 'orange']


'''
Instance testing
'''
max_time: int = 60*5 # 5 minutes
test_bed = env.sizes['s'][:2] + env.sizes['m'][:2] + env.sizes['l'][:3]

for instance in env.sizes['l']:
    # Saving performance
    Results = {}
    Incumbents = []
    Times = []

    # Setting runnign times
    if instance in env.sizes['s']:  max_time = 60*2
    if instance in env.sizes['s']:  max_time = 60*4
    else:   max_time = 60*8
   
    # Constructive
    start = time()
    env.load_data(instance)
    env.generate_parameters()
    constructive.reset(env)
    incumbent = 1e9
    ind = -1

    # Adaptative

    # Printing results
    if verbose: 
        print(f'\n\n########################################################################')
        print(f'######################### Instance {instance} #########################')
        print(f'########################################################################')
        print(f'- size: {len(list(env.C.keys()))}')
        print(f'- bkFO: {env.bkFO[instance]}')
        print(f'- bkEV: {env.bkEV[instance]}')

        print(f'\nTime \t \tInd \t \tIncumbent \tgap \t \t#Routes')

    while time() - start < max_time:
        ind += 1
        individual: list = []
        distance: float = 0
        distances: list = []
        t_time: float = 0
        times: list = []

        # Intitalizing environemnt
        constructive.reset(env)
        while len(constructive.pending_c) > 0:
            t, d, q, k, route = constructive.RCL_based_constructive(env, RCL_alpha)
            individual.append(route)
            distance += d
            distances.append(d)
            t_time += t
            times.append(t)
            
        if distance < incumbent:
            incumbent = distance
            best_individual: list = [individual, distance, t_time, (distances, times), time() - start]
            constructive.print_constructive(env, instance, time() - start, ind, incumbent, len(individual))

        Incumbents.append(incumbent)
        Times.append(time() - start)

    Results['best individual'] = best_individual[0]
    Results['best distance'] = best_individual[1]
    Results['total time'] = best_individual[2]
    Results['others'] = best_individual[3]
    Results['time to find'] = best_individual[4]
    Results['incumbents'] = Incumbents
    Results['inc times'] = Times


    ### Print performance
    print('\n')
    print(f'##### Testing done inst {instance} #####')
    print(f'total running time: {round(time() - start,2)}')
    print(f'incumbent: {round(incumbent,2)}')
    print(f'gap: {round(lab.compute_gap(instance, incumbent),2)}')
    print(f'time to find: {round(Results["time to find"],2)}')
    #print(f'best solution: {best_individual}')
    print('\n')

    ### Save performance
    a_file = open(path + f'Experimentation/Constructive/alpha/Adaptative-Reactive/results_{instance}', "wb")
    pickle.dump(Results, a_file)
    a_file.close()



# file = open('un archivo', 'rb')

# # dump information to that file
# data = pickle.load(file)

# # close the file
# file.close()

# print(data)

# %%
