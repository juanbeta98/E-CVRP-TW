#%%
from numpy.random import seed
from time import time
import sys
import pickle
import matplotlib.pyplot as plt

#path: str = '/Users/juanbeta/My Drive/Research/Energy/E-CVRP-TW/Code/' ##### CHANGE WHEN NECESSARY!!!
path: str = 'C:/Users/jm.betancourt/Documents/Research/Energy/E-CVRP-TW/Code/' ##### CHANGE WHEN NECESSARY!!!

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
RCL_alpha: float = 0.5              # RCL alpha
constructive: Constructive = Constructive(RCL_alpha)


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

for instance in env.instances:
    # Saving performance
    Results = {}
    Incumbents = []
    Times = []

    if instance in env.sizes['s']:  
        max_time = 60*2
        RCL_alpha = 0.5

    elif instance in env.sizes['m']:   
        max_time = 60*4
        RCL_alpha = 0.3

    else:   
        max_time = 60*8
        RCL_alpha = 0.15


    if verbose: 
        print(f'\n\n################# Instance {instance} a = ({RCL_alpha})#################')
        print(f'Time \t \tInd \t \tIncumbent \t#Routes')

    # Constructive
    start = time()
    env.load_data(instance)
    env.generate_parameters()
    constructive.reset(env)
    incumbent = 1e9
    ind = -1

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
            t, d, q, k, route = constructive.RCL_based_constructive(env)
            individual.append(route)
            distance += d
            distances.append(d)
            t_time += t
            times.append(t)
            
        if distance < incumbent:
            incumbent = distance
            best_individual: list = [individual, distance, t_time, (distances, times), time() - start]
            constructive.print_constructive(time() - start, ind, incumbent, len(individual))

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
    print(f'total time: {round(time() - start,2)}')
    print(f'incumbent: {round(incumbent,2)}')
    print(f'time to find: {round(Results["time to find"],2)}')
    #print(f'best solution: {best_individual}')
    print('\n')

    ### Save performance
    a_file = open(path + f'Experimentation/Constructive/alpha/{str(RCL_alpha)}/results_{instance}', "wb")
    pickle.dump(Results, a_file)
    a_file.close()



# file = open('un archivo', 'rb')

# # dump information to that file
# data = pickle.load(file)

# # close the file
# file.close()

# print(data)

# %%
