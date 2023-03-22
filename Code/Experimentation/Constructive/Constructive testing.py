from numpy.random import seed
from time import time
import sys
import pickle
sys.path.insert(0,'/Users/juanbeta/My Drive/Research/Energy/E-CVRP-TW/Code/')
from E_CVRP_TW import  E_CVRP_TW, Constructive, Genetic, Feasibility, Reparator, Experiment

'''
General parameters
'''
start: float = time()

rd_seed: int = 0
seed(rd_seed)

path: str = '/Users/juanbeta/My Drive/Research/Energy/E-CVRP-TW/Code/' ##### CHANGE WHEN NECESSARY!!!

'''
Environment
'''
env: E_CVRP_TW = E_CVRP_TW(path)


'''
Constructive heuristic
'''
RCL_alpha: float = 0.3              # RCL alpha
constructive: Constructive = Constructive(RCL_alpha)


'''
EXPERIMENTATION
Variable convention:
- Details: List of tuples (individual), where the tuple (distances, times) are discriminated per route
- best_individual: list with (individual, distance, time, details)
'''
lab: Experiment = Experiment()
colors: list = ['blue', 'red', 'black', 'purple', 'green', 'orange']
III: list = []


'''
Single instance testing
'''
max_time: int = 300 # 5 minutes

# Saving performance
Results = {}
Incumbents = []
Times = []


for instance in env.instances:
    # Constructive
    start = time()
    env.load_data(instance)
    env.generate_parameters()
    constructive.reset(env)
    incumbent = 1e9

    while time() - start < max_time:
        individual: list = []
        distance: float = 0
        distances: list = []
        t_time: float = 0
        times: list = []

        # Intitalizing environemnt
        constructive.reset(env)
        rou = 0
        while len(constructive.pending_c) > 0:
            t, d, q, k, route = constructive.RCL_based_constructive(env)
            individual.append(route)
            distance += d
            distances.append(d)
            t_time += t
            times.append(t)
            
            if distance < incumbent:
                incumbent = distance
                best_individual: list = [individual, distance, t_time, (distances, times)]

        Incumbents.append(incumbent)
        Times.append(time() - start)

    Results['best individual'] = best_individual[0]
    Results['best distance'] = best_individual[1]
    Results['total time'] = best_individual[2]
    Results['others'] = best_individual[3]
    Results['incumbents'] = Incumbents
    Results['inc times'] = Times


    ### Print performance
    print('\n')
    print(f'############## Testing done instance {instance} ################')
    print(f'total time: {round(time() - start,2)}')
    print(f'incumbent: {round(incumbent,2)}')
    #print(f'best solution: {best_individual}')
    print('\n')

    ### Save performance
    a_file = open(path + f'Experimentation/Constructive/Results/RCL Heuristic/results_{instance}', "wb")
    pickle.dump(Results, a_file)
    a_file.close()


# file = open('un archivo', 'rb')

# # dump information to that file
# data = pickle.load(file)

# # close the file
# file.close()

# print(data)