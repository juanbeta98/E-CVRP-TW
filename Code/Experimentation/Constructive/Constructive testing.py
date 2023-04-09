from numpy.random import seed, choice
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
RCL_alpha_list: list[float] = [0.15, 0.25, 0.35, 0.5]
training_prop = 0.5
constructive: Constructive = Constructive()

RCL_criterion: str = 'Exo-Hybrid'

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
test_bed = env.sizes['l']

for instance in test_bed:
    # Saving performance 
    Results = {}
    Incumbents = []
    Times = []

    # Setting runnign times depending on instance size
    if instance in env.sizes['s']:  max_time = 30
    elif instance in env.sizes['m']:  max_time = 60*2
    else:   max_time = 60*6
   
    # Constructive
    start = time()
    env.load_data(instance)
    env.generate_parameters()
    constructive.reset(env)
    incumbent = 1e9
    ind = -1

    # Adaptative format
    alpha_performance = {alpha:0 for alpha in RCL_alpha_list}

    # Printing results
    if verbose: 
        print(f'\n\n########################################################################')
        print(f'################ Instance {instance} / {RCL_criterion} ################')
        print(f'########################################################################')
        print(f'- size: {len(list(env.C.keys()))}')
        print(f'- bkFO: {env.bkFO[instance]}')
        print(f'- bkEV: {env.bkEV[instance]}')

        print(f'\nTime \t \tInd \t \tIncumbent \tgap \t \t#Routes')

    
    while time() - start < max_time:

        # Storing individual
        ind += 1
        individual: list = []
        distance: float = 0
        distances: list = []
        t_time: float = 0
        times: list = []

        # Choosing alpha
        if time() - start < max_time * training_prop:
            RCL_alpha = choice(RCL_alpha_list)
        else:
            RCL_alpha = choice(RCL_alpha_list, p = [alpha_performance[alpha]/sum(alpha_performance.values()) for alpha in RCL_alpha_list])

        # Intitalizing environemnt
        constructive.reset(env)

        # Building individual
        while len(constructive.pending_c) > 0:
            if RCL_criterion == 'Exo-Hybrid': RCL_criterion_prime = choice(['distance', 'TimeWindow'])
            t, d, q, k, route = constructive.RCL_based_constructive(env, RCL_alpha, RCL_criterion_prime)
            individual.append(route)
            distance += d
            distances.append(d)
            t_time += t
            times.append(t)
        
        # Updating incumbent
        if distance < incumbent:
            incumbent = distance
            best_individual: list = [individual, distance, t_time, (distances, times), time() - start]
            constructive.print_constructive(env, instance, time() - start, ind, incumbent, len(individual))

        # Updating alpha
        alpha_performance[RCL_alpha] += 1/distance

        # Storing iteration performance
        Incumbents.append(incumbent)
        Times.append(time() - start)


    # Storing overall performance
    Results['best individual'] = best_individual[0]
    Results['best distance'] = best_individual[1]
    Results['gap'] = round(lab.compute_gap(env, instance, incumbent)*100,2)
    Results['total time'] = best_individual[2]
    Results['others'] = best_individual[3]
    Results['time to find'] = best_individual[4]
    Results['incumbents'] = Incumbents
    Results['inc times'] = Times


    ### Print performance
    print('\n')
    print(f'########## Performance ##########')
    print(f'total running time: {round(time() - start,2)}')
    print(f'incumbent: {round(incumbent,2)}')
    print(f'gap: {round(lab.compute_gap(env, instance, incumbent)*100,2)}%')
    print(f'time to find: {round(Results["time to find"],2)}')
    #print(f'best solution: {best_individual}')
    print('\n')

    ### Save performance
    a_file = open(path + f'Experimentation/Constructive/RCL criterion/{RCL_criterion}/results_{instance}', "wb")
    pickle.dump(Results, a_file)
    a_file.close()




