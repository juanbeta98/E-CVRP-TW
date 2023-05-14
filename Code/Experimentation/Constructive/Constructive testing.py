from numpy.random import seed, choice
from time import process_time
import sys
import pickle
import matplotlib.pyplot as plt

# path: str = '/Users/juanbeta/My Drive/Research/Energy/E-CVRP-TW/Code/' ##### CHANGE WHEN NECESSARY!!!
path: str = 'C:/Users/jm.betancourt/Documents/Research/Energy/E-CVRP-TW/Code/' ##### CHANGE WHEN NECESSARY!!!

sys.path.insert(0,path)
from E_CVRP_TW import  E_CVRP_TW, Constructive, Experiment, Feasibility

'''
General parameters
'''
start: float = process_time()

rd_seed: int = 0
seed(rd_seed)

verbose:bool = True
save_performance:bool = True
evaluate_feasibility:bool = True

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

RCL_criterions:list = ['distance','TimeWindow','Intra-Hybrid','Exo-Hybrid']
RCL_criterion: str = input('RCL criterion to be tested: ')

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
lab: Experiment = Experiment(path, [])
colors: list = ['blue', 'red', 'black', 'purple', 'green', 'orange']


'''
Instance testing
'''
for instance in env.instances:
    # Saving performance
    Results = dict()
    min_EV_Results = dict()

    Incumbents = list()
    Times = list()

    min_EV_Incumbents = list()
    min_EV_Times = list()

    # Setting runnign times depending on instance size
    if instance in env.sizes['s']:  max_time = 30
    elif instance in env.sizes['m']:  max_time = 60*2
    else:   max_time = 60*8
   
    # Constructive
    start = process_time()
    env.load_data(instance)
    env.generate_parameters()
    constructive.reset(env)
    incumbent = 1e9
    min_EV_incumbent = 1e9
    ind = -1

    # Adaptative format
    alpha_performance = {alpha:0 for alpha in RCL_alpha_list}

    # Printing results
    if verbose: 
        print(f'\n\n########################################################################')
        print(f'              Instance {instance} / {RCL_criterion} ')
        print(f'########################################################################')
        print(f'- size: {len(list(env.C.keys()))}')
        print(f'- bkFO: {env.bkFO[instance]}')
        print(f'- bkEV: {env.bkEV[instance]}')

        print(f'\nTime \t \tInd \t \tIncumbent \tgap \t \t#Routes')

    
    while process_time() - start < max_time:
        # Storing individual
        ind += 1
        individual: list = list()
        distance: float = 0
        distances: list = list()
        t_time: float = 0
        times: list = list()

        # Choosing alpha
        if process_time() - start < max_time * training_prop:
            RCL_alpha = choice(RCL_alpha_list)
        else:
            RCL_alpha = choice(RCL_alpha_list, p = [alpha_performance[alpha]/sum(alpha_performance.values()) for alpha in RCL_alpha_list])

        # Intitalizing environemnt
        constructive.reset(env)

        # Building individual
        while len(constructive.pending_c) > 0:
            if RCL_criterion == 'Exo-Hybrid': RCL_criterion_prime = choice(['distance', 'TimeWindow'])
            else:   RCL_criterion_prime = RCL_criterion
            t, d, q, k, route, dep_details = constructive.RCL_based_constructive(env, RCL_alpha, RCL_criterion_prime)
            individual.append(route)
            distance += d
            distances.append(d)
            t_time += t
            times.append(t)
        

            # Individual feasibility check
            if evaluate_feasibility:
                feasible, _ = feas_op.individual_check(env, [route], complete = False)
                assert feasible, f'!!!!!!!!!!!!!! \tNon feasible individual generated (ind {ind})'


        # Updating incumbent
        if distance < incumbent:
            incumbent = distance
            best_individual: list = [individual, distance, t_time, (distances, times), process_time() - start]
            #constructive.print_constructive(env, instance, process_time() - start, ind, incumbent, len(individual))
        
        # Updating best found solution with least number of vehicles
        if ind == 0 or \
            len(individual) < len(best_min_EV_individual[0]) or \
            distance < min_EV_incumbent and len(individual) <= len(best_min_EV_individual[0]):

            min_EV_incumbent = distance
            best_min_EV_individual: list = [individual, distance, t_time, (distances, times), process_time() - start]
            constructive.print_constructive(env, instance, process_time() - start, ind, min_EV_incumbent, len(individual))    
    

        # Updating alpha
        alpha_performance[RCL_alpha] += 1/distance

        # Storing iteration performance
        Incumbents.append(incumbent)
        Times.append(process_time() - start)
        min_EV_Incumbents.append(min_EV_incumbent)
        min_EV_Times.append(process_time() - start)


    # Storing overall performance
    Results['best individual'] = best_individual[0]
    Results['best distance'] = best_individual[1]
    Results['gap'] = round(lab.compute_gap(env, instance, incumbent)*100,2)
    Results['total time'] = best_individual[2]
    Results['others'] = best_individual[3]
    Results['time to find'] = best_individual[4]
    Results['incumbents'] = Incumbents
    Results['inc times'] = Times

    min_EV_Results['best individual'] = best_min_EV_individual[0]
    min_EV_Results['best distance'] = best_min_EV_individual[1]
    min_EV_Results['gap'] = round(lab.compute_gap(env, instance, min_EV_incumbent)*100,2)
    min_EV_Results['total time'] = best_min_EV_individual[2]
    min_EV_Results['others'] = best_min_EV_individual[3]
    min_EV_Results['time to find'] = best_min_EV_individual[4]
    min_EV_Results['incumbents'] = min_EV_Incumbents
    min_EV_Results['inc times'] = min_EV_Times


    ### Print performance
    print('\n')
    print(f'Evolution finished finished at {round(process_time() - start,2)}s')
    print('\tFO \tgap \tEV \ttime to find')
    print(f'dist\t{round(incumbent,1)} \t{round(lab.compute_gap(env, instance, incumbent)*100,1)}% \t{len(best_individual[0])} \t{round(best_individual[4],2)}')
    print(f'min_EV \t{round(min_EV_incumbent,1)} \t{round(lab.compute_gap(env, instance, min_EV_incumbent)*100,1)}% \t{len(best_min_EV_individual[0])} \t{round(best_min_EV_individual[4],2)}')
    print('\n')

    ### Save performance
    if save_performance:
        a_file = open(path + f'Experimentation/Constructive/RCL criterion/{RCL_criterion}/results_{instance}', "wb")
        pickle.dump([Results, min_EV_Results], a_file)
        a_file.close()





