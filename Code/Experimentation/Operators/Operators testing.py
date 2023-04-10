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

constructive_verbose = True

constructive:Constructive = Constructive()


'''
Genetic algorithm
'''
Population_size:int = 1000
training_ind:int = int(round(Population_size * training_ind_prop,0))
Elite_size:int = int(Population_size * 0.5)

crossover_rate:float = 0.5
mutation_rate:float = 0.5

genetic: Genetic = Genetic(Population_size, Elite_size, crossover_rate, mutation_rate)

Operator:str = 'Testing'

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

for instance in test_bed:
    # Saving performance 
    Results = {}
    Incumbents = []
    ploting_Times = []

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
        print(f'################ Instance {instance} / {Operator} ################')
        print(f'########################################################################')
        print(f'- size: {len(list(env.C.keys()))}')
        print(f'- bkFO: {env.bkFO[instance]}')
        print(f'- bkEV: {env.bkEV[instance]}')

    # Population generation
    Population, Distances, Times, Details, incumbent, best_individual = genetic.generate_population(env, constructive, training_ind, 
                                                                                                    g_start, instance, constructive_verbose)
    Results['Constructive'] = best_individual
    Incumbents.append(incumbent)
    ploting_Times.append(time() - g_start)

    # Print progress
    if verbose: 
        print('\n')
        print(f'Population generation finished')
        print(f'- total running time: {round(time() - g_start,2)}s')
        print(f'- incumbent: {round(incumbent,2)}')
        print(f'- gap: {round(lab.compute_gap(env, instance, incumbent)*100,2)}%')
        print(f'- time to find: {round(best_individual[4],2)}s')
        print('\n')
        print(f'Genetic process started at {round(time() - g_start,2)}s')
        print(f'\nTime \t \tgen \t \tIncumbent \tgap \t \t#EV')
    
    # Genetic process
    generation = 0
    incumbent = 1e9
    while generation < 100:
        ### Elitism
        Elite = genetic.elite_class(Distances)

        ### Selection: From a population, which parents are able to reproduce
        # Intermediate population: Sample of the initial population 
        inter_population = genetic.intermediate_population(Distances)            
        inter_population = Elite + list(inter_population)

        ### Tournament: Select two individuals and leave the best to reproduce
        Parents = genetic.tournament(inter_population, Distances)

        # For operators, a chorizo format is needed, this representation is generated
        c_Population = repair_op.generate_chorizos_population(env, Population)

        ### Recombination: Combine 2 parents to produce 1 offsprings 
        New_c_Population = []
        for i in range(len(Parents)):
            chosen_parent = choice([Parents[i][0], Parents[i][1]])
            chorizo = c_Population[chosen_parent]

            # Crossover Operators

            New_c_Population.append(chorizo)

        ### Repair solutions
        Population, Distances, Times = [],[],[]
        for i in range(genetic.Population_size):
            individual, distance, distances, t_time, times  = repair_op.repair_chorizo(env, New_c_Population[i])
            # print(chech_all(env, individual))

            Population.append(individual);  Distances.append(distance); Times.append(t_time)


            if distance <= incumbent:
                incumbent = distance
                best_individual = [individual, distance, t_time, (distances, times), time() - g_start]

                if verbose: genetic.print_evolution(env, instance, time() - g_start, generation, incumbent, len(individual))
        
        ### Store progress
        Incumbents.append(incumbent)
        ploting_Times.append(time() - g_start)
        generation += 1


    # Print progress
    if verbose: 
        print('\n')
        print(f'Evolution finished finished')
        print(f'- total running time: {round(time() - g_start,2)}s')
        print(f'- incumbent: {round(incumbent,2)}')
        print(f'- gap: {round(lab.compute_gap(env, instance, incumbent)*100,2)}%')
        print(f'- time to find: {round(best_individual[4],2)}s')
        print('\n')

    # Storing overall performance
    Results['best individual'] = best_individual[0]
    Results['best distance'] = best_individual[1]
    Results['gap'] = round(lab.compute_gap(env, instance, incumbent)*100,2)
    Results['total time'] = best_individual[2]
    Results['others'] = best_individual[3]
    Results['time to find'] = best_individual[4]
    Results['incumbents'] = Incumbents
    Results['inc times'] = ploting_Times

    ### Save performance
    a_file = open(path + f'Experimentation/Operators/{Operator}/results_{instance}', "wb")
    pickle.dump(Results, a_file)
    a_file.close()




