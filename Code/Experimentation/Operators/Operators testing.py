from numpy.random import seed, choice, randint, random
from time import process_time
import sys
import pickle
import matplotlib.pyplot as plt

path: str = '/Users/juanbeta/My Drive/Research/Energy/E-CVRP-TW/Code/' ##### CHANGE WHEN NECESSARY!!!
# path: str = 'C:/Users/jm.betancourt/Documents/Research/Energy/E-CVRP-TW/Code/' ##### CHANGE WHEN NECESSARY!!!

sys.path.insert(0,path)
from E_CVRP_TW import  E_CVRP_TW, Constructive, Experiment, Genetic, Feasibility

'''
General parameters
'''
start: float = process_time()

rd_seed: int = 0
seed(rd_seed)

verbose:bool = True
saving:bool = True
evaluate_feasibility: bool = True

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
Population_size:int = 2000
training_ind:int = int(round(Population_size * training_ind_prop,0))
Elite_size:int = int(Population_size * 0.5)

crossover_rate:float = 0.5
mutation_rate:float = 0.5

genetic: Genetic = Genetic(Population_size, Elite_size, crossover_rate, mutation_rate)

Operator:str = 'Darwinian phi rate'

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

testing_times = {'s':2, 'm':5, 'l':8}


'''
Instance testing
'''
test_bed = [env.sizes['s'][0], env.sizes['m'][0], env.sizes['l'][0]]
# test_bed = env.generate_test_bed(['s','m'], 1)
# test_bed = [env.sizes['l'][0]]

for num, instance in enumerate(test_bed):
    # Saving performance 
    constructive_Results = dict()

    Results = dict()
    Incumbents = list()
    ploting_Times = list()

    min_EV_Results = dict()
    min_EV_Incumbents = list()
    min_EV_ploting_Times = list()
    

    # Setting runnign times depending on instance size
    max_time:int = 60
    if instance in env.sizes['s']:  max_time *= testing_times['s']
    elif instance in env.sizes['m']:  max_time *= testing_times['m']
    else:   max_time *= testing_times['l']
   
    # Constructive
    g_start = process_time()
    env.load_data(instance)
    env.generate_parameters()
    constructive.reset(env)

    # Printing progress
    if verbose: 
        print(f'\n\n########################################################################')
        print(f'             Instance {instance} / {Operator} / {round(num/len(test_bed),2)*100}%')
        print(f'########################################################################')
        print(f'- size: {len(list(env.C.keys()))}')
        print(f'- bkFO: {env.bkFO[instance]}')
        print(f'- bkEV: {env.bkEV[instance]}')

    # Population generation
    Population, Distances, Times, Details, incumbent, best_individual, min_EV_incumbent, best_min_EV_individual, RCL_alpha = \
                            genetic.generate_population(env, constructive, training_ind, g_start, instance, constructive_verbose)
    
    constructive_Results['best individual (min dist)'] = best_individual
    constructive_Results['best individual (min EV)'] = best_min_EV_individual

    Incumbents.append(incumbent)
    ploting_Times.append(best_individual[4])

    min_EV_Incumbents.append(min_EV_incumbent)
    min_EV_ploting_Times.append(best_min_EV_individual[4])

    # Print progress
    if verbose: 
        print('\n')
        print(f'Population generation finished at {round(process_time() - g_start,2)}s')
        print('\tFO \tgap \tEV \ttime to find')
        print(f'dist\t{round(incumbent,1)} \t{round(lab.compute_gap(env, instance, incumbent)*100,1)}% \t{len(best_individual[0])} \t{round(best_individual[4],2)}')
        print(f'min_EV \t{round(min_EV_incumbent,1)} \t{round(lab.compute_gap(env, instance, min_EV_incumbent)*100,1)}% \t{len(best_min_EV_individual[0])} \t{round(best_min_EV_individual[4],2)}')
        print('\n')
        print('Genetic process started')
        print(f'\nTime \t \tgen \t \tIncumbent \tgap \t \t#EV')
    
    # Genetic process
    generation = 0
    while process_time() - g_start < max_time:
        # print(f'Generation: {generation}')
        ### Elitism
        Elite = genetic.elite_class(Distances)

        ### Selection: From a population, which parents are able to reproduce
        # Intermediate population: Sample of the initial population 
        inter_population = genetic.intermediate_population(Distances)            
        inter_population = Elite + list(inter_population)


        ### Tournament: Select two individuals and leave the best to reproduce
        Parents = genetic.tournament(inter_population, Distances)


        ### Evolution
        New_Population:list = list();   New_Distances:list = list();   New_Times:list = list();   New_Details:list = list()
        for i in range(genetic.Population_size):
            individual_i = Parents[i][randint(0,2)]

            ### Shake
            

            ### Crossover
            # new_individual, new_distance, new_time, details = \
            #                     genetic.evaluated_insertion(env, Population[individual_i], Details[individual_i])

            ### Mutation
            new_individual, new_distance, new_time, details = \
                                genetic.Darwinian_phi_rate(env, constructive, Population[individual_i], Details[individual_i], RCL_alpha)

            
            # Individual feasibility check
            if evaluate_feasibility:
                feasible, _ = feas_op.individual_check(env, new_individual, complete = True)
                assert feasible, f'!!!!!!!!!!!!!! \tNon feasible individual generated (gen {generation}, ind {i}) / {new_individual}'

            # Store new individual
            New_Population.append(new_individual); New_Distances.append(new_distance); New_Times.append(new_time); New_Details.append(details)

            # Updating incumbent
            if new_distance < incumbent:
                incumbent = new_distance
                best_individual: list = [new_individual, new_distance, new_time, details, process_time() - start]

                # if verbose:
                #     genetic.print_evolution(env, instance, process_time() - g_start, generation, incumbent, len(new_individual))

                ### Store progress
                Incumbents.append(incumbent)
                ploting_Times.append(process_time() - g_start)

            # Updating best found solution with least number of vehicles
            if len(new_individual) < len(best_min_EV_individual[0]) or \
                new_distance < min_EV_incumbent and len(new_individual) <= len(best_min_EV_individual[0]):

                min_EV_incumbent = new_distance
                best_min_EV_individual: list = [new_individual, new_distance, new_time, details, process_time() - g_start]

                if verbose:
                    genetic.print_evolution(env, instance, process_time() - g_start, generation, incumbent, len(new_individual))

                ### Store progress
                min_EV_Incumbents.append(incumbent)
                min_EV_ploting_Times.append(process_time() - g_start)

        # Update
        Population = New_Population
        Distances = New_Distances
        Times = New_Times
        Details = New_Details
        generation += 1

    # Print progress
    if verbose: 
        print('\n')
        print(f'Evolution finished finished at {round(process_time() - g_start,2)}s')
        print('\tFO \tgap \tEV \ttime to find')
        print(f'dist\t{round(incumbent,1)} \t{round(lab.compute_gap(env, instance, incumbent)*100,1)}% \t{len(best_individual[0])} \t{round(best_individual[4],2)}')
        print(f'min_EV \t{round(min_EV_incumbent,1)} \t{round(lab.compute_gap(env, instance, min_EV_incumbent)*100,1)}% \t{len(best_min_EV_individual[0])} \t{round(best_min_EV_individual[4],2)}')
        print('\n')

    if saving:
        # Storing overall performance
        Results['best individual'] = best_individual[0]
        Results['best distance'] = best_individual[1]
        Results['gap'] = round(lab.compute_gap(env, instance, incumbent)*100,2)
        Results['total time'] = best_individual[2]
        Results['others'] = best_individual[3]
        Results['time to find'] = best_individual[4]
        Results['incumbents'] = Incumbents
        Results['inc times'] = ploting_Times

        # Storing overall performance for min EV
        min_EV_Results['best individual'] = best_min_EV_individual[0]
        min_EV_Results['best distance'] = best_min_EV_individual[1]
        min_EV_Results['gap'] = round(lab.compute_gap(env, instance, incumbent)*100,2)
        min_EV_Results['total time'] = best_min_EV_individual[2]
        min_EV_Results['others'] = best_min_EV_individual[3]
        min_EV_Results['time to find'] = best_min_EV_individual[4]
        min_EV_Results['incumbents'] = min_EV_Incumbents
        min_EV_Results['inc times'] = min_EV_ploting_Times

        ### Save performance
        a_file = open(path + f'Experimentation/Operators/{Operator}/results_{instance}', "wb")
        pickle.dump([constructive_Results, Results, min_EV_Results], a_file)
        a_file.close()