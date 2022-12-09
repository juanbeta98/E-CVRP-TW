from numpy.random import seed, choice, random
from C_G_VRP_TW import  CG_VRP_TW, Constructive, Genetic, Feasibility, Reparator, Experiment
from time import time


'''
Constructive heuristic
'''
constructive = Constructive()
RCL_alpha = 0.5             # RCL alpha
RCL_mode = 'Hybrid'
End_slack = 20               # Slack to send veicles to depot


'''
Genetic algorithm
'''
Population_size = 500  
Elite_size = int(Population_size * 0.05)

genetic = Genetic(Population_size, Elite_size)
max_generations = 400
max_time = 3600 # 2 hours

crossover_rate = 0.7
mutation_rate = 0.5


'''
Feasibility operators
'''
feas_op = Feasibility()



'EXPERIMENT'
lab = Experiment()
Test = ['simple_crossover', '2opt', 'simple_insertion', 'smart_crossover' , 'Hybrid']
colors = ['blue', 'red', 'black', 'purple', 'green', 'orange']
Results = []
best_obj = 1e9
best_ind = []
Hardest = ['c204_21.txt', 'r201_21.txt', 'rc207_21.txt', 'rc208_21.txt']
# Hardest = ['c101_21.txt']#['c204_21.txt', 'r201_21.txt', 'rc207_21.txt', 'rc208_21.txt']

for instance in Hardest:
    for cross_mode in Test:

        '''
        Environment and general parameters
        '''
        env = CG_VRP_TW()
        env.load_data(instance)
        env.generate_parameters()
        rd_seed = 0
        seed(rd_seed)


        '''
        Repair operators
        '''
        repair_op = Reparator()
        repair_op.reset(env)


        '''
        Evolution
        '''
        Population, Distances, Times, generation, best_individual, incumbent, Incumbents, TTimes, start = lab.generate_intial_population(env, constructive, genetic, Population_size, RCL_alpha, RCL_mode, End_slack)
        Incumbents, TTimes, Results, best_individual, indcumbent = lab.evolution(Population, Distances, Times, Incumbents, TTimes, Results, generation, best_individual, incumbent, start, env, genetic, repair_op, Population_size, Elite_size, max_generations, max_time,  RCL_alpha, RCL_mode, End_slack, crossover_rate, cross_mode)

        with open(f'/Users/juanbeta/My Drive/2022-2/Metaheurísticas/Tareas/Tarea 4/CG-VRP-TW/Source/Results/R_gra_{instance}', 'w') as f:
            f.write(str(Results))

        if incumbent < best_obj:
            best_obj = incumbent
            best_ind = best_individual

    with open(f'/Users/juanbeta/My Drive/2022-2/Metaheurísticas/Tareas/Tarea 4/CG-VRP-TW/Source/Results/R_res_{instance}', 'w') as f:
        f.write(str(best_obj, best_ind))


    lab.save_performance(Results, instance, f'/Users/juanbeta/My Drive/2022-2/Metaheurísticas/Tareas/Tarea 4/CG-VRP-TW/Source/Results/plot_{instance[:-4]}.png')










# for cross_mode in Test:
#     print(cross_mode)
#     '''
#     ------------------------------------------------------------------------------------------------
#     Initial Population
#     ------------------------------------------------------------------------------------------------
#     - Population: (list of lists) Each individual's routes
#     - Objectives: (list of int) Each individual's total time
#     - Distances: (list of list) Each individual's total distance
#     - Times: (list of list) Each individual's route's time
#     - Resluts: (list of lists of lists): Each individual's [q,k]
#     '''
#     start = time()
#     const_parameters = (env, RCL_alpha, RCL_mode, End_slack)
#     Population, Distances, Times, Details = genetic.generate_population(constructive, const_parameters)
#     generation = 0
#     Phases['Constructive'] = round(time() - start,2)

#     incumbent = 1e9
#     best_individual, Incumbents, TTimes = [], [], []
#     for i in range(Population_size):
#         if Distances[i] < incumbent:
#             incumbent = Distances[i]
#             best_individual = [Population[i], Distances[i], Times[i], Details[i]]
#     Incumbents.append(incumbent)

#     # genetic.print_initial_population(env, start, Population, Distances, feas_op)
#     # print(f'Best generated individual {best_individual[0]} \n')


#     '''
#     ------------------------------------------------------------------------------------------------
#     Genetic proccess
#     ------------------------------------------------------------------------------------------------
#     '''
#     start_g = time()

#     while generation <= max_generations and time() - start_g <= max_time:
#         generation += 1

#         ### Selecting elite class
#         Elite = [x for _, x in sorted(zip(Distances,[i for i in range(Population_size)]))][:Elite_size] 


#         ### Selection: From a population, which parents are able to reproduce
#         # Fitness function
#         tots = sum(Distances)
#         probs = [Distances[i]/tots for i in range(len(Distances))]
        

#         # Intermediate population: Sample of the initial population    
#         inter_population = choice([i for i in range(Population_size)], size = int(Population_size - Elite_size), replace = True, p = probs)
#         inter_population = Elite + list(inter_population)


#         ### Tournament: Select two individuals and leave the best to reproduce
#         Parents = genetic.tournament(inter_population, Distances)
        

#         ### Recombination: Combine 2 parents to produce 1 offsprings 
#         New_chorizos = []
#         for index in range(len(Parents)):
#             couple = Parents[index]
#             chosen_parent = choice([couple[0], couple[1]])
#             chorizo = repair_op.build_chorizo(env, Population[chosen_parent])
            
#             if random() < crossover_rate:
#                 new_chorizo = genetic.crossover(env, Population[chosen_parent], chorizo, cross_mode, repair_op)
#             else:
#                 new_chorizo = chorizo
        
#             New_chorizos.append(new_chorizo)


#         ### Mutation: 'Shake' an individual
        

#         ### Repair solutions
#         Population, Distances, Times = [],[],[]
#         for i in range(Population_size):
#             parent, distance, distances, ttime, times  = repair_op.repair_chorizo(env, New_chorizos[i], RCL_alpha, RCL_mode, End_slack)

#             Population.append(parent);  Distances.append(distance); 
#             Times.append(times)#;  Details.append((distances, times))

#             if distance < incumbent:
#                 incumbent = distance
#                 best_individual = [parent, distance, ttime, (distances, times)]
            
#         # if generation % 50 == 0:
#         #     genetic.print_evolution(env, start, Population, generation, Distances, feas_op, incumbent)

#         Incumbents.append(incumbent)
#         TTimes.append(round(time()-start,2))
        
        
#     Results.append((Incumbents,TTimes))

# with open(f'/Users/juanbeta/My Drive/2022-2/Metaheurísticas/Tareas/Tarea 4/CG-VRP-TW/Source/Results/R_gra_{instance}', 'w') as f:
#     f.write(str(Results))


# with open(f'/Users/juanbeta/Library/CloudStorage/OneDrive-UniversidaddelosAndes/WS 2&3/Results/R_res_{instance}', 'w') as f:
#     f.write(str(best_individual))