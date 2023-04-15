

def crossover(self, env: E_CVRP_TW, chorizo: list, mode: str, repair_op: Reparator):
    if mode == 'simple_crossover':      return self.simple_crossover(chorizo)
    elif mode == '2opt':                return self.opt2(chorizo)
    elif mode == 'smart_crossover':     return self.smart_crossover(env, chorizo)
    elif mode == 'simple_insertion':    return self.simple_insertion(env, chorizo)
    elif mode == 'Hybrid':
        num = choice([0,1,2,3])
        if num == 0:    return self.simple_crossover(chorizo)
        elif num == 1:  return self.opt2(chorizo)
        elif num == 2:     return self.smart_crossover(env, chorizo)
        elif num == 3:     return self.simple_insertion(env, chorizo)

        # Errors
        else: return chorizo
    else: return chorizo
    #elif rand == 2:     return repair_op.build_chorizo(env, self.compound_crossover(env, parent))


'''
Change two possitions of a chorizo
'''
def simple_crossover(self, chorizo: list):
    pos1 = randint(0, len(chorizo) - 3)
    pos2 = randint(pos1 + 3, len(chorizo))
    if pos1 == pos2:
        if pos1 == len(chorizo) - 1:
            pos1 -= 1
        else:
            pos1 += 1

    chorizo[pos1], chorizo[pos2] = chorizo[pos2], chorizo[pos1]
    return chorizo


'''
Chose two positions of the chorizo and reverse the vector in between
'''
def opt2(self, chorizo: list):
    pos1 = randint(0, len(chorizo) - 3)
    pos2 = randint(pos1 + 3, len(chorizo))

    algo = [chorizo[pos2-i] for i in range(1,pos2-pos1)]
    return chorizo[:pos1+1] + algo + chorizo[pos2:]


'''
Smart crossover   
'''
def smart_crossover(self, env:E_CVRP_TW, chorizo: list[str]) -> list[str]:
    best = 0
    best_candidate = ''
    best_index = 1e9
    change = False
    
    pos  = randint(0, len(chorizo))
    for i in range(len(chorizo)):
        if i != pos and abs(i - pos) > 2:
            cand = chorizo[i]
            index = abs(env.C[chorizo[pos]]['DueDate'] - env.C[cand]['DueDate'] + env.C[chorizo[pos]]['ReadyTime'] - env.C[cand]['ReadyTime'])
            if pos != 0:
                index += env.dist[chorizo[pos-1],cand]
            if pos != len(chorizo) - 1:
                index += env.dist[cand,chorizo[pos+1]] 
            if i != 0:
                index += env.dist[chorizo[i-1],chorizo[pos]]
            if i != len(chorizo) - 1:
                index += env.dist[chorizo[pos],chorizo[i+1]] 

            if index < best_index:
                change = True
                best = i
                best_candidate = chorizo[i]

    if change:
        chorizo[pos], chorizo[best] = chorizo[best], chorizo[pos]
        return chorizo
    else:
        return chorizo


'''
Simple insertion
'''
def simple_insertion(self, env: E_CVRP_TW, chorizo: list[str]):
    pos1 = randint(0, len(chorizo))
    pos2 = randint(0, len(chorizo))

    if pos1 < pos2:
        return chorizo[:pos1] + chorizo[pos1+1:pos2] + [chorizo[pos1]] + chorizo[pos2:] 
    elif pos1 > pos2:
        return chorizo[:pos2] + [chorizo[pos1]] + chorizo[pos2:pos1] +  chorizo[pos1+1:] 
    else:
        return chorizo

















# For operators, a chorizo format is needed, this representation is generated
c_Population = repair_op.generate_chorizos_population(env, Population)

### Recombination: Combine 2 parents to produce 1 offsprings 
New_c_Population = []
for i in range(len(Parents)):
    chosen_parent = choice([Parents[i][0], Parents[i][1]])
    chorizo = c_Population[chosen_parent]

    # Crossover
    chorizo = genetic.crossover(env, chorizo, 'simple_insertion', repair_op)

    # mutation
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