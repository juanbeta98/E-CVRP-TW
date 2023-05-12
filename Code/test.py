import sys
import pickle
import pandas as pd
path: str = '/Users/juanbeta/My Drive/Research/Energy/E-CVRP-TW/Code/'
#path: str = 'C:/Users/jm.betancourt/Documents/Research/Energy//E-CVRP-TW/Code/'

from E_CVRP_TW import  E_CVRP_TW, Feasibility
env = E_CVRP_TW(path)

sys.path.insert(0,path+'Experimentation/')


instance = 'r202C15.txt'
env.load_data(instance)
env.generate_parameters()

feas_op = Feasibility()

individual = [['D', 'C79', 'C25', 'S19', 'S15', 'C11', 'S5', 'D']]

visited = list(); count = 0

feasible, _ = feas_op.individual_check(env, individual, complete=False)
print(feasible)
# for route in individual:
for pos in individual[0]:
    if env.node_type[pos] == 'c' and pos not in visited:
        visited.append(pos)
        count += 1
print(f'The individual covers {count} costumers')
