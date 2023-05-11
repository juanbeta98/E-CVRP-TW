import sys
import pickle
import pandas as pd
path: str = '/Users/juanbeta/My Drive/Research/Energy/E-CVRP-TW/Code/'
#path: str = 'C:/Users/jm.betancourt/Documents/Research/Energy//E-CVRP-TW/Code/'

from E_CVRP_TW import  E_CVRP_TW, Feasibility
env = E_CVRP_TW(path)

sys.path.insert(0,path+'Experimentation/')
import plot_performance as plot


instance = 'r101_21.txt'
env.load_data(instance)
env.generate_parameters()

feas_op = Feasibility()

route = ['D', 'C70', 'C34', 'S4', 'D']

visited = list(); count = 0

feasible, _ = feas_op.individual_check(env, [route])
for pos in route:
    if env.node_type[pos] == 'c' and pos not in visited:
        visited.append(pos)
        count += 1
print(f'The individual covers {count} costumers')
