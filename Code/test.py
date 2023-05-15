import sys
import pickle
import pandas as pd
path: str = '/Users/juanbeta/My Drive/Research/Energy/E-CVRP-TW/Code/'
#path: str = 'C:/Users/jm.betancourt/Documents/Research/Energy//E-CVRP-TW/Code/'

from E_CVRP_TW import  E_CVRP_TW, Feasibility
env = E_CVRP_TW(path)

sys.path.insert(0,path+'Experimentation/')


instance = 'c109_21.txt'
env.load_data(instance)
env.generate_parameters()

feas_op = Feasibility()

individual = [['D', 'C30', 'C32', 'C35', 'C37', 'C52', 'S13', 'C49', 'D'], 
                ['D', 'C29', 'C28', 'C10', 'C5', 'C8', 'C7', 'C3', 'D'], 
                ['D', 'C86', 'C90', 'C84', 'C88', 'C91', 'C89', 'S1', 'D'], 
                ['D', 'C23', 'C27', 'C24', 'C34', 'C38', 'C39', 'C31', 'S11', 'C43', 'C50', 'D'], 
                ['D', 'C70', 'C79', 'S19', 'C78', 'C80', 'C81', 'S20', 'D'], 
                ['D', 'C87', 'C82', 'C85', 'C83', 'S1', 'C99', 'C9', 'D'], 
                ['D', 'C67', 'C42', 'C40', 'C48', 'C45', 'C46', 'C47', 'C51', 'S13', 'C57', 'D'], 
                ['D', 'C59', 'C41', 'C44', 'C68', 'S15', 'C77', 'S19', 'C55', 'D'], 
                ['D', 'C25', 'C21', 'C36', 'C33', 'S11', 'C4', 'C17', 'S7', 'D'], 
                ['D', 'C73', 'C71', 'C76', 'S20', 'C26', 'S9', 'D'], 
                ['D', 'C65', 'C63', 'C62', 'C69', 'C58', 'C56', 'C53', 'C54', 'S16', 'D'], 
                ['D', 'C75', 'C1', 'C98', 'C93', 'C94', 'C92', 'C97', 'C100', 'S3', 'D'], 
                ['D', 'C95', 'C96', 'S3', 'C16', 'C19', 'C18', 'S9', 'D'], 
                ['D', 'C22', 'C20', 'C74', 'S15', 'C6', 'C2', 'C11', 'D'], 
                ['D', 'C12', 'C14', 'C15', 'C13', 'S7', 'S0', 'D'], 
                ['D', 'C64', 'C60', 'S15', 'C61', 'C72', 'C66', 'D']]

visited = list(); count = 0

feasible, _ = feas_op.individual_check(env, individual, instance, complete=True)
print(feasible)
print(_[2][3][0])
# for route in individual:
for pos in individual[0]:
    if env.node_type[pos] == 'c' and pos not in visited:
        visited.append(pos)
        count += 1
print(f'The individual covers {count} costumers')
