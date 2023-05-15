import sys
import pickle
import pandas as pd
path: str = '/Users/juanbeta/My Drive/Research/Energy/E-CVRP-TW/Code/'
#path: str = 'C:/Users/jm.betancourt/Documents/Research/Energy//E-CVRP-TW/Code/'

from E_CVRP_TW import  E_CVRP_TW, Feasibility
env = E_CVRP_TW(path)

sys.path.insert(0,path+'Experimentation/')


instance = 'c108_21.txt'
env.load_data(instance)
env.generate_parameters()

feas_op = Feasibility()

individual = [['D', 'C30', 'C27', 'C26', 'C49', 'C50', 'C52', 'D'], 
              ['D', 'C62', 'C32', 'C34', 'C36', 'C38', 'S12', 'C10', 'C3', 'D'], 
              ['D', 'C22', 'C74', 'C86', 'C82', 'S1', 'C78', 'C85', 'C88', 'S1', 'D'], 
              ['D', 'C41', 'C46', 'C48', 'C51', 'C47', 'C43', 'C66', 'D'], 
              ['D', 'C63', 'C98', 'C96', 'C95', 'C94', 'C92', 'C93', 'C97', 'S3', 'D'], 
              ['D', 'C20', 'C60', 'C58', 'C53', 'C56', 'S16', 'C54', 'C99', 'S3', 'D'], 
              ['D', 'C29', 'C24', 'C28', 'C18', 'C17', 'C13', 'S7', 'D'], 
              ['D', 'C67', 'C72', 'C61', 'C57', 'S15', 'D'], 
              ['D', 'C42', 'C40', 'C45', 'C31', 'C33', 'C37', 'C35', 'S11', 'D'], 
              ['D', 'C75', 'C87', 'C1', 'C2', 'S3', 'C6', 'C7', 'C11', 'C91', 'D'], 
              ['D', 'C23', 'C65', 'C21', 'C25', 'S9', 'C84', 'C89', 'S1', 'D'], 
              ['D', 'C90', 'C59', 'C55', 'S16', 'D'], 
              ['D', 'C12', 'C16', 'C14', 'C19', 'C15', 'S7', 'C9', 'C8', 'D'], 
              ['D', 'C5', 'D'], 
              ['D', 'C64', 'C44', 'C68', 'C69', 'S15', 'C100', 'S3', 'D'], 
              ['D', 'C39', 'S12', 'C4', 'S5', 'D'], 
              ['D', 'C83', 'C81', 'C80', 'S19', 'D'], 
              ['D', 'C73', 'C70', 'C71', 'C76', 'S20', 'D'], 
              ['D', 'C77', 'C79', 'S19', 'D']]

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
