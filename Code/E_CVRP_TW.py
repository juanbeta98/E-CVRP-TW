'''
Capacitated Electric Vehicle Routing Problem with Time windows
E-CVRP-TW

Authors:
Juan Betancourt
jm.betancourt@uniandes.edu.co
'''
from copy import deepcopy
from time import process_time
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import random, choice, seed, randint
import networkx as nx
import os
import pickle
import scipy.stats as stats

''' E_CVRP_TW Class: Parameters and information '''
class E_CVRP_TW(): 

    def __init__(self):
        self.path = '/Users/juanbeta/My Drive/Research/Energy/E-CVRP-TW/Code/'
        self.colors = ['black', 'red', 'green', 'blue', 'purple', 'orange', 'pink', 'grey', 
                       'yellow', 'tab:red', 'tab:green', 'tab:blue', 'tab:purple', 'tab:orange', 
                       'tab:pink', 'tab:grey', 
                       'black', 'red', 'green', 'blue', 'purple', 'orange', 'pink', 'grey', 
                       'yellow', 'tab:red', 'tab:green', 'tab:blue', 'tab:purple', 'tab:orange', 
                        'tab:pink', 'tab:grey']

        self.instances = os.listdir(self.path + 'Instances/')
        self.instances.remove('readme.txt')
        self.instances.sort()
        if '.DS_Store' in self.instances:
            self.instances.remove('.DS_Store')

        self.sizes = {
                        's': ['c103C5.txt', 'rc208C5.txt', 'c101C5.txt', 'c206C5.txt', 'rc108C5.txt', 'r104C5.txt', 'r203C5.txt', 
        'r105C5.txt', 'c208C5.txt', 'rc105C5.txt', 'r202C5.txt', 'rc204C5.txt'], 
                        'm': ['rc108C10.txt', 'r202C15.txt', 'r103C10.txt', 'rc205C10.txt', 'rc108C15.txt', 'rc201C10.txt', 
        'c205C10.txt', 'rc102C10.txt', 'c104C10.txt', 'r105C15.txt', 'c106C15.txt', 'c103C15.txt', 'r102C10.txt', 'rc204C15.txt', 
        'c101C10.txt', 'c202C10.txt', 'r201C10.txt', 'r203C10.txt', 'c202C15.txt', 
        'r102C15.txt', 'r209C15.txt', 'rc202C15.txt', 'c208C15.txt', 'rc103C15.txt'], 
                        'l': ['c202_21.txt', 'r201_21.txt', 'rc105_21.txt', 'r102_21.txt', 'rc206_21.txt', 'c101_21.txt',
        'r110_21.txt', 'r112_21.txt', 'c103_21.txt', 'rc208_21.txt', 'rc204_21.txt', 'rc107_21.txt', 'r203_21.txt', 'r211_21.txt', 
        'r104_21.txt', 'r108_21.txt', 'c107_21.txt', 'c204_21.txt', 'rc103_21.txt', 'c208_21.txt', 'r207_21.txt', 'r205_21.txt', 
        'rc101_21.txt', 'c206_21.txt', 'r209_21.txt', 'c105_21.txt', 'r106_21.txt', 'rc202_21.txt', 'c109_21.txt', 'rc207_21.txt', 
        'r103_21.txt', 'r111_21.txt', 'rc108_21.txt', 'c203_21.txt', 'rc104_21.txt', 'r202_21.txt', 'rc106_21.txt', 'c201_21.txt', 
        'r210_21.txt', 'c102_21.txt', 'r101_21.txt', 'rc205_21.txt', 'c205_21.txt', 'r206_21.txt', 'rc102_21.txt', 'r105_21.txt', 
        'rc201_21.txt', 'c106_21.txt', 'r109_21.txt', 'c104_21.txt', 'c108_21.txt', 'rc203_21.txt', 'r107_21.txt', 'r204_21.txt', 
        'r208_21.txt', 'c207_21.txt']
                    }
        self.sizes['s'].sort()
        self.sizes['m'].sort()
        self.sizes['l'].sort()
        
        self.bkFO = { 
                     'c101C5.txt': 257.75, 'c103C5.txt': 176.05, 'c206C5.txt': 242.55, 'c208C5.txt': 158.48, 'r104C5.txt': 136.69, 'r105C5.txt': 156.08,
                     'r202C5.txt': 128.78, 'r203C5.txt': 179.06, 'rc105C5.txt': 241.3, 'rc108C5.txt': 253.93, 'rc204C5.txt': 176.39, 'rc208C5.txt': 167.98,
                     
                     'c101C10.txt': 393.76, 'c104C10.txt': 273.93, 'c202C10.txt': 304.06, 'c205C10.txt': 228.28, 'r102C10.txt': 249.19, 'r103C10.txt': 207.05, 
                     'r201C10.txt': 241.51, 'r203C10.txt': 218.21, 'rc102C10.txt': 423.51, 'rc108C10.txt': 345.93, 'rc201C10.txt': 412.86, 'rc205C10.txt': 325.98,

                     'c103C15.txt': 384.29, 'c106C15.txt': 275.13, 'c202C15.txt': 383.62, 'c208C15.txt': 300.55, 'r102C15.txt': 413.93, 'r105C15.txt': 336.15,
                     'r202C15.txt': 358, 'r209C15.txt': 313.24, 'rc103C15.txt': 397.67, 'rc108C15.txt': 370.25, 'rc202C15.txt': 394.39, 'rc204C15.txt': 407.45,

                     'c101_21.txt': 1053.83, 'c102_21.txt': 1056.47, 'c103_21.txt': 1002.03, 'c104_21.txt': 979.51, 'c105_21.txt': 1075.37, 'c106_21.txt': 1057.97,
                     'c107_21.txt': 1031.56, 'c108_21.txt': 1015.73, 'c109_21.txt': 1036.64, 'c201_21.txt': 645.16, 'c202_21.txt': 645.16, 'c203_21.txt': 644.98,
                     'c204_21.txt': 636.43, 'c205_21.txt': 641.13, 'c206_21.txt': 638.17, 'c207_21.txt': 638.17, 'c208_21.txt': 638.17, 'r101_21.txt': 1670.8,
                     'r102_21.txt': 1495.31, 'r103_21.txt': 1299.17, 'r104_21.txt': 1088.43, 'r105_21.txt': 1401.24, 'r106_21.txt': 1334.66, 'r107_21.txt': 1154.52,
                     'r108_21.txt': 1050.04, 'r109_21.txt': 1294.05, 'r110_21.txt': 1126.74, 'r111_21.txt': 1106.19, 'r112_21.txt': 1026.52, 'r201_21.txt': 1264.82,
                     'r202_21.txt': 1052.32, 'r203_21.txt': 895.91, 'r204_21.txt': 790.57, 'r205_21.txt': 988.67, 'r206_21.txt': 925.2, 'r207_21.txt': 848.53,
                     'r208_21.txt': 736.6, 'r209_21.txt': 872.36, 'r210_21.txt': 847.06, 'r211_21.txt': 847.45, 'rc101_21.txt': 1731.07, 'rc102_21.txt': 1554.61,
                     'rc103_21.txt': 1351.15, 'rc104_21.txt': 1238.56, 'rc105_21.txt': 1475.31, 'rc106_21.txt': 1437.96, 'rc107_21.txt': 1275.89, 'rc108_21.txt': 1209.61,
                     'rc201_21.txt': 1444.94, 'rc202_21.txt': 1412.91, 'rc203_21.txt': 1073.98, 'rc204_21.txt': 885.35, 'rc205_21.txt': 1321.75, 'rc206_21.txt': 1190.75,
                     'rc207_21.txt': 995.52, 'rc208_21.txt': 837.82,
                    }
        
        self.bkEV = {
                     'c101C5.txt': 2, 'c103C5.txt': 1, 'c206C5.txt': 1, 'c208C5.txt': 1, 'r104C5.txt': 2, 'r105C5.txt': 2, 'r202C5.txt': 1,
                     'r203C5.txt': 1, 'rc105C5.txt': 2, 'rc108C5.txt': 1, 'rc204C5.txt': 1, 'rc208C5.txt': 1,

                     'c101C10.txt': 3, 'c104C10.txt': 2, 'c202C10.txt': 1, 'c205C10.txt': 2, 'r102C10.txt': 3, 'r103C10.txt': 2,
                     'r201C10.txt': 1, 'r203C10.txt': 1, 'rc102C10.txt': 4, 'rc108C10.txt': 3, 'rc201C10.txt': 1, 'rc205C10.txt': 2,

                     'c103C15.txt': 3, 'c106C15.txt': 3, 'c202C15.txt': 2, 'c208C15.txt': 2, 'r102C15.txt': 5, 'r105C15.txt': 4, 
                     'r202C15.txt': 2, 'r209C15.txt': 1, 'rc103C15.txt': 4, 'rc108C15.txt': 3, 'rc202C15.txt': 2, 'rc204C15.txt': 1,

                     'c101_21.txt': 12, 'c102_21.txt': 11, 'c103_21.txt': 10, 'c104_21.txt': 10, 'c105_21.txt': 11, 'c106_21.txt': 11, 'c107_21.txt': 11, 'c108_21.txt': 10,
                     'c109_21.txt': 10, 'c201_21.txt': 4, 'c202_21.txt': 4, 'c203_21.txt': 4, 'c204_21.txt': 4, 'c205_21.txt': 4, 'c206_21.txt': 4, 'c207_21.txt': 4, 
                     'c208_21.txt': 4, 'r101_21.txt': 18, 'r102_21.txt': 16, 'r103_21.txt': 13, 'r104_21.txt': 11, 'r105_21.txt': 14, 'r106_21.txt': 13, 'r107_21.txt': 12, 
                     'r108_21.txt': 11, 'r109_21.txt': 12, 'r110_21.txt': 11, 'r111_21.txt': 12, 'r112_21.txt': 11, 'r201_21.txt': 3, 'r202_21.txt': 3, 'r203_21.txt': 3, 
                     'r204_21.txt': 2, 'r205_21.txt': 3, 'r206_21.txt': 3, 'r207_21.txt': 2, 'r208_21.txt': 2, 'r209_21.txt': 3, 'r210_21.txt': 3, 'r211_21.txt': 2, 
                     'rc101_21.txt': 16, 'rc102_21.txt': 15, 'rc103_21.txt': 13, 'rc104_21.txt': 11, 'rc105_21.txt': 14, 'rc106_21.txt': 13, 'rc107_21.txt': 12, 'rc108_21.txt': 11, 
                     'rc201_21.txt': 4, 'rc202_21.txt': 3, 'rc203_21.txt': 3, 'rc204_21.txt': 3, 'rc205_21.txt': 3, 'rc206_21.txt': 3, 'rc207_21.txt': 3, 'rc208_21.txt': 3,
                    }


    ''' Load data from txt file '''
    def load_data(self,instance:str):
        file = open(self.path + 'Instances/'+instance,mode='r');     file = file.readlines()
        
        fila = 1
        att = [i for i in str(file[fila]).split(' ') if i != '']
        self.T = float(att[6])
        self.D = {'ID':att[0], 'type': att[1], 'x': float(att[2]), 'y':float(att[3])}      # Depot
        fila += 1
        
        self.S, self.Stations  = dict(), list()             # Charging stations
        self.C, self.Costumers = dict(), list()             # Costumers
        self.node_type = {'D':'D'}

        while True:
            
            ID, typ, x, y = [i for i in str(file[fila]).split(' ') if i != ''][0:4]
            x = float(x);   y = float(y)

            if typ == 'f':
                self.S[ID] = {'type': 's', 'x': x, 'y': y}
                self.Stations.append(ID)
                self.node_type[ID] = 's'

            else:
                d, ReadyTime, DueDate, ServiceTime = [float(i) for i in [i for i in str(file[fila]).split(' ') if i != ''][4:8]]
                self.C[ID] = { 'type': typ, 'x': x, 'y': y, 'd': d, 'ReadyTime': ReadyTime, 
                               'DueDate': DueDate, 'ServiceTime': ServiceTime}
                self.Costumers.append(ID)
                self.node_type[ID] = 'c'

            fila += 1
            if file[fila][0] == "\n":
                break
                
        fila += 1
        self.Q = float([i for i in str(file[fila]).split(' ') if i != ''][5][1:-2]); fila += 1      # Vehicle max charge
        self.K = float([i for i in str(file[fila]).split(' ') if i != ''][4][1:-2]); fila += 1      # Vehicle max load
        self.r = float([i for i in str(file[fila]).split(' ') if i != ''][4][1:-2]); fila += 1      # Fuel consumption rate
        self.g = float([i for i in str(file[fila]).split(' ') if i != ''][4][1:-2]); fila += 1      # Vehichle refueling rate
        self.v = float([i for i in str(file[fila]).split(' ') if i != ''][3][1:-2])                 # Vehicles speed

        
    ''' Compute several parameters from intial file '''
    def generate_parameters(self):
        self.compute_distances()
        self.closest_stations()


    ''' Generate test bed of instances 
    - Sets of size 'size' of a given inst_size
    - Sets of size 'size' of a given list of instances
    '''
    def generate_test_batch(self,computer='mac'):
        if computer == 'mac':
            return self.instances[:int(3.5*len(self.instances)/10)]
        else: 
            return self.instances[int(3.5*len(self.instances)/10):]
    
    

    ''' Generate test bed of instances 
    - Sets of size 'size' of a given inst_size
    - Sets of size 'size' of a given list of instances
    '''
    def generate_test_batch_per_size(self, inst_set:str or list, size:int):
        test_bed = list()
        if type(inst_set) == str:
            inst_set = self.sizes[inst_set]
        else: 
            inst_sett = list()
            for sz in inst_set:
                inst_sett.extend(self.sizes[sz])
            inst_set = inst_sett

        if size == 1:
            return inst_set 
        
        for i in range(size):
            if i != size-1:
                test_bed.append(inst_set[i * int(len(inst_set)/size): (i+1) * int(len(inst_set)/size)])
            else:
                test_bed.append(inst_set[i * int(len(inst_set)/size):])

        return test_bed


    ''' Generate list with pending instances to test '''
    def generate_missing_instances(self, path):
        inst_set = deepcopy(self.instances)
        completed_instances = os.listdir(self.path + 'Experimentation/' + path)
        if '.DS_Store' in completed_instances:
            completed_instances.remove('.DS_Store')
        completed_instances.sort()
        for inst in completed_instances:
            inst_set.remove(inst[8:])

        return inst_set


    '''
    Distance between:
    - Depot to stations
    - Depot to costumers
    - Costumers to costumers
    - Costumers to stations
    '''
    def compute_distances(self):
        self.dist = dict()
        for c1 in self.Costumers:
            # Depot-Costumers
            self.dist['D',c1] = self.euclidean_distance(self.C[c1], self.D)
            self.dist[c1, 'D'] = self.dist['D',c1]

            # Costumers-Costumers
            for c2 in self.Costumers:
                if c1 != c2:
                    self.dist[c1, c2] = self.euclidean_distance(self.C[c1], self.C[c2])
                    self.dist[c2, c1] = self.dist[c1, c2] 
            
            # Costumers-Stations
            for s in self.Stations:
                self.dist[c1, s] = self.euclidean_distance(self.C[c1], self.S[s])
                self.dist[s, c1] = self.dist[c1, s]
        
        # Depot-Stations
        for s in self.Stations:
            self.dist['D',s] = self.euclidean_distance(self.S[s], self.D)
            self.dist[s, 'D'] = self.dist['D',s]

            # Stations-Stations
            for s1 in self.Stations:
                if s != s1:
                    self.dist[s, s1] = self.euclidean_distance(self.C[c1], self.S[s])
                    self.dist[s1, s] = self.dist[s, s1]
    
    
    ''' Compute euclidean distance between two nodes '''
    def euclidean_distance(self,x: float, y: float):
        distance = ((x['x'] - y['x'])**2 + (x['y'] - y['y'])**2)**(1/2)
        return distance


    ''' Generates the closest station to each costumer '''
    def closest_stations(self):
        self.closest = {}
        for c in self.C:
            new_dic = {s:self.dist[c,s] for s in self.Stations}
            self.closest[c] = min(new_dic, key = new_dic.get)
    

    ''' Plots nodes of the VRP '''
    def render(self, routes: list, save:bool = False):
        G = nx.MultiDiGraph()

        # Nodes
        node_list = ['D']
        node_list += self.Stations
        node_list += self.Costumers
        G.add_nodes_from(node_list)

        node_color = ['purple']
        node_color += ['tab:green' for s in self.Stations[1:]]
        node_color += ['tab:blue' for c in self.Costumers]
        nodes_to_draw = deepcopy(node_list)
        nodes_to_draw.remove('S0')   

        # Edges
        edges = list()
        edge_colors = list()
        orders = dict()
        for i in range(len(routes)):
            route = routes[i]
            for node in range(len(route) - 1):
                edge = (route[node], route[node + 1])
                edges.append(edge)
                orders[edge] = i

        G.add_edges_from(edges) 
        edges = G.edges()
        for edge in edges:
            color = self.colors[orders[edge]]
            edge_colors.append(color)

        pos = {c: (self.C[c]['x'], self.C[c]['y']) for c in self.Costumers}
        pos.update({s: (self.S[s]['x'], self.S[s]['y']) for s in self.Stations})
        pos['D'] = (self.D['x'], self.D['y'])

        nx.draw_networkx(G, pos = pos, with_labels = True, nodelist = nodes_to_draw, 
                         node_color = node_color, edge_color = edge_colors, alpha = 0.8, 
                         font_size = 7, node_size = 200)
        if save:
            plt.savefig(self.path + 'sexting.png', dpi = 600)
        plt.show()
        

    def plot_evolution(self, Incumbents: list, color: str = 'purple'):
        plt.plot(range(len(Incumbents)), Incumbents, color = color)
        plt.title('Evolution of best individual')
        plt.xlabel('Iterations')
        plt.ylabel('Objective (distance)')
        plt.show()


    def detail_route(self, route: list):
        # TODO: update method to display distances
        t, d, q, k = 0, 0, self.Q, 0
        print('########## Route started ##########\n')
        for i in range(len(route) - 1):
            node = route[i]
            target = route[i+1]
            if target in self.Costumers:
                print(f'Travel from {node} to {target}')
                print(f'- travel time {round(self.dist[node, target] / self.v,2)}')
                t += round(self.dist[node, target]/self.v,2)
                print(f'- t: {t}')
                d += round(self.dist[node, target],2)
                print(f'- d: {d}')
                q -= round(self.dist[node, target] / self.r,2)
                print(f'- q: {q}')
                print('\n')


                print(f'COSTUMER {target}')
                print(f'Rea \tDue \tSt')
                print(f'{round(self.C[target]["ReadyTime"],2)} \t{round(self.C[target]["DueDate"],2)} \t{round(self.C[target]["ServiceTime"],2)}')
                print('------------------------')
                t = round(max(t, self.C[target]['ReadyTime']),2)
                print(f'- start_time: {t}')
                t += self.C[target]['ServiceTime']
                print(f'- departure_time: {t}')
                k += self.C[target]['d']
                print(f'- k: {k}')
            
            elif target in self.Stations:
                print(f'Travel from {node} to {target}')
                print(f'- travel time {round(self.dist[node, target] / self.v,2)}')
                t += round(self.dist[node, target]/self.v,2)
                print(f'- t: {t}')
                d += round(self.dist[node, target],2)
                print(f'- d: {d}')
                q -= round(self.dist[node, target] / self.r,2)
                print(f'- q: {q}')
                print('\n')
                print(f'STATION {target}')
                
                t += round((self.Q - q) * self.g,2)
                print(f'- t: {round(t,2)}')
                print(f'- charged: {self.Q - q}')
                q = self.Q
                print(f'- q: {q}')
        
            print('\n')
        
        print('########## Route finished ##########\n')
        print(f'Total time: {t}')
        print(f'Total load: {k}')
        print(f'total distance: {d}')



'''
Factibility checks
'''
class Feasibility():

    def __init__(self):
        pass
    

    ''' Check feasibility for al whole population
    '''
    def population_check(self, env: E_CVRP_TW, Population: list):
        # Intial population feasibility check
        datas = []
        for individual in Population:
            feasible = self.individual_check(env, individual)
            datas.append(int(feasible))
        return datas


    ''' Checks the feasibility of an individual (all the routes)
    '''
    def individual_check(self, env: E_CVRP_TW, individual: list, instance:str, complete = False):
        feasible = True
        distance = 0
        distances = list()
        ttime = 0
        times = list()
        loads = list()
        dep_t_details = list()
        dep_q_details = list()

        if complete:
            count = 0
            visited = list()

            
        for num, route in enumerate(individual):
            t: float = 0
            d: float = 0
            q: float = env.Q
            k: int = 0 

            dep_t = [0] # Auxiliary list with departure times
            dep_q = [env.Q]

            for i in range(len(route)-1):
                node = route[i]
                target = route[i + 1]
                station_depot_route = False
                if target != 'D' and env.node_type[target]=='s' and route[i+2] == 'D':
                    station_depot_route = True
                feasible, t, q, k, _ = self.transition_check(env, node, target, station_depot_route, t, q, k)

                d += env.dist[node,target]

                if i < len(route)-2:
                    dep_t.append(t)
                    dep_q.append(q)

                if complete and env.node_type[target] == 'c':
                    if target in visited:
                        feasible = False
                        print(f'Costumer visited more than once')
                        break
                    else:
                        visited.append(target)
                        count += 1

                if not feasible:
                    if not _[0]: print(f'❌ Instance {instance} - Not time or energy feasiblefeasible')
                    if not _[1]: print(f'❌ Instance {instance} - Not total load feasible feasible')
                    if not _[2]: print(f'❌ Instance {instance} - Not time window feasible')
                    break
            
            distance += d
            distances.append(d)
            ttime += t
            times.append(t)
            loads.append(k)
            dep_t_details.append(dep_t)
            dep_q_details.append(dep_q)

            if not feasible:
                break
        
        if complete and feasible and count != len(env.Costumers):
            feasible = False
            print(f'Individual only visited {count} costumers')

        return feasible, (distance, ttime, (distances, times, loads, (dep_t_details, dep_q_details)))

                
    def transition_check(self, env: E_CVRP_TW, node: str, target: str, station_depot_route:bool, t: float, q: float, k: int):
        if target != 'D' and env.node_type[target] == 'c':    time_window_feasible = self.time_window_check(env, node, target, t)
        else:           time_window_feasible = True
        time_energy_feasible, t, q = self.time_energy_check(env, node, target, station_depot_route, t, q)
        load_feasible, k = self.load_check(env, target, k)

        if time_energy_feasible and load_feasible and time_window_feasible:
            return True, t, q ,k, ''
        else:
            return False, t, q, k, (time_energy_feasible,load_feasible,time_window_feasible)


    def time_window_check(self, env:E_CVRP_TW, node:str, target:str, t:float):
        return t <= env.C[target]['DueDate']
    

    def time_energy_check(self, env: E_CVRP_TW, node: str, target: str, station_depot_route:bool, t: float, q: float):
        travel_time = env.dist[node,target] / env.v

        q -= env.dist[node, target] / env.r
        if q < -0.001: return False, t, q

        # Total time check
        if target in env.Costumers:
            arrival = max(t+travel_time, env.C[target]['ReadyTime'])
            new_t = arrival + env.C[target]['ServiceTime']
            if new_t > env.T:      
                return False, t, q
            else:                       
                return True, new_t, q

        elif target in env.Stations:
            if not station_depot_route:
                recharge = (env.Q - q)
            else:
                recharge = max(0, env.dist[target,'D']/env.r - q)
            
            update = travel_time + recharge * env.g

            if t + update > env.T:      
                return False, t, q
            else:
                return True, t+update, q + recharge

        elif target == 'D':
            update = travel_time 
            if t + update > env.T:      
                return False, t, q
            else:                       
                return True, t+update, q
            

    def load_check(self, env:E_CVRP_TW, target:str, k:int):
        if target in env.Costumers:
            if k + env.C[target]['d'] <= env.K:
                k += env.C[target]['d']
                return True, k
            else:
                return False, k
        else:
            return True, k




'''
Algorithms Class: Compilation of heuristics to generate a feasible route
- RCL based constructive
'''
class Constructive(Feasibility):

    def __init__(self):
        pass
    
    ''' Reset the environment to restart experimentation (another parent) '''
    def reset(self, env: E_CVRP_TW):
        self.pending_c = deepcopy(env.Costumers)


    ''' BUILD RLC DEPENDING ON:
    - DISTANCES
    - OPENING TIME WINDOW
    '''
    def generate_candidate_from_RCL(self,env:E_CVRP_TW,RCL_alpha:float,RCL_criterion:str,node:str,t:float,q:float,k:int):
        feasible_candidates:list = list()
        feasible_energy_candidates:list = list()
        max_crit:float = -1e9
        min_crit:float = 1e9

        RCL_mode = RCL_criterion
        if RCL_criterion == 'Intra-Hybrid': RCL_mode = choice(['distance', 'TimeWindow'])

        energy_feasible: bool = False     # Indicates if there is at least one candidate feasible by time and load but not charge
        for target in self.pending_c:
            distance = env.dist[node,target]

            global_c,energy_feasible, easible_energy_candidates = self.evaluate_candidate(env, target, distance, t, q, k, energy_feasible, feasible_energy_candidates)

            if global_c:
                feasible_candidates.append(target)

                if RCL_mode == 'distance':      crit = distance
                elif RCL_mode == 'TimeWindow':  crit = env.C[target]['DueDate']
                
                max_crit = max(crit, max_crit)
                min_crit = min(crit, min_crit)
                
            elif energy_feasible: 
                feasible_energy_candidates.append(target)


        upper_bound = min_crit + RCL_alpha * (max_crit - min_crit)
        if RCL_mode == 'distance':
            feasible_candidates = [i for i in feasible_candidates if env.dist[node, i] <= upper_bound]
        else:
            feasible_candidates = [i for i in feasible_candidates if env.C[i]['DueDate'] <= upper_bound]
        
        if node != 'D' and t + env.dist[node,'D'] / env.v >= env.T:
            return False, False, feasible_energy_candidates
        if len(feasible_candidates) != 0:
            target = choice(feasible_candidates)
            return target, energy_feasible, feasible_energy_candidates
        else:
            return False, energy_feasible, feasible_energy_candidates


    ''' Evalutates feasiblity for a node to enter the RCL
    Evaluates:
        1. Capacity feasibility
        2. Time windows
        3. Charge at least to go to target and go to closest station 
    '''
    def evaluate_candidate(self,env:E_CVRP_TW,target:str,distance:float,t:float,q:float,k:int,energy_feasible:bool,feasible_energy_candidates:list):
        capacity_c = k + env.C[target]['d'] <= env.K
        time_c = t + distance / env.v  <= env.C[target]['DueDate']
        t_time_c = t + distance / env.v + env.C[target]['ServiceTime']  + env.dist[target,'D'] / env.v < env.T
        # Energy enough to go to target and then to closest station
        energy_c = q - distance / env.r - env.dist[target,env.closest[target]] / env.r >= 0

        # The candidate is completely feasible
        if capacity_c and time_c and t_time_c and energy_c:
            return True, True, feasible_energy_candidates
        # The candidate is completely feasible but not reachable by energy
        elif capacity_c and time_c and t_time_c:
            feasible_energy_candidates.append(target)
            return False, True, feasible_energy_candidates
        # The candidate is unfeasible by any condition 
        else:
            return False, energy_feasible, feasible_energy_candidates


    ''' Find closest station to both the current costumer and the depot '''
    def optimal_station(self,env:E_CVRP_TW,node:str,target:str='D'):
        if node != 'D' and target != 'D':
            distances = {s: env.dist[node,s] + env.dist[s,target] for s in env.Stations if s != node}
            min_station = min(distances, key = distances.get)
        else:
            distances = {s: env.dist[node,s] + env.dist[s,target] for s in env.Stations if s != node and s != 'S0'}
            min_station = min(distances, key = distances.get)

        return min_station


    ''' Routes vehicle to depot from current node
    returns:
    -   route in list (excluding current node)
    '''
    def route_to_depot(self,env:E_CVRP_TW,node:str,t:float,d:float,q:float,k:int,route:list,dep_t:list[float],dep_q:list[float]):
        finish_route:list = list()
        extra_t:float = 0
        extra_d:float = 0
        extra_q:float = 0
        
        extra_dep_t = list()
        extra_dep_q = list()

        # The vehicle can go directly to depot
        if env.dist[node,'D'] / env.r < q:

            finish_route += ['D']
            extra_t += env.dist[node,'D'] / env.v
            extra_d += env.dist[node,'D']
            extra_q -= env.dist[node,'D'] / env.r
        
        # Vehicle hasn't enough energy to get to depot
        else:
            s = self.optimal_station(env, node)

            # Optimal station on route to depot exists and is reachable
            if q - env.dist[node,s] / env.r >= 0:
                # Update to station
                extra_t += env.dist[node, s] / env.v
                extra_d += env.dist[node,s]
                extra_q -= env.dist[node,s] / env.r

                # Update in station
                finish_route.append(s)
                recarga = env.dist[s,'D'] / env.r - extra_q - q 
                extra_t += recarga * env.g
                extra_q += recarga

                extra_dep_t.append(t+extra_t)
                extra_dep_q.append(q+extra_q)

                # Update to depot
                finish_route.append('D')
                extra_t += env.dist[s,'D'] / env.v
                extra_d += env.dist[s,'D']
                extra_q -= env.dist[s,'D'] / env.r   
                
            # Optimal station is not reachable
            else:
                s = env.closest[node]
                # Update to station
                extra_t += env.dist[node, s] / env.v
                extra_d += env.dist[node,s]
                extra_q -= env.dist[node,s] / env.r

                # Update in station
                finish_route.append(s)
                recarga = env.dist[s,'D'] / env.r - extra_q - q
                extra_t += recarga * env.g
                extra_q += recarga

                extra_dep_t.append(t+extra_t)
                extra_dep_q.append(q+extra_q)

                # Update to depot
                finish_route.append('D')
                extra_t += env.dist[s,'D'] / env.v
                extra_d += env.dist[s,'D']
                extra_q -= env.dist[s,'D'] / env.r 

        if t + extra_t >= env.T:
            removed = route.pop()
            if env.node_type[removed]=='c':
                self.pending_c.append(removed)
            node = route[-1]

            if removed in env.Costumers:
                d -= env.dist[node,removed]
                k -= env.C[removed]['d']
            else:
                d -= env.dist[node,removed]
            
            del dep_t[-1]
            t = dep_t[-1]
            del dep_q[-1]
            q = dep_q[-1]
            
            t, d, q, k, route, dep_t, dep_q = self.route_to_depot(env, node, t, d, q, k, route, dep_t, dep_q)  
            return t, d, q, k, route, dep_t, dep_q

        else:
            t += extra_t; d += extra_d; q += extra_q
            dep_t.extend(extra_dep_t); dep_t.append(t)
            dep_q.extend(extra_dep_q); dep_q.append(q)
            return t, d , q, k, route + finish_route, dep_t, dep_q


    ''' Routes between costumers to costumers or stations '''
    def direct_routing(self, env:E_CVRP_TW, node:str, target:str, t:float, d:float, q:float, k:int, route:list):
        # Time update
        tv = env.dist[node, target] / env.v

        # Distance update
        d += env.dist[node, target]

        # Charge update
        q -= env.dist[node, target] / env.r

        # Target is costumer
        if env.node_type[target] == 'c':
            tgt = env.C[target]

            # Time update
            start_service = max(t + tv, tgt['ReadyTime'])
            t = start_service + tgt['ServiceTime']

            # Load update
            k += tgt['d']
            
            # Route update
            self.pending_c.remove(target)

        # Target is station
        else:
            # Time update
            t += tv + (env.Q - q) * env.g

            ## Charge update
            q = env.Q
        
        # Route update
        route.append(target)
        
        return t, d, q, k, route


    '''  RCL based constructive
    parametrs:
    -   RCL_alpha

    returns:
    -   t: Total time of route
    -   q: Final charge of vehichle
    -   k: Final capacity of vehicle
    -   route: list with sequence of nodes of route
    '''
    def RCL_based_constructive(self, env:E_CVRP_TW, RCL_alpha:float, RCL_criterion:str):
        t: float = 0
        d: float = 0
        q: float = env.Q
        k: int = 0     
        node = 'D'
        route = [node]   # Initialize route
    
        dep_t = [0] # Auxiliary list with departure times
        dep_q = [env.Q]

        # Adding nodes to route
        while True:
            target, energy_feasible, feasible_energy_candiadates = self.generate_candidate_from_RCL(env,RCL_alpha,RCL_criterion, node, t, q, k)

            # Found a target
            if target != False:
                t, d, q, k, route = self.direct_routing(env, node, target, t, d, q, k, route)
                dep_t.append(t); dep_q.append(q)
                node = target
                
            # No feasible direct target
            else:
                
                # One candidate but not enough energy to travel
                if energy_feasible:
                    
                    # One candidate left but unreachable from depot bc energy 
                    if node == 'D' and len(self.pending_c) == 1:
                        target = self.optimal_station(env, node, self.pending_c[0])
                        t, d, q, k, route = self.direct_routing(env, node, target, t, d, q, k, route)
                        dep_t.append(t); dep_q.append(q)
                        node = target

                    else:

                        if node != 'D' and env.node_type[node] == 'c':
                            # Route to closest station and charge
                            target = env.closest[node]
                            
                        else:
                            # Route to closes station to feasible candidate
                            new_dict = {(node,j):env.dist[node,j] for j in feasible_energy_candiadates}
                            target = self.optimal_station(env, node, min(new_dict, key = new_dict.get)[1])

                            
                        # Check for total time and energy, station is reachable 
                        if t + (env.dist[node,target]/env.v) + ((env.Q - (q - (env.dist[node,target] / env.r))) * env.g) < env.T and \
                            q - env.dist[node,target]/env.v >= 0:
                            t, d, q, k, route = self.direct_routing(env, node, target, t, d, q, k, route)
                            dep_t.append(t); dep_q.append(q)
                            node = target

                        # Total time unfeasible
                        else:
                            # Nothing to do , go to depot
                            t, d, q, k, route, dep_t, dep_q = self.route_to_depot(env, node, t, d, q, k, route, dep_t, dep_q)
                            break

                # Nothing to do , go to depot
                else:
                    t, d, q, k, route, dep_t, dep_q = self.route_to_depot(env, node, t, d, q, k, route, dep_t, dep_q)
                    break
        

        assert t <= env.T, f'The vehicle exceeds the maximum time \n- Max time: {env.T} \n- Route time: {t}'
        assert round(q) >= 0, f'The vehicle ran out of charge'
        assert k <= env.K, f'The vehicles capacity is exceeded \n-Max load: {env.K} \n- Current load: {k}'
            
        return t, d, q, k, route, (dep_t, dep_q)


    # Auxiliary direct routing method for external usage
    def evaluate_direct_routing(env:E_CVRP_TW, node:str, target:str, t:float, d:float, q:float, k:int):
        # Time update
        tv = env.dist[node, target] / env.v
        if env.node_type[target] == 'c' and t + tv > env.C[target]['DueDate']: return False, False, False, False

        # Distance update
        d += env.dist[node, target]

        # Charge update
        q -= env.dist[node, target] / env.r
        if q < 0:   return False, False, False, False

        # Target is costumer
        if env.node_type[target] == 'c':
            tgt = env.C[target]

            # Time update
            start_service = max(t + tv, tgt['ReadyTime'])
            t = start_service + tgt['ServiceTime']

            # Load update
            k += tgt['d']

        # Target is station
        else:
            # Time update
            t += tv + (env.Q - q) * env.g

            ## Charge update
            q = env.Q

        return t, d, q, k
    

    def print_constructive(self, env: E_CVRP_TW, instance: str, t: float, ind: int, Incumbent: float, routes: int):
        gap = round((Incumbent - env.bkFO[instance])/env.bkFO[instance],4)
        print(*[round(t,2), ind, round(Incumbent,2), f'{round(gap*100,2)}%', routes], sep = '\t \t')




''' Genetic algorithm  '''
class Genetic():

    def __init__(self,Population_size:int,Elite_size:int,crossover_rate:float,mutation_rate:float,
                 darwinian_configuration,eval_insert_configuration) -> None:
        self.Population_size: int = Population_size
        self.Elite_size: int = Elite_size
        self.crossover_rate: float = crossover_rate
        self.mutation_rate: float = mutation_rate

        self.darwinian_configuration = darwinian_configuration
        self.eval_insert_configuration = eval_insert_configuration

    ''' Initial population generator '''
    def generate_population(self,env:E_CVRP_TW,constructive:Constructive,start:float=0,
                            instance:str = '',verbose:bool=False) -> tuple[list, list[float], list[float], list[tuple]]:

        # Initalizing data storage
        Population:list = list()
        Distances:list[float] = list()
        Times:list[float] = list()
        Details:list[tuple] = list()

        incumbent:float = 1e9
        best_individual:list = list()

        min_EV_incumbent:float = 1e9
        best_min_EV_individual:list = list()

        if verbose:
            print(f'\nPopulation generation started: {self.Population_size} individuals')
            print(f' - Using a Constructive with adaptative-reactive alpha and Exo-Hybrid criterion')

        # Adaptative-Reactive Constructive
        RCL_alpha_list:list[float] = [0.15, 0.25, 0.35, 0.5]
        alpha_performance = {alpha:0 for alpha in RCL_alpha_list}

        # Calibrating alphas
        for tr_ind in range(int(self.Population_size*0.2)):
            constructive.reset(env)
            tr_distance: float = 0
            RCL_alpha = choice(RCL_alpha_list)
            while len(constructive.pending_c) > 0:
                RCL_criterion = choice(['distance','TimeWindow'])
                t,d,q,k,route,_ = constructive.RCL_based_constructive(env,RCL_alpha,RCL_criterion)
                tr_distance += d
            alpha_performance[RCL_alpha] += 1/tr_distance

        # Generating initial population
        for ind in range(self.Population_size):
            # Storing individual
            individual:list = list()
            distance:float = 0;    distances:list = list()
            t_time:float = 0;      times:list = list()
            loads:list = list()
            dep_t_details:list = list(); dep_q_details:list = list()

            # Intitalizing environemnt
            constructive.reset(env)

            # Choosing alpha
            RCL_alpha = choice(RCL_alpha_list, p = [alpha_performance[alpha]/sum(alpha_performance.values()) for alpha in RCL_alpha_list])

            # Generating individual
            while len(constructive.pending_c) > 0:
                RCL_criterion = choice(['distance', 'TimeWindow'])
                t,d,q,k,route,dep_details = constructive.RCL_based_constructive(env,RCL_alpha,RCL_criterion)
                individual.append(route)
                distance += d;      distances.append(d)
                t_time += t;        times.append(t)
                loads.append(k)
                dep_t_details.append(dep_details[0])
                dep_q_details.append(dep_details[1])
            
            # Updating incumbent
            if distance < incumbent:
                incumbent = distance
                best_individual: list = [individual, distance, t_time, (distances, times, loads, dep_details), process_time() - start]

                # if verbose:
                #     constructive.print_constructive(env, instance, process_time() - start, ind, incumbent, len(individual))
            
            # Updating best found solution with least number of vehicles
            if ind == 0 or \
                len(individual) < len(best_min_EV_individual[0]) or \
                distance < min_EV_incumbent and len(individual) <= len(best_min_EV_individual[0]):

                min_EV_incumbent = distance
                best_min_EV_individual: list = [individual, distance, t_time, (distances, times, loads, dep_details), process_time() - start]
                if verbose:     constructive.print_constructive(env, instance, process_time() - start, ind, min_EV_incumbent, len(individual))
                
            Population.append(individual)
            Distances.append(distance)
            Times.append(t_time)
            Details.append((distances, times, loads, (dep_t_details,dep_q_details)))

        return Population,Distances,Times,Details, (incumbent,best_individual), \
                (min_EV_incumbent,best_min_EV_individual),max(alpha_performance,key=alpha_performance.get)


    ''' Elite class '''
    def elite_class(self, Distances: list):
        return [x for _, x in sorted(zip(Distances,[i for i in range(self.Population_size)]))][:self.Elite_size] 


    ''' Intermediate population '''
    def intermediate_population(self,Distances):
        # Fitness function
        tots = sum(Distances)
        fit_f:list = list()
        for i in range(self.Population_size):
            fit_f.append(tots/Distances[i])
            # probs.append(fit_f[i]/sum(fit_f))
        probs = [func/sum(fit_f) for func in fit_f]

        return choice([i for i in range(self.Population_size)], size = int(self.Population_size - self.Elite_size), replace = True, p = probs)


    ''' Tournament '''
    def tournament(self, inter_population: list, Distances: list) -> list:
        Parents:list = list()
        for i in range(self.Population_size):
            parents:list = list()
            for j in range(2):
                candidate1 = choice(inter_population);  val1 = Distances[candidate1]
                candidate2 = choice(inter_population);  val2 = Distances[candidate2]

                if val1 < val2:     parents.append(candidate1)
                else:               parents.append(candidate2)
            
            Parents.append(parents)
        
        return Parents


    ''' SHAKE: Same individual, same route '''
    def calculate_dist(self, env:E_CVRP_TW, route):
        distance = 0
        for i in range(len(route) - 1):
            distance += env.dist[route[i],route[i+1]]
        return distance


    def two_opt(self, env:E_CVRP_TW, feas_op: Feasibility, individual:list, Details:tuple):
        
        distances, times = Details

        new_individual = deepcopy(individual)
        
        route = new_individual[randint(0,len(individual))]
        #print(route)
        new_individual.remove(route)
        
        
        best_route = deepcopy(route)
        improved = True
        while improved == True:
            improved = False       
            for i in range(1, len(route) - 2):
                for j in range(i+1, len(route)):
                    if j-i == 1:
                        continue
                    new_route = route[:]
                    new_route[i:j] = route[j-1:i-1:-1]
                    new_distance = self.calculate_dist(env, new_route)
                    if new_distance < self.calculate_dist(env, best_route):
                        best_route = new_route
                        improved = True
                        route = best_route
                        


        i = 0
        while i + 2 < len(best_route):
            if best_route[i] == best_route[i+1]:
                del best_route[i]
            else:
                i += 1

        # Calcular la distance de new_distance y new_times
        new_individual.append(best_route)
        #print(new_route)
        feasible, _ = feas_op.individual_check(env, new_individual)
        if feasible:
            return new_individual, *_
        
        else: 
            #print('NON FEASIBLE')
            return individual, sum(distances), sum(times), Details

    
    def route_extension(sefl, constructive:Constructive, individual:list): 
        pass


    ''' CROSSOVER: Same individual, different routes '''
    # evaluated insertion: A given costumer is iteratively placed on an existing route. 
    def evaluated_insertion(self,env:E_CVRP_TW,individual:list,Details:list):
        # Select route that visits less costumers and disolve it
        visited_c:list = list()
        n_visited_c:list = list()
        eff_rates:list = list()

        distances, times, loads, (dep_t_details, dep_q_details) = Details

        for idx, route in enumerate(individual):
            if self.eval_insert_configuration['penalization'] == 'regular': penalization = 1
            elif self.eval_insert_configuration['penalization'] == 'cuadratic': penalization = 2
            elif self.eval_insert_configuration['penalization'] == 'cubic': penalization = 3
            eff_rates.append(distances[idx]/(len(route)**penalization))

            visited_c.append([i for i in route if i[0]=='C'])
            n_visited_c.append(sum([1 for i in route if i[0]=='C']))

        rank_index:list = self.rank_indexes(eff_rates)

        if self.eval_insert_configuration['criterion'] == 'visited costumers':      worst_route_index = n_visited_c.index(min(n_visited_c))
        elif self.eval_insert_configuration['criterion'] == 'phi rate':             worst_route_index = rank_index.index(max(rank_index))
        elif self.eval_insert_configuration['criterion'] == 'Hybrid':               worst_route_index = choice([rank_index.index(max(rank_index)),n_visited_c.index(min(n_visited_c))])
        elif self.eval_insert_configuration['criterion'] == 'random':               worst_route_index = randint(0,len(individual))
        
        pending_c = visited_c[worst_route_index]

        new_individual = deepcopy(individual);  
        new_distances = deepcopy(distances);    
        new_times = deepcopy(times);    
        new_loads = deepcopy(loads);   
        new_dep_t_details = deepcopy(dep_t_details);    
        new_dep_q_details = deepcopy(dep_q_details);    


        # Relocate costumers from first route
        for candidate in pending_c:
            placed = False
            for i in range(len(new_individual)):   
                real_index = rank_index.index(i)

                if real_index != worst_route_index:
                    # Load feasibility check
                    if env.C[candidate]['d'] + new_loads[real_index] > env.K:    continue

                    route = new_individual[real_index]
                    for pos in range(0,len(route[1:])):
                        node = route[pos]

                        # Preliminary feasibility check
                        # if env.C[candidate]['ReadyTime'] <= dep_t_details[rank_index.index(i)][pos]:     continue                                # Waiting to service not allowed
                        if env.C[candidate]['DueDate'] < new_dep_t_details[real_index][pos] + env.dist[node,candidate]/env.v:   continue    # TimeWindow nonfeasible

                        # Feasibility
                        feasible, dep_t, dep_q = self.check_insertion_feasibility(env, new_individual[real_index], pos, candidate, new_dep_t_details[real_index], new_dep_q_details[real_index])
                        
                        if feasible:
                            route.insert(pos+1,candidate)
                            new_individual[real_index] = route

                            d = sum(env.dist[route[ii],route[ii+1]] for ii in range(len(route)-1))
                            
                            new_distances[real_index] = d
                            new_times[real_index] = dep_t[-1]
                            new_loads[real_index] += env.C[candidate]['d']
                            new_dep_t_details[real_index] = dep_t
                            new_dep_q_details[real_index] = dep_q
                            placed = True
                            break

                    if placed:   break
                        
            if not placed: return individual, sum(distances), sum(times), (distances, times, loads, (dep_t_details, dep_q_details))
        
        # print('Successfully inserted a complete route')
        del new_individual[worst_route_index]
        del new_distances[worst_route_index]
        del new_times[worst_route_index]
        del new_loads[worst_route_index]
        del new_dep_t_details[worst_route_index]
        del new_dep_q_details[worst_route_index]
        # print('New individual generated')
        return new_individual, sum(new_distances), sum(new_times), (new_distances, new_times, new_loads, (new_dep_t_details, new_dep_q_details))



    # Check if an instertion is feasible 
    def check_insertion_feasibility(self, env:E_CVRP_TW, route:list, pos:int, candidate:str, dep_t:list[float], dep_q:list[float]):
        new_dep_t = deepcopy(dep_t[:pos+1])
        new_dep_q = deepcopy(dep_q[:pos+1])

        current_t:float = new_dep_t[-1]
        current_q:float = new_dep_q[-1]
        
        # Arch from position to candidate
        current_t, _, current_q, _ = Constructive.evaluate_direct_routing(env, route[pos], candidate, current_t, 0, current_q, 0)
        if current_t == False or current_q < 0 or current_t > env.T:      return False, None, None    # Feasibility check
        new_dep_t.append(current_t)
        new_dep_q.append(current_q)


        # Arch from candidate to pos+1
        # TODO check what happens when route[pos+1] == 'D'
        current_t, _, current_q, _ = Constructive.evaluate_direct_routing(env, candidate, route[pos+1], current_t, 0, current_q, 0) 
        if current_t == False or current_q < 0 or current_t > env.T:      return False, None, None    # Feasibility check
        new_dep_t.append(current_t)
        new_dep_q.append(current_q)

        for i in range(pos+1,len(route)-1):
            current_t, _, current_q, _ = Constructive.evaluate_direct_routing(env, route[i], route[i+1], current_t, 0, current_q, 0)
            if current_t == False or current_q < 0 or current_t > env.T:      return False, None, None    # Feasibility check      
            new_dep_t.append(current_t)
            new_dep_q.append(current_q)      
        
        # TODO check for last segment to depot (updates on new_dep_details)
        return True, new_dep_t, new_dep_q



    ''' MUTATION: Same individual, all routes '''
    # Darwinian phi rate: A proportion of best routes of the individual, according to the phi rate (total distance/total costumers)
    # are advanced to the offspring. The resting routes are disolved and new routes are built with the RCL-based constructive. 
    def Darwinian_phi_rate(self,env:E_CVRP_TW,constructive:Constructive,individual:list,Details:tuple,RCL_alpha:float):
        eff_rates = list()
        distances, times, loads, (dep_t_details, dep_q_details) = Details
        for idx, route in enumerate(individual):
            if self.darwinian_configuration['penalization'] == 'regular': penalization = 1
            elif self.darwinian_configuration['penalization'] == 'cuadratic': penalization = 2
            elif self.darwinian_configuration['penalization'] == 'cubic': penalization = 3
            eff_rates.append(   /(len(route)**penalization))
        
        rank_index = self.rank_indexes(eff_rates)


        new_individual:list = list()
        new_distance = 0;       new_distances:list = list()
        new_time = 0;           new_times:list = list()
        new_loads:list = list()
        new_dep_t_details:list = list(); new_dep_q_details:list = list()
        
        pending_c = deepcopy(env.Costumers)

        i = 0; cont = 0
        while i/len(individual) < self.darwinian_configuration['conservation proportion']
            ii = rank_index.index(i)
            cont += 1
            if cont == len(individual):break
            if self.darwinian_configuration['length restriction'] and not len(individual[ii]) >= 0.5*max([len(route) for route in individual]):
                continue
            new_individual.append(individual[ii])
            new_distance += distances[ii];      new_distances.append(distances[ii])
            new_time += times[ii];              new_times.append(times[ii])
            new_loads.append(loads[ii])
            new_dep_t_details.append(dep_t_details[ii]);    new_dep_q_details.append(dep_q_details[ii])

            for node in new_individual[-1]:
                if node != 'D' and env.node_type[node] == 'c':
                    pending_c.remove(node)

            i += 1

        constructive.pending_c = pending_c

        # Generating individual
        while len(constructive.pending_c) > 0:
            RCL_criterion = choice(['distance', 'TimeWindow'])
            t, d, q, k, route, (dep_t, dep_q) = constructive.RCL_based_constructive(env, RCL_alpha, RCL_criterion)
            new_individual.append(route)
            new_distance += d;      new_distances.append(d)
            new_time += t;          new_times.append(t)
            new_loads.append(k)
            new_dep_t_details.append(dep_t);    new_dep_q_details.append(dep_q)


        return new_individual, new_distance, new_time, (new_distances, new_times, new_loads, (new_dep_t_details,new_dep_q_details))
    

    # Auxiliary method. Returns list that indicates for each route (possition), the rank
    # occupied by the route
    def rank_indexes(self, indicators:list):
        sorted_lst = sorted(indicators)
        
        lista = list()
        for v in indicators:
            indexx = sorted_lst.index(v)
            # Deal with two routes with same distance
            while indexx in lista:
                indexx += 1
            lista.append(indexx)

        return lista



    ''' RECOMBINAITON: different individual, different routes '''

    


    def print_evolution(self, env: E_CVRP_TW, instance: str, t: float, generation: int, Incumbent: float, routes: int):
        gap = round((Incumbent-env.bkFO[instance])/env.bkFO[instance],4)
        if gap < 100:
            print(f'{round(t,2)} \t {generation} \t{round(Incumbent,2)}\t {round(gap*100,1)}% \t {routes}')
        else:
            print(f'{round(t,2)} \t {generation} \t{round(Incumbent,2)}\t {round(gap*100)}% \t {routes}')




'''

'''
class Experiment():

    def __init__(self,path:str,):
        self.path = path
        self.times = {'s':2,'m':3,'l':6}
        self.verbose = False

    def experiment(self,instance:str,configs:dict[str],verbose:bool,save_results:bool):
        '''
        EXPERIMENTATION
        Variable convention:
        - Details: List of tuples (individual), where the tuple (distances, times) are discriminated per route
        - best_individual: list with (individual,distance,time,details)
        '''

        ''' General parameters '''
        evaluate_feasibility:bool=True

        ''' Environment '''
        env:E_CVRP_TW = E_CVRP_TW()

        # Setting runnign times depending on instance size
        if instance in env.sizes['s']:  max_time=60*self.times['s']
        elif instance in env.sizes['m']:  max_time=60*self.times['m']
        else:   max_time=60*self.times['l']

        ''' Constructive heuristic '''
        constructive:Constructive = Constructive()


        ''' Genetic algorithm '''
        Population_size:int = configs['gen-population size']
        Elite_size:int = int(Population_size*configs['gen-elite size'])

        crossover_rate:float = configs['gen-crossover rate']
        mutation_rate:float = configs['gen-mutation rate']

        darwinian_configuration = {'penalization':configs['Dar-penalization'],
                                   'conservation proportion':configs['Dar-conservation proportion'],
                                   'length restriction':configs['Dar-length restriction']}
        eval_insert_configuration = {'penalization':configs['eval-penalization'],
                                    'criterion':configs['eval-criterion']}                                   

        genetic: Genetic = Genetic(Population_size,Elite_size,crossover_rate,mutation_rate,
                                   darwinian_configuration,eval_insert_configuration)
        
        gaps = list()
        times = list()
        for rd_seed in range(30):
            start:float = process_time()
            time_to_find,gap = self.HGA(env,constructive,genetic,instance,max_time,start,evaluate_feasibility,rd_seed,False)

            Experiment._update_lists((gaps,times),(gap,time_to_find))

        if verbose:
            mean_value,median_value,std_deviation,min_value,max_value,confidence_interval = Experiment._compute_stats(gaps)
            print(f'{round(np.mean(times),2)} \t {mean_value} \t {median_value} \t {std_deviation} \t {min_value} \t {max_value} \t {round(confidence_interval[0],2)} \t {round(confidence_interval[1],2)}')
        
        if save_results:
            a_file = open(self.path + f'/Baseline/{instance}',"wb")
            pickle.dump(gaps,a_file)
            a_file.close()
        
        return gaps


    def HGA(self,env:E_CVRP_TW,constructive:Constructive,genetic:Genetic,instance:str,max_time:float, 
            start:float,evaluate_feasibility:bool,rd_seed:float,save_result:bool):
        '''
        ------------------------------------------------------------------------------------------------
        Genetic proccess
        ------------------------------------------------------------------------------------------------
        '''
        seed(rd_seed)

        # Saving performance 
        constructive_Results = dict()
        Results = dict(); Incumbents = list(); ploting_Times = list()
        min_EV_Results = dict(); min_EV_Incumbents = list(); min_EV_ploting_Times = list()
                
        # Constructive
        env.load_data(instance)
        env.generate_parameters()
        constructive.reset(env)
        
        # Printing progress
        if self.verbose: 
            print(f'----- Instance {instance[:-4]} - M{len(list(env.C.keys()))} - {env.bkFO[instance]} - {env.bkEV[instance]} -----')
            print(f' t \t gen \t FO \t gap \t #EV')

        # Population generation
        Population,Distances,Times,Details,(incumbent,best_individual),(min_EV_incumbent,best_min_EV_individual),RCL_alpha = \
                                genetic.generate_population(env,constructive,start,instance,False)
        
        constructive_Results['best individual (min dist)'] = best_individual
        constructive_Results['best individual (min EV)'] = best_min_EV_individual

        Incumbents.append(incumbent)
        ploting_Times.append(best_individual[4])

        min_EV_Incumbents.append(min_EV_incumbent)
        min_EV_ploting_Times.append(best_min_EV_individual[4])

        # Print progress
        if self.verbose: 
            print(f'{round(best_min_EV_individual[4],2)} \tinit \t{round(min_EV_incumbent,1)} \t {round(self.compute_gap(env, instance, min_EV_incumbent)*100,1)}% \t {len(best_min_EV_individual[0])}')     
        
        # Genetic process
        generation = 0
        while process_time()-start<max_time:
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
                # print(generation,i)
                individual_i = Parents[i][randint(0,2)]
                mutated = False

                ### Crossover
                if random() <= genetic.crossover_rate: 
                    new_individual, new_distance, new_time, details = \
                                genetic.evaluated_insertion(env,Population[individual_i],Details[individual_i])
                    mutated = True

                ### Mutation
                if random() <= genetic.mutation_rate:
                    new_individual, new_distance, new_time, details = \
                                genetic.Darwinian_phi_rate(env, constructive,Population[individual_i],Details[individual_i],RCL_alpha)
                    mutated = True

                # No operator is performed
                if not mutated: 
                    new_individual = Population[individual_i]; new_distance = sum(Details[individual_i][0]); new_time = sum(Details[individual_i][1]); details = Details[individual_i]
                

                # Individual feasibility check
                if evaluate_feasibility:
                    feasible, _ = constructive.individual_check(env,new_individual,instance,complete=True)
                    assert feasible, f'!!!!!!!!!!!!!! \tNon feasible individual generated (gen {generation}, ind {i}) / {new_individual}'

                # Store new individual
                New_Population.append(new_individual); New_Distances.append(new_distance); New_Times.append(new_time); New_Details.append(details)

                # Updating incumbent
                if new_distance < incumbent:
                    incumbent = new_distance
                    best_individual: list = [new_individual, new_distance, new_time, details, process_time() - start]

                # Updating best found solution with least number of vehicles
                if len(new_individual) < len(best_min_EV_individual[0]) or \
                    new_distance < min_EV_incumbent and len(new_individual) <= len(best_min_EV_individual[0]):

                    min_EV_incumbent = new_distance
                    best_min_EV_individual: list = [new_individual, new_distance, new_time, details, process_time() - start]

                    if self.verbose:
                        genetic.print_evolution(env,instance,process_time()-start,generation,min_EV_incumbent,len(new_individual))

                ### Store progress
                Incumbents.append(incumbent)
                ploting_Times.append(process_time() - start)

                min_EV_Incumbents.append(min_EV_incumbent)
                min_EV_ploting_Times.append(process_time() - start)

            # Update
            Population = New_Population
            Distances = New_Distances
            Times = New_Times
            Details = New_Details
            generation += 1
            

        # Print progress
        if self.verbose: 
            print('\n')
            print(f'Evolution finished finished at {round(process_time() - start,2)}s')
            print('\tFO \tgap \tEV \ttime to find')
            print(f'dist\t{round(incumbent,1)} \t{round(self.compute_gap(env, instance, incumbent)*100,1)}% \t{len(best_individual[0])} \t{round(best_individual[4],2)}')
            print(f'min_EV \t{round(min_EV_incumbent,1)} \t{round(self.compute_gap(env, instance, min_EV_incumbent)*100,1)}% \t{len(best_min_EV_individual[0])} \t{round(best_min_EV_individual[4],2)}')
            print('\n')

        
        # Storing overall performance for min distance
        Results['best individual'] = best_individual[0]
        Results['best distance'] = best_individual[1]
        Results['gap'] = round(self.compute_gap(env, instance, incumbent)*100,2)
        Results['total time'] = best_individual[2]
        Results['others'] = best_individual[3]
        Results['time to find'] = best_individual[4]
        Results['incumbents'] = Incumbents
        Results['inc times'] = ploting_Times

        # Storing overall performance for min EV
        min_EV_Results['best individual'] = best_min_EV_individual[0]
        min_EV_Results['best distance'] = best_min_EV_individual[1]
        min_EV_Results['gap'] = round(self.compute_gap(env, instance, incumbent)*100,2)
        min_EV_Results['total time'] = best_min_EV_individual[2]
        min_EV_Results['others'] = best_min_EV_individual[3]
        min_EV_Results['time to find'] = best_min_EV_individual[4]
        min_EV_Results['incumbents'] = min_EV_Incumbents
        min_EV_Results['inc times'] = min_EV_ploting_Times
        
        ### Save performance
        if save_result:
            a_file = open(self.saving_path + f'/Exp {self.exp_num}/results-{instance}', "wb")
            pickle.dump([constructive_Results, Results, min_EV_Results], a_file)
            a_file.close()

        
        if not self.verbose:
            # print(f'✅ Instance {instance} - {round(self.compute_gap(env, instance, min_EV_incumbent)*100,2)}%')
            return best_min_EV_individual[4], round(self.compute_gap(env, instance, min_EV_incumbent)*100,2)
    

    def compute_gap(self, env: E_CVRP_TW, instance: str, incumbent: float) -> float:
        return round((incumbent - env.bkFO[instance])/env.bkFO[instance],4)
    
    @staticmethod
    def _update_lists(updating_list:tuple[list],values:tuple):
        for i in range(len(updating_list)):
            updating_list[i].append(values[i])
        return updating_list
    
    @staticmethod
    def _compute_stats(gaps):
        return  round(np.mean(gaps),2),round(np.median(gaps),2),round(np.std(gaps),2),np.min(gaps),np.max(gaps),\
                stats.t.interval(0.95, len(gaps)-1,loc=np.mean(gaps),scale=stats.sem(gaps))
