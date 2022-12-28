'''
Capacitated Electric Vehicle Routing Problem with Time windows
E-CVRP-TW

Authors:
Juan Betancourt
jm.betancourt@uniandes.edu.co

Daniel Giraldo
ds.giraldoh@uniandes.edu.co
'''

from copy import copy, deepcopy
from time import time
import matplotlib.pyplot as plt
from numpy.random import random, choice, seed, randint
import networkx as nx
import os

'''
CE_VRP_TW Class: Parameters and information
- render method 
'''
class E_CVRP_TW(): 

    def __init__(self, path: str = '/Users/juanbeta/My Drive/Research/Energy/CG-VRP-TW/'):
        self.path = path
        self.colors = ['black', 'red', 'green', 'blue', 'purple', 'orange', 'pink', 'grey', 
                       'yellow', 'tab:red', 'tab:green', 'tab:blue', 'tab:purple', 'tab:orange', 
                       'tab:pink', 'tab:grey', 
                       'black', 'red', 'green', 'blue', 'purple', 'orange', 'pink', 'grey', 
                       'yellow', 'tab:red', 'tab:green', 'tab:blue', 'tab:purple', 'tab:orange', 
                        'tab:pink', 'tab:grey']

        self.instances = os.listdir(self.path + 'Instances/')
        self.instances.remove('readme.txt')
        if '.DS_Store' in self.instances:
            self.instances.remove('.DS_Store')

        self.sizes = {'s': ['c103C5.txt', 'rc208C5.txt', 'c101C5.txt', 'c206C5.txt', 'rc108C5.txt', 'r104C5.txt', 'r203C5.txt', 
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
        'r208_21.txt', 'c207_21.txt']}


    '''
    Load data from txt file
    '''
    def load_data(self, file_name: str):
        file = open(self.path + 'Instances/' + file_name, mode = 'r');     file = file.readlines()
        
        fila = 1
        att = [i for i in str(file[fila]).split(' ') if i != '']
        self.T = float(att[6])
        self.D = {'ID':att[0], 'type': att[1], 'x': float(att[2]), 'y':float(att[3])}      # Depot
        fila += 1
        
        self.S, self.Stations  = {}, []             # Charging stations
        self.C, self.Costumers = {}, []             # Costumers

        while True:
            
            ID, typ, x, y = [i for i in str(file[fila]).split(' ') if i != ''][0:4]
            x = float(x);   y = float(y)

            if typ == 'f':
                self.S[ID] = {'type': 's', 'x': x, 'y': y}
                self.Stations.append(ID)

            else:
                d, ReadyTime, DueDate, ServiceTime = [float(i) for i in [i for i in str(file[fila]).split(' ') if i != ''][4:8]]
                self.C[ID] = { 'type': typ, 'x': x, 'y': y, 'd': d, 'ReadyTime': ReadyTime, 
                               'DueDate': DueDate, 'ServiceTime': ServiceTime}
                self.Costumers.append(ID)

            fila += 1
            if file[fila][0] == "\n":
                break
                
        fila += 1
        self.Q = float([i for i in str(file[fila]).split(' ') if i != ''][5][1:-2]); fila += 1      # Vehicle max charge
        self.K = float([i for i in str(file[fila]).split(' ') if i != ''][4][1:-2]); fila += 1      # Vehicle max load
        self.r = float([i for i in str(file[fila]).split(' ') if i != ''][4][1:-2]); fila += 1      # Fuel consumption rate
        self.g = float([i for i in str(file[fila]).split(' ') if i != ''][4][1:-2]); fila += 1      # Vehichle refueling rate
        self.v = float([i for i in str(file[fila]).split(' ') if i != ''][3][1:-2])                 # Vehicles speed

        
    '''
    Compute several parameters from intial file
    '''
    def generate_parameters(self):
        self.compute_distances()
        self.closest_stations()


    '''
    Compute euclidean distance between two nodes
    '''
    def euclidean_distance(self,x: float, y: float):
        distance = ((x['x'] - y['x'])**2 + (x['y'] - y['y'])**2)**(1/2)
        return distance


    '''
    Distance between:
    - Depot to stations
    - Depot to costumers
    - Costumers to costumers
    - Costumers to stations
    '''
    def compute_distances(self):
        self.dist = {}
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
    
    
    '''
    Generates the closest station to each costumer
    '''
    def closest_stations(self):
        self.closest = {}
        for c in self.C:
            new_dic = {s:self.dist[c,s] for s in self.Stations}
            self.closest[c] = min(new_dic, key = new_dic.get)
    

    '''
    Plots nodes of the VRP
    '''
    def render(self, routes: list):
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
        edges = []
        edge_colors = []
        orders = {}
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
        plt.savefig('results.png', dpi = 600)
        plt.show()
        

    def plot_evolution(self, Incumbents: list, color: str = 'purple'):
        plt.plot(range(len(Incumbents)), Incumbents, color = color)
        plt.title('Evolution of best individual')
        plt.xlabel('Iterations')
        plt.ylabel('Objective (distance)')
        plt.show()


    def detail_route(self, route: list):
        t, q, k = 0, self.Q, 0
        print('########## Departure from the depot ##########\n')
        for i in range(len(route) - 1):
            node = route[i]
            target = route[i+1]
            if target in self.Costumers:
                print('Travel')
                print(f'- travel_t {round(self.dist[node, target] / self.v,2)}')
                t += round(self.dist[node, target]/self.v,2)
                print(f'- t: {t}')
                q -= round(self.dist[node, target] / self.r,2)
                print(f'- q: {q}')
                print('\n')


                print(f'COSTUMER {target}')
                print(f'R \tD \tSt')
                print(f'{round(self.C[target]["ReadyTime"],2)} \t{round(self.C[target]["DueDate"],2)} \t{round(self.C[target]["ServiceTime"],2)}')
                print('------------------------')
                t = round(max(t, self.C[target]['ReadyTime']),2)
                print(f'- start_time: {t}')
                t += self.C[target]['ServiceTime']
                print(f'- t: {t}')
                k += self.C[target]['d']
                print(f'- k: {k}')
            
            elif target in self.Stations:
                print('Travel')
                print(f'- travel_t {round(self.dist[node, target] / self.v,2)}')
                t += round(self.dist[node, target]/self.v,2)
                print(f'- t: {t}')
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



'''
Algorithms Class: Compilation of heuristics to generate a feasible route
- RCL based constructive
'''
class Constructive():

    def __init__(self, RCL_alpha: float, End_slack: int):
        self.RCL_alpha = RCL_alpha
        self.End_slack = End_slack

    
    '''
    Reset the environment to restart experimentation (another parent)
    '''
    def reset(self, env: E_CVRP_TW):

        self.pending_c = deepcopy(env.Costumers)


    '''
    BUILD RLC DEPENDING ON:
    - DISTANCES
    - OPENING TIME WINDOW
    '''
    def generate_candidate_from_RCL(self, env: E_CVRP_TW, node: str, t: float, q: float, k: int):
        feasible_candidates = []
        max_crit = -1e9
        min_crit = 1e9

        RCL_mode = choice(['distance', 'TimeWindow'])

        energy_feasible = False
        for target in self.pending_c:
            distance = env.dist[node,target]

            global_c, energy_feasible = self.evaluate_candidate(env, target, distance, t, q, k, energy_feasible)

            if global_c:
                feasible_candidates.append(target)

                if RCL_mode == 'distance':      crit = distance
                elif RCL_mode == 'TimeWindow':  crit = env.C[target]['DueDate']
                
                max_crit = max(crit, max_crit)
                min_crit = min(crit, min_crit)

        upper_bound = min_crit + self.RCL_alpha * (max_crit - min_crit)
        if RCL_mode == 'distance':
            feasible_candidates = [i for i in feasible_candidates if env.dist[node, i] <= upper_bound]
        else:
            feasible_candidates = [i for i in feasible_candidates if env.C[i]['DueDate'] <= upper_bound]
        
        if node != 'D' and t + env.dist[node,'D'] / env.v + self.End_slack >= env.T:
            return False, False
        if feasible_candidates != []:
            target = choice(feasible_candidates)
            return target, energy_feasible
        else:
            return False, energy_feasible


    '''
    Evalutates feasiblity for a node to enter the RCL
    Evaluates:
        1. Capacity feasibility
        2. Time windows
        3. Charge at least to go to target and go to closest station 
    '''
    def evaluate_candidate(self, env: E_CVRP_TW, target: str, distance, t: float, q: float, k: int, energy_feasible: bool):
        capacity_c = k + env.C[target]['d'] <= env.K
        time_c = t + distance / env.v  <= env.C[target]['DueDate']
        energy_c = q - distance / env.r - env.dist[target,env.closest[target]] / env.r >= 0

        if capacity_c and time_c and energy_c:
            return True, True
        elif capacity_c and time_c:
            return False, False
        elif energy_c:
            return False, True
        else:
            return False, energy_feasible
    

    '''
    Find closest station to both the current costumer and the depot
    '''
    def optimal_station(self, env: E_CVRP_TW, node: str, target: str = 'D'):
        if node != 'D':
            distances = {s: env.dist[node,s] + env.dist[s,target] for s in env.Stations if s != node}
            min_station = min(distances, key = distances.get)
        else:
            distances = {s: env.dist[node,s] + env.dist[s,target] for s in env.Stations if s != node and s != 'S0'}
            min_station = min(distances, key = distances.get)

        return min_station


    '''
    Routes vehicle to depot from current node
    returns:
    -   route in list (excluding current node)
    '''
    def route_to_depot(self, env: E_CVRP_TW, node: str, t: float, d: float, q: float):
        route = []

        # The vehicle can go directly to depot
        if env.dist[node,'D'] / env.v < q:
            route += ['D']
            t += env.dist[node,'D']
            d += env.dist[node,'D']
            q -= env.dist[node,'D'] / env.v
        
        # Vehicle hasn't enough energy to get to depot
        else:
            s = self.optimal_station(env, node)

            # Optimal station on route to depot exists and is reachable
            if q - env.dist[node,s] / env.r >= 0:
                # Update to station
                t += env.dist[node, s] / env.v
                d += env.dist[node,s]
                q -= env.dist[node,s] / env.r

                # Update in station
                route.append(s)
                recarga = env.dist[s,'D'] / env.r - q 
                t += recarga * env.g
                q += recarga

                # Update to depop
                route.append('D')
                t += env.dist[s,'D'] / env.v
                d += env.dist[s,'D']
                q -= env.dist[s,'D'] / env.r

                if env.dist[s,'D'] / env.r > env.Q:     
                    print('Daniel es un marica \nDepot is not reachable from station')
                
            # Optimal station is not reachable
            else:
                s = env.closest[node]
                # Update to station
                t += env.dist[node, s] / env.v
                d += env.dist[node,s]
                q -= env.dist[node,s] / env.r

                # Update in station
                route.append(s)
                recarga = env.dist[s,'D'] / env.r - q 
                t += recarga * env.g
                q += recarga

                # Update to depto
                route.append('D')
                t += env.dist[s,'D'] / env.v
                d += env.dist[s,'D']
                q -= env.dist[s,'D'] / env.r

                if env.dist[s,'D'] / env.r > env.Q:     
                    print('Daniel es un marica \nDepot is not reachable from station')
        
        return t, d, q, route


    '''
    Routes between to ca costumer
    '''
    def simple_routing(self, env: E_CVRP_TW, node: str, target: str, route: list, t: float, d: float, q: float, k: int):
        tgt = env.C[target]

        # Time update
        tv = env.dist[node, target] / env.v
        start_service = max(t + tv, tgt['ReadyTime'])
        t = start_service + tgt['ServiceTime']

        # Distance update
        d += env.dist[node, target]
        
        # Load update
        k += tgt['d']

        # Charge update
        q -= env.dist[node, target] / env.r
        
        # Route update
        route.append(target)
        self.pending_c.remove(target)

        return t, d, q, k, route


    '''
    RCL based constructive
    parametrs:
    -   RCL_alpha
    -   RCL_mode: {'distance', 'TimeWindow}
    -   End_slack: Energy remaining to route to depot

    returns:
    -   t: Total time of route
    -   q: Final charge of vehichle
    -   k: Final capacity of vehicle
    -   route: list with sequence of nodes of route
    '''
    def RCL_based_constructive(self, env: E_CVRP_TW):
        t, d, q, k = 0, 0, env.Q, 0     # Initialize parameters
        node = 'D'; route = [node]   # Initialize route

        # Adding nodes to route
        while True:

            target, energy_feasible = self.generate_candidate_from_RCL(env, node, t, q, k)

            # Found a target
            if target != False:
                t, d, q, k, route = self.simple_routing(env, node, target, route, t, d, q, k)
                node = target
                
            # No feasible target
            else:
                
                # One candidate but not enough energy to travel
                if energy_feasible:
                    # Route to closest station and charge
                    ## Time update
                    t += env.dist[node,env.closest[node]] / env.v 
                    t += (env.Q - q) * env.g

                    # Distance update
                    d += env.dist[node,env.closest[node]] 

                    ## Charge update
                    q = env.Q 

                    ## Update route 
                    node = env.closest[node]
                    route.append(node)

                    # Chose candidate from RCL
                    target, energy_feasible = self.generate_candidate_from_RCL(env, node, t, q, k)
                    if target != False:
                        t, d, q, k, route = self.simple_routing(env, node, target, route, t, d, q, k)
                        node = target
                
                    # No feasible target
                    else:
                        # Nothing to do , go to depot
                        t, d, q, finish_route = self.route_to_depot(env, node, t, d, q)
                        route += finish_route
                        break

                else:
                    # Pending node but not enough energy to get there and then to station
                    if len(self.pending_c) == 1 and t + env.C[self.pending_c[0]]['DueDate'] / env.v <= env.C[self.pending_c[0]]['DueDate']:
                        s = self.optimal_station(env, node, self.pending_c[0])

                        # Optimal station on route to depot exists and is reachable
                        if q - env.dist[node,s] / env.r >= 0:
                            pass
                        elif q - env.dist[node,env.closest[self.pending_c[0]]] / env.r >= 0:
                            s = self.closest[env.pending_c[0]]
                        else:
                            break

                        # Update to station
                        t += env.dist[node, s] / env.v
                        d += env.dist[node,s]
                        q -= env.dist[node,s] / env.r

                        # Update in station
                        route.append(s)
                        recarga = env.Q - q
                        t += recarga * env.g
                        q = env.Q

                        # Update to depop
                        node = s

                    # Nothing to do , go to depot
                    else:
                        t, d, q, finish_route = self.route_to_depot(env, node, t, d, q)
                        route += finish_route
                        break
            
        if round(q) < 0:
            print(q)
            print("ERROR FATAL")

        return t, d, q, k, route




'''
Factibility checks
'''
class Feasibility():

    def __init__(self):
        pass
    

    '''
    Check feasibility for al whole population
    '''
    def population_check(self, env: E_CVRP_TW, Population: list):
        # Intial population feasibility check
        datas = []
        for individual in Population:
            feasible = self.individual_check(env, individual)
            datas.append(int(feasible))
        return datas


    '''
    Checks the feasibility of an individual (all the routes)
    '''
    def individual_check(self, env: E_CVRP_TW, individual: list):
        feasible = True
        for route in range(len(individual)):
            route = individual[route]
            t = 0
            q = env.Q
            k = 0
            for i in range(len(route)-1):
                node = route[i]
                target = route[i + 1]
                feasible, t, q, k, self.transition_check(env, node, target, t, q, k)
                if not feasible:
                    break

            if not feasible:
                break

        return feasible

                
    def transition_check(self, env: E_CVRP_TW, node: str, target: str, t: float, q: float, k: int):
        time_feasible, t = self.time_check(env, node, target, t, q)
        energy_feasible, q = self.energy_check(env, node, target, q)
        load_feasible, k = self.load_check(env, target, k)

        if time_feasible and energy_feasible and load_feasible:
            return True, t, q ,k
        else:
            return False, t, q, k


    def time_check(self, env: E_CVRP_TW, node: str, target: str, t: float, q: float):
        feasible = True
        travel_time = env.dist[node,target] / env.v
        # Total time check
        if target in env.Costumers:
            update = travel_time + env.C[target]['ServiceTime']
            if t + update > env.T:      feasible  = False
            else:                       t += update
        elif target in env.Stations:
            update = travel_time + (env.Q - q) * env.g
            if t + update > env.T:      feasible  = False
            else:                       t += update
        elif target == 'D':
            update = travel_time 
            if t + update > env.T:      feasible  = False
            else:                       t += update

        return feasible, t


    def energy_check(self, env: E_CVRP_TW, node: str, target: str, q: float):
        if q - env.dist[node, target] / env.r >= 0:
            q -= env.dist[node, target] / env.r
            if target in env.Stations:
                q = env.Q
            return True, q
        else:  
            return False, q


    def load_check(self, env: E_CVRP_TW, target: str, k: int):
        if target in env.Costumers:
            if k + env.C[target]['d'] <= env.K:
                k += env.C[target]['d']
                return True, k
            else:
                return False, k
        else:
            return True, k




'''
Reparation class
'''
class Reparator(Constructive):
    
    
    '''
    Repair protocol
    '''
    def repair_individual(self, env: E_CVRP_TW, individual: list):
        if individual[0] != 'D' or individual[-1] != 'D':
            pass
        visited = []

        for route in individual:

            for node in route[1:-1]:
                if node not in env.Stations and node in visited:
                    route.remove(node)
                elif node not in env.Stations:
                    pass


    def repair_termination(self, env: E_CVRP_TW, node: str, target: str, t: float, q: float, k: int):
        # Not enough time
        travel_time = env.dist[node,target] / env.v
        extra_tiem = 0


    def repair_chorizo(self, env: E_CVRP_TW, chorizo: list):
        if type(chorizo[0]) == list:
            chorizo = self.build_chorizo(env, chorizo)
        parent = []
        distance = 0
        distances = []
        ttime = 0
        times = [] 
        pending_c = []
        self.reset(env)
        
        ### Construct repaired routes
        i = 0
        while i < len(chorizo) - 1:
            # Append removed nodes
            chorizo += pending_c
            pending_c = []    

            # Initialize first route
            route = ['D']
            t, d, q, k = 0, 0, env.Q, 0

            node = chorizo[i]
            if node in env.Costumers:
                t, d, q, k, route = self.simple_routing(env, 'D', node, route, t, d, q, k)
            else:
                t += env.dist['D',node]/env.v
                q -= env.dist['D',node]/env.r
                t += (env.Q - q)*env.g
                q = env.Q

            route_done = False

            while not route_done:

                target = chorizo[i+1]
   
                # Load unfeasible:
                if k + env.C[target]['d'] >= env.K:           
                    t, d, q, finish_route = self.route_to_depot(env, node, t, d, q)
                    route += finish_route
                    i += 1
                    route_done = True
                
                # Total time unfeasible
                elif t + env.dist[node,'D'] / env.v > env.T:
                    i -= 1
                    node = chorizo[i]
                    t, d, q, finish_route = self.route_to_depot(env, node, t, d, q)
                    route += finish_route[:1]
                    i += 2
                    route_done = True
                

                # Time window unfeasible
                elif t + env.dist[node, target] / env.v > env.C[target]['DueDate']:
                    missed = chorizo.pop(i+1)
                    pending_c.append(missed)
                    if i + 1 >= len(chorizo):
                        t, d, q, finish_route = self.route_to_depot(env, node, t, d, q)
                        route += finish_route[:1]
                        i += 1
                        route_done = True


                # Charge unfeasible
                elif q - env.dist[node,target] / env.r - env.dist[target,env.closest[target]] / env.r < 0 and node not in env.Stations:
                    s = self.optimal_station(env, node, target)
                    if env.dist[node,s] / env.r > q:
                        s = env.closest[node]
                    
                    ## Time update
                    t += env.dist[node,s] / env.v 
                    t += (env.Q - q) * env.g

                    # Distance update
                    d += env.dist[node,s] 

                    ## Charge update
                    q = env.Q 

                    ## Update route 
                    node = s
                    route.append(node)


                else:
                    t, d, q, k, route = self.simple_routing(env, node, target, route, t, d, q, k)
                    node = target
                    i += 1
                    if i + 1 >= len(chorizo):
                        t, d, q, finish_route = self.route_to_depot(env, node, t, d, q)
                        route += finish_route[:1]
                        i += 1
                        route_done = True

            if route[-2] == 'S0':
                del route[-2]
            
            parent.append(route)
            distance += d
            distances.append(d)
            ttime += t
            times.append(t)

        self.pending_c = pending_c
        while len(self.pending_c) > 0:
            t, d, q, k, route = self.RCL_based_constructive(env)
            parent.append(route)
            distance += d
            distances.append(d)
            ttime += t
            times.append(t)
        
        return parent, distance, distances, ttime, times


    def build_chorizo(self, env: E_CVRP_TW, individual: list):
        chorizo = []
        for route in individual:
            for j in route[1:-1]:
                if j in env.Costumers:
                    chorizo.append(j)
        return chorizo




'''
Genetic algorithm: 
'''
class Genetic():

    def __init__(self, Population_size: int, Elite_size: int, crossover_rate: float, mutation_rate: float) -> None:
        self.Population_size: int = Population_size
        self.Elite_size: int = Elite_size
        self.crossover_rate: float = crossover_rate
        self.mutation_rate: float = mutation_rate

    '''
    Initial population generator
    '''
    def generate_population(self, env: E_CVRP_TW, constructive: Constructive, verbose: bool = False) -> tuple[list, list[float], list[float], list[tuple]]:

        # Initalizing data storage
        Population: list = []
        Distances: list[float] = []
        Times: list[float] = []
        Details: list[tuple] = []

        incumbent: float = 1e9
        best_individual: list = []

        # Generating initial population
        for individual in range(self.Population_size):
            if verbose and individual%20 == 0:     
                print(f'Generation progress: {round(individual/self.Population_size)}')

            individual: list = []
            distance: float = 0
            distances: list = []
            t_time: float = 0
            times: list = []


            # Intitalizing environemnt
            constructive.reset(env)
            while len(constructive.pending_c) > 0:

                t, d, q, k, route = constructive.RCL_based_constructive(env)
                individual.append(route)
                distance += d
                distances.append(d)
                t_time += t
                times.append(t)
                
            
            Population.append(individual)
            Distances.append(distance)
            Times.append(t_time)
            Details.append((distances, times))
            
            if distance < incumbent:
                incumbent = distance
                best_individual: list = [individual, distance, t_time, (distances, times)]
        
        return Population, Distances, Times, Details, incumbent, best_individual


    '''
    Tournament
    '''
    def tournament(self, inter_population: list, Distances: list):
        Parents = []
        for i in range(self.Population_size):
            parents = []
            for j in range(2):
                candidate1 = choice(inter_population);  val1 = Distances[candidate1]
                candidate2 = choice(inter_population);  val2 = Distances[candidate2]

                if val1 < val2:     parents.append(candidate1)
                else:               parents.append(candidate2)
            
            Parents.append(parents)
        
        return Parents


    '''
    RECOMBINATION/COSS-OVER
    '''
    def crossover(self, env: E_CVRP_TW, parent: list, chorizo: list, mode: str, repair_op: Reparator):
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


    # def compound_crossover(self, env: CG_VRP_TW, parent: list):
    #     new_individual = []
        
    #     if len(parent) % 2 == 0:
    #         finish = len(parent) - 1
    #     else:
    #         finish = len(parent) - 2
    #         new_individual.append(parent[-1])
    #     for i in range(0, finish, 2):
    #         chromosome_1 = parent[i]
    #         chromosome_2 = parent[i+1]

    #         new_chromosome_1, new_chromosome_2 = self.cross_route_by_positions(env, chromosome_1, chromosome_2)

    #         new_individual.extend([new_chromosome_1,new_chromosome_2])
        
    #     return new_individual
            

    # def cross_route_by_positions(self, env, chromosome_1: list, chromosome_2: list):
        # found = False
        # while not found:
        #     pos = randint(1, min(len(chromosome_1) - 1, len(chromosome_2) - 1))
        #     if chromosome_1[pos][0] == 'S' or chromosome_2[pos][0] == 'S':
        #         pass
        #     else:
        #         break
        # new_chromosome_1 = deepcopy(chromosome_1)
        # new_chromosome_1[pos] = chromosome_2[pos]
        # new_chromosome_2 = deepcopy(chromosome_2)
        # new_chromosome_2[pos] = chromosome_1[pos]

        # return new_chromosome_1, new_chromosome_2


    def print_initial_population(self, env: E_CVRP_TW, start: float, Population: list[list], Distances: list[float], feas_op: Reparator):
        print('\n###################   Initial Population   ####################\n')
        print(f'Total generation time: {time() - start} s')
        print(f'Number of individuals: {self.Population_size}')
        print(f'Best generated individual:  {round(min(Distances), 2)}')
        print(f'Worst generated individual: {round(max(Distances), 2)}')
        print(f'Number of unfeasible individuals: {self.Population_size - sum(feas_op.population_check(env, Population))}')
        print('\n')


    def print_evolution(self, env: E_CVRP_TW, start: float, Population: list[list], generation: int, Distances: list, feas_op: Reparator, incumbent: float):
        print(f'\n###################   Generation {generation}   ####################\n')
        print(f'Total evolution time: {time() - start} s')
        print(f'Number of individuals: {len(Population)}')
        print(f'Best generated individual (dist): {incumbent}')
        print(f'Number of unfeasible individuals: {self.Population_size - sum(feas_op.population_check(env, Population))}')
        print('\n')




class Experiment():

    def __init__(self):
        pass
    

    def evolution(self, env: E_CVRP_TW, genetic: Genetic, repair_op: Reparator, Population: list[list], Distances: list[float], Incumbents: list[float], T_Times: list[float], Results: list, best_individual: list, start: float, max_time: int):
        '''
        ------------------------------------------------------------------------------------------------
        Genetic proccess
        ------------------------------------------------------------------------------------------------
        '''
        generation: int = 0
        incumbent = Incumbents[0]

        while time() - start <= max_time:
            generation += 1

            ### Selecting elite class
            Elite = [x for _, x in sorted(zip(Distances,[i for i in range(genetic.Population_size)]))][:genetic.Elite_size] 


            ### Selection: From a population, which parents are able to reproduce
            # Fitness function
            tots = sum(Distances)
            fit_f = [tots/Distances[i] for i in range(len(Distances))]
            probs = [fit_f[i]/sum(fit_f) for i in range(len(Distances))]
            

            # Intermediate population: Sample of the initial population    
            inter_population = choice([i for i in range(genetic.Population_size)], size = int(genetic.Population_size - genetic.Elite_size), replace = True, p = probs)
            inter_population = Elite + list(inter_population)


            ### Tournament: Select two individuals and leave the best to reproduce
            Parents = genetic.tournament(inter_population, Distances)
            

            ### Recombination: Combine 2 parents to produce 1 offsprings 
            New_chorizos = []
            for index in range(len(Parents)):
                couple = Parents[index]
                chosen_parent = choice([couple[0], couple[1]])
                chorizo = repair_op.build_chorizo(env, Population[chosen_parent])
                
                if random() < genetic.crossover_rate:
                    cross_mode = choice(['2opt', 'simple_insertion', 'smart_crossover'])
                    new_chorizo = genetic.crossover(env, Population[chosen_parent], chorizo, cross_mode, repair_op)
                else:
                    new_chorizo = chorizo
            
                New_chorizos.append(new_chorizo)


            ### Mutation: 'Shake' an individual
            

            ### Repair solutions
            Population, Distances, Times = [],[],[]
            for i in range(genetic.Population_size):
                individual, distance, distances, t_time, times  = repair_op.repair_chorizo(env, New_chorizos[i])

                Population.append(individual);  Distances.append(distance); 
                Times.append(t_time)

                if distance < incumbent:
                    incumbent = distance
                    best_individual = [individual, distance, t_time, (distances, times)]
                
            # if generation % 50 == 0:
            #     genetic.print_evolution(env, start, Population, generation, Distances, feas_op, incumbent)

            Incumbents.append(round(incumbent,3))
            T_Times.append(round(time() - start,2))
            
        Results.append((Incumbents,T_Times))
    
        return Incumbents, T_Times, Results, incumbent, best_individual



    def save_performance(self, Results, instance, path, xlim = None, cols = None):
        colors = ['blue' ,'black', 'red', 'purple', 'green', 'orange', 'pink', 'brown']
        if len(Results) == 1: colors = choice(colors)
        for algorithm in range(5):
            print(algorithm)
            plt.plot(Results[algorithm][1], Results[algorithm][0], color = colors[algorithm])
        plt.title(f'Performance of Genetic A. on: {instance[:-4]}')
        plt.xlabel('Time (s)')
        plt.ylabel('Objective (d)')
        if xlim != None:
            plt.xlim(0, xlim)
        plt.savefig(f'{path}', dpi = 600)
        #plt.show()
