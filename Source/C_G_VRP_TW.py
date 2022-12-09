'''
Metaheurísticas 2022-2
Universidad de los Andes
Workshop 3-4

Juan Betancourt
201632544

Daniel Giraldo
201920055
'''
from copy import copy, deepcopy
from time import time
import matplotlib.pyplot as plt
from numpy.random import random, choice, seed, randint
import networkx as nx
import os

'''
CE_VRP_TW Class: Parameters and inforamtion
- render method 
'''
class CG_VRP_TW(): 

    def __init__(self):
        self.path = '/Users/juanbeta/Library/CloudStorage/OneDrive-UniversidaddelosAndes/WS 2&3/'
        self.colors = ['black', 'red', 'green', 'blue', 'purple', 'orange', 'pink', 'grey', 
                       'yellow', 'tab:red', 'tab:green', 'tab:blue', 'tab:purple', 'tab:orange', 
                       'tab:pink', 'tab:grey', 
                       'black', 'red', 'green', 'blue', 'purple', 'orange', 'pink', 'grey', 
                       'yellow', 'tab:red', 'tab:green', 'tab:blue', 'tab:purple', 'tab:orange', 
                        'tab:pink', 'tab:grey', ]

        self.instances = os.listdir(self.path+'Instances/')
        self.instances.remove('readme.txt')


    '''
    Load data from txt file
    '''
    def load_data(self, file_name: str):
        file = open(self.path + '/Instances/' + file_name, mode = 'r');     file = file.readlines()
        
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
                d, ReadyTime, DueDate, ServiceTime = [float(i) for i in [i for i in str(file[fila]).split(' ') if i != ''][4:8] ]
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
        plt.show()
        plt.savefig('results', dpi = 600)


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

    def __init__(self):
        pass
    
    '''
    Reset the environment to restart experimentation (another parent)
    '''
    def reset(self, env):

        self.pending_c = deepcopy(env.Costumers)


    '''
    BUILD RLC DEPENDING ON:
    - DISTANCES
    - OPENING TIME WINDOW
    '''
    def generate_candidate_from_RCL(self, env, node, t, q, k, RCL_alpha, RCL_mode, End_slack):
        feasible_candidates = []
        max_crit = -1e9
        min_crit = 1e9

        if RCL_mode == 'Hybrid':
            RCL_mode = choice(['distance', 'TimeWindow'])

        energy_feasible = False
        for target in self.pending_c:
            distance = env.dist[node,target]

            global_c, energy_feasible = self.evaluate_cadidate(env, target, distance, t, k, q, energy_feasible)

            if global_c:
                feasible_candidates.append(target)

                if RCL_mode == 'distance':      crit = distance
                elif RCL_mode == 'TimeWindow':  crit = env.C[target]['DueDate']
                
                max_crit = max(crit, max_crit)
                min_crit = min(crit, min_crit)

        upper_bound = min_crit + RCL_alpha * (max_crit - min_crit)
        if RCL_mode == 'distance':
            feasible_candidates = [i for i in feasible_candidates if env.dist[node, i] <= upper_bound]
        else:
            feasible_candidates = [i for i in feasible_candidates if env.C[i]['DueDate'] <= upper_bound]
        
        if node != 'D' and t + env.dist[node,'D'] / env.v + End_slack >= env.T:
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
    def evaluate_cadidate(self, env, target, distance, t, k, q, energy_feasible):
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
    def optimal_station(self, env, node, target = 'D'):
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
    def route_to_depot(self, env, node, t, d, q):
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
    def simple_routing(self, env, node, target, route, t, d, q, k):
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

        return route, t, d, q, k


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
    def RCL_based_constructive(self, env, RCL_alpha: float, RCL_mode: str, End_slack: int):
        t, d, q, k = 0, 0, env.Q, 0     # Initialize parameters
        node = 'D'; route = [node]   # Initialize route

        # Adding nodes to route
        while True:

            target, energy_feasible = self.generate_candidate_from_RCL(env, node, t, q, k, RCL_alpha, RCL_mode, End_slack)

            # Found a target
            if target != False:
                route, t, d, q, k = self.simple_routing(env, node, target, route, t, d, q, k)
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
                    target, energy_feasible = self.generate_candidate_from_RCL(env, node, t, q, k, RCL_alpha, RCL_mode, End_slack)
                    if target != False:
                        route, t, d, q, k = self.simple_routing(env, node, target, route, t, d, q, k)
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
    def population_check(self, env, Population):
        # Intial population feasibility check
        datas = []
        for individual in Population:
            feasible = self.individual_check(env, individual)
            datas.append(int(feasible))
        return datas


    '''
    Checks the feasibility of an individual (all the routes)
    '''
    def individual_check(self, env: CG_VRP_TW, individual):
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

                
    def transition_check(self, env: CG_VRP_TW, node, target, t, q, k):
        time_feasible, t = self.time_check(env, node, target, t, q)
        energy_feasible, q = self.energy_check(env, node, target, q)
        load_feasible, k = self.load_check(env, target, k)

        if time_feasible and energy_feasible and load_feasible:
            return True, t, q ,k
        else:
            return False, t, q, k


    def time_check(self, env: CG_VRP_TW, node, target, t, q):
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


    def energy_check(self, env: CG_VRP_TW, node, target, q):
        if q - env.dist[node, target] / env.r >= 0:
            q -= env.dist[node, target] / env.r
            if target in env.Stations:
                q = env.Q
            return True, q
        else:  
            return False, q


    def load_check(self, env: CG_VRP_TW, target, k):
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

    def __init__(self):
        pass
    
    
    '''
    Repair protocol
    '''
    def repair_individual(self, env: CG_VRP_TW, individual):
        if individual[0] != 'D' or individual[-1] != 'D':
            pass
        visited = []

        for route in individual:

            for node in route[1:-1]:
                if node not in env.Stations and node in visited:
                    route.remove(node)
                elif node not in env.Stations:
                    pass


    def repair_termination(self, env: CG_VRP_TW, node, target, t, q, k):
        # Not enough time
        travel_time = env.dist[node,target] / env.v
        extra_tiem = 0


    def repair_chorizo(self, env: CG_VRP_TW, chorizo: list, RCL_alpha: float, RCL_mode: str, End_slack: int):
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
                route, t, d, q, k = self.simple_routing(env, 'D', node, route, t, d, q, k)
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
                    route, t, d, q, k = self.simple_routing(env, node, target, route, t, d, q, k)
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
            t, d, q, k, route = self.RCL_based_constructive(env, RCL_alpha, RCL_mode, End_slack)
            parent.append(route)
            distance += d
            distances.append(d)
            ttime += t
            times.append(t)
        
        return parent, distance, distances, ttime, times


    def build_chorizo(self, env, individual):
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

    def __init__(self, Population_size, Elite_size):
        self.Population_size = Population_size
        self.Elite_size = Elite_size

    '''
    Initial population generator
    '''
    def generate_population(self, constructive, const_parameters: tuple, verbose = False):
        env, RCL_alpha, RCL_mode, End_slack = const_parameters

        # Initalizing data storage
        Population = []
        Distances = []
        Times = []
        Details = []

        # Generating initial population
        for individual in range(self.Population_size):
            if verbose and individual%20 == 0:     
                print(f'Generation progress: {round(individual/self.Population_size)}')

            parent = []
            distance = 0
            distances = []
            ttime = 0
            times = []


            # Intitalizing environemnt
            constructive.reset(env)
            while len(constructive.pending_c) > 0:

                RCL_mode = 'Hybrid'

                t, d, q, k, route = constructive.RCL_based_constructive(env, RCL_alpha, RCL_mode, End_slack)
                parent.append(route)
                distance += d
                distances.append(d)
                ttime += t
                times.append(t)
                
            
            Population.append(parent)
            Distances.append(distance)
            Times.append(ttime)
            Details.append((distances, times))
        
        return Population, Distances, Times, Details


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
    def crossover(self, env:CG_VRP_TW, parent: list, chorizo: list, mode: str, repair_op: Reparator):
        if mode == 'simple_crossover':      return self.simple_crossover(chorizo)
        elif mode == '2opt':                return self.opt2(chorizo)
        elif mode == 'Hybrid':
            num = choice([0,1])
            if num == 0:    return self.simple_crossover(chorizo)
            elif num == 1:  return self.opt2(chorizo)
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


    def compound_crossover(self, env: CG_VRP_TW, parent: list):
        new_individual = []
        
        if len(parent) % 2 == 0:
            finish = len(parent) - 1
        else:
            finish = len(parent) - 2
            new_individual.append(parent[-1])
        for i in range(0, finish, 2):
            chromosome_1 = parent[i]
            chromosome_2 = parent[i+1]

            new_chromosome_1, new_chromosome_2 = self.cross_route_by_positions(env, chromosome_1, chromosome_2)

            new_individual.extend([new_chromosome_1,new_chromosome_2])
        
        return new_individual
            

    def cross_route_by_positions(self, env, chromosome_1: list, chromosome_2: list):
        found = False
        while not found:
            pos = randint(1, min(len(chromosome_1) - 1, len(chromosome_2) - 1))
            if chromosome_1[pos][0] == 'S' or chromosome_2[pos][0] == 'S':
                pass
            else:
                break
        new_chromosome_1 = deepcopy(chromosome_1)
        new_chromosome_1[pos] = chromosome_2[pos]
        new_chromosome_2 = deepcopy(chromosome_2)
        new_chromosome_2[pos] = chromosome_1[pos]

        return new_chromosome_1, new_chromosome_2


    def print_initial_population(self, env: CG_VRP_TW, start: float, Population: list, Distances: list, feas_op: Reparator):
        print('\n###################   Initial Population   ####################\n')
        print(f'Total generation time: {time() - start} s')
        print(f'Number of individuals: {self.Population_size}')
        print(f'Best generated individual:  {round(min(Distances), 2)}')
        print(f'Worst generated individual: {round(max(Distances), 2)}')
        print(f'Number of unfeasible individuals: {self.Population_size - sum(feas_op.population_check(env, Population))}')
        print('\n')


    def print_evolution(self, env: CG_VRP_TW, start: float, Population: list, generation: int, Distances: list, feas_op: Reparator, incumbent: float):
        print(f'\n###################   Generation {generation}   ####################\n')
        print(f'Total evolution time: {time() - start} s')
        print(f'Number of individuals: {len(Population)}')
        print(f'Best generated individual (dist): {incumbent}')
        print(f'Number of unfeasible individuals: {self.Population_size - sum(feas_op.population_check(env, Population))}')
        print('\n')
