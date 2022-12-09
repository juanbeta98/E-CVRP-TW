# CG-VRP-TW
Capacitated Green Vehicle-Routing-Problem with Time Windows

Ths repository contains a metaheuristic approach to the Capacitated Green Vehicle-Routing-Problem with Time Windows. In this problem, a set of clients must be serviced with a homogeneous fleet of vehicles, a fixed load capacity and a fixed energy/fuel capacity. There is a set of stations a vehicle can visit to reload the energy/fuel. All clients have a particular time window in which they can be serviced.

We propose a genetic algorithm with several crossover and mutation operators for the routes. The algorithm combines elitism, random sampling with a fitness function, and tournament. An randomized RCL-based constructive heuristic is used to build the original popoulation using different criterion to define the candidates. Another heuristic with similar caracteristics is used to repair and reconstruct routes. 
