def chech_all(env, individual):
    costum = []
    for route in individual:
        for i in route:
            if env.node_type[i] == 'c':
                costum.append(i)
    print(len(costum))
    print(len(set(costum)))