import random
import time
import math
import networkx as nx
import matplotlib.pyplot as plt

def myprint(matrix):
    for row in matrix:
        print(row)

# Generate syndromes after distributing errors
# N = number of cells in plane
# p = probability of error
def distribute_errors(N, p):
    # errors[r][c] = [up, down, left, right]
    # entries are 0 if no error, 1 if error on that qubit
    errors = [[[0, 0, 0, 0] for x in range(N+1)] for y in range(N+1)]
    
    # Distribute errors; leave 1 cell margin on boundaries
    for r in range(1, N-1):
        for c in range(1, N-1):
            # Insert random error at an edge including this vertex
            if random.random() < p:
                rand = random.randint(1, 4)
                if rand == 1 and r+1 <= N:
                    errors[r][c][1] += 1
                    errors[r][c][1] %= 2
                elif rand == 2 and r-1 >= 0:
                    errors[r][c][0] += 1
                    errors[r][c][0] %= 2
                elif rand == 3 and c+1 <= N:
                    errors[r][c][3] += 1
                    errors[r][c][3] %= 2
                elif c-1 >= 0:
                    errors[r][c][2] += 1
                    errors[r][c][2] %= 2
    
    #myprint(errors)

    # Get syndrome
    syndrome = []
    for r in range(1, N-1):
        for c in range(1, N-1):
            parity = 0
            parity += errors[r][c][0]
            parity += errors[r][c][1]
            parity += errors[r][c][2]
            parity += errors[r][c][3]

            # from other sides
            parity += errors[r-1][c][1]
            parity += errors[r+1][c][0]
            parity += errors[r][c-1][3]
            parity += errors[r][c+1][2]

            parity %= 2

            if parity == 1:
                syndrome.append([r, c])
    
    return syndrome

# Generate syndromes after distributing errors
# N = number of cells in plane
# p = probability of error
# p_corr = probability of error on non-error qubit if neighbor has error
def distribute_errors_corr(N, p, p_corr):
    # errors[r][c] = [up, down, left, right]
    # entries are 0 if no error, 1 if error on that qubit
    errors = [[[0, 0, 0, 0] for x in range(N+1)] for y in range(N+1)]
    
    # Distribute errors; leave 1 cell margin on boundaries
    for r in range(1, N-1):
        for c in range(1, N-1):
            # Insert random error at an edge including this vertex
            if (random.random() < p) or (random.random() < p_corr and (errors[r-1][c] == 1 or errors[r+1][c] == 1 or errors[r][c-1] == 1 or errors[r][c+1] == 1)):
                rand = random.randint(1, 4)
                if rand == 1 and r+1 <= N:
                    errors[r][c][1] += 1
                    errors[r][c][1] %= 2
                elif rand == 2 and r-1 >= 0:
                    errors[r][c][0] += 1
                    errors[r][c][0] %= 2
                elif rand == 3 and c+1 <= N:
                    errors[r][c][3] += 1
                    errors[r][c][3] %= 2
                elif c-1 >= 0:
                    errors[r][c][2] += 1
                    errors[r][c][2] %= 2
    
    #myprint(errors)

    # Get syndrome
    syndrome = []
    for r in range(1, N-1):
        for c in range(1, N-1):
            parity = 0
            parity += errors[r][c][0]
            parity += errors[r][c][1]
            parity += errors[r][c][2]
            parity += errors[r][c][3]

            # from other sides
            parity += errors[r-1][c][1]
            parity += errors[r+1][c][0]
            parity += errors[r][c-1][3]
            parity += errors[r][c+1][2]

            parity %= 2

            if parity == 1:
                syndrome.append([r, c])
    
    return syndrome


# Generate dense graph from syndrome
# Return edges after abstract-ifying nodes, weight = Manhattan distance
def makegraph(syndrome):
    # print(syndrome)
    edges = []
    for i in range(0, len(syndrome)):
        for j in range(i+1, len(syndrome)):
            edges.append((
                i,
                j,
                abs(syndrome[i][0] - syndrome[j][0]) + abs(syndrome[i][1] - syndrome[j][1])
            ))
    
    return edges
    # print(edges)


# Pure Blossom in O(N^3)
def blossom(syndrome, edges):
    G = nx.Graph()
    G.add_weighted_edges_from(edges)

    # Find edges
    start = time.perf_counter() # BENCHMARK ---------------------
    ans = nx.min_weight_matching(G)
    end = time.perf_counter() # BENCHMARK ---------------------
    t = end - start

    # Find maximal reversed weight (i.e. minimal weight)
    weight = 0
    for an in ans:
        weight += abs(syndrome[an[0]][0] - syndrome[an[1]][0]) + abs(syndrome[an[0]][1] - syndrome[an[1]][1])
    
    return t, weight

# Greedy heuristic in O(N^2 log N)
def greedy(syndrome, edges):
    my_edges = edges.copy()

    has_partner = []
    for i in range(len(syndrome)):
        has_partner.append(0)

    start = time.perf_counter() # BENCHMARK ---------------------
    my_edges.sort(key=lambda x:x[2])

    # Sorted by smallest edge weight first
    weight = 0
    for edge in my_edges:
        if has_partner[edge[0]] != 0:
            continue
        if has_partner[edge[1]] != 0:
            continue

        has_partner[edge[0]] = 1
        has_partner[edge[1]] = 1

        weight += edge[2]

    end = time.perf_counter() # BENCHMARK ---------------------
    t = end - start

    return t, weight

def serpentine(N, syndrome, edges):
    my_syn1 = syndrome.copy()
    my_syn2 = syndrome.copy()
    my_syn3 = syndrome.copy()
    my_syn4 = syndrome.copy()
    
    start = time.perf_counter() # BENCHMARK ---------------------
    # pos[r][c] = position along serpentine curve
    # 4 different orientations
    pos1 = [[0 for x in range(N+1)] for y in range(N+1)]
    pos2 = [[0 for x in range(N+1)] for y in range(N+1)]
    pos3 = [[0 for x in range(N+1)] for y in range(N+1)]
    pos4 = [[0 for x in range(N+1)] for y in range(N+1)]
    for r in range(N+1):
        for c in range(N+1):
            if c % 2 == 0:
                pos1[r][c] = c*(N+1) + (N+1) - r - 1
                pos2[r][c] = c*(N+1) + r
            else:
                pos1[r][c] = c*(N+1) + r
                pos2[r][c] = c*(N+1) + (N+1) - r - 1
            
            if r %2 == 0:
                pos3[r][c] = r*(N+1) + (N+1) - c - 1
                pos4[r][c] = r*(N+1) + c
            else:
                pos3[r][c] = r*(N+1) + c
                pos4[r][c] = r*(N+1) + (N+1) - c - 1


    # Sort syndromes by their position along the serpentine curve
    # Choose best option from 4
    my_syn1.sort(key=lambda x:pos1[x[0]][x[1]])
    my_syn2.sort(key=lambda x:pos2[x[0]][x[1]])
    my_syn3.sort(key=lambda x:pos3[x[0]][x[1]])
    my_syn4.sort(key=lambda x:pos4[x[0]][x[1]])

    i = 0
    w1, w2, w3, w4 = 0, 0, 0, 0
    while len(my_syn1) > 1:
        if(i+1 >= len(my_syn1)):
            break

        # Manhattan distance
        w1 += abs(my_syn1[i][0] - my_syn1[i+1][0]) + abs(my_syn1[i][1] - my_syn1[i+1][1])
        w2 += abs(my_syn2[i][0] - my_syn2[i+1][0]) + abs(my_syn2[i][1] - my_syn2[i+1][1])
        w3 += abs(my_syn3[i][0] - my_syn3[i+1][0]) + abs(my_syn3[i][1] - my_syn3[i+1][1])
        w4 += abs(my_syn4[i][0] - my_syn4[i+1][0]) + abs(my_syn4[i][1] - my_syn4[i+1][1])

        i += 2
    
    end = time.perf_counter() # BENCHMARK ---------------------
    t = end - start

    return t, min(w1, w2, w3, w4)

def nearest_neighbor(syndrome, edges):
    # Find nearest-neighbor in O(N^2)
    # Mark connected components
    # Go through each connected component and perform blossom
    # Using the edges returned by blossom, knock out paired vertices
    # Perform a final blossom unpaired vertices

    start = time.perf_counter() # BENCHMARK ---------------------

    # Find nearest neighbor in O(N^2)
    adjList = [[0 for x in range(len(syndrome))] for y in range(len(syndrome))]
    for edge in edges:
        adjList[edge[0]][edge[1]] = edge[2]
        adjList[edge[1]][edge[0]] = edge[2]

    #myprint(adjList)

    # s-th syndrome
    # Find nearest neighbor edges
    nn_edges = []
    for s in range(len(syndrome)):
        mn = math.inf
        s_pair = -1
        for i in range(len(syndrome)):
            if i == s:
                continue
            if adjList[s][i] < mn:
                mn = adjList[s][i]
                s_pair = i
        
        if s_pair != -1:
            nn_edges.append([s, s_pair, adjList[s][s_pair]])
    
    #print("NN edges")
    #myprint(nn_edges)

    # Run blossom on this (hopefully) smaller graph of connected components
    G = nx.Graph()
    G.add_weighted_edges_from(nn_edges)
    ans = nx.min_weight_matching(G)

    #print("initial")
    #print(ans)

    unpaired = [True for x in range(len(syndrome))]
    for edge in ans:
        unpaired[edge[0]] = False
        unpaired[edge[1]] = False

    # For all remaining unpaired syndrome vertices, construct an edgelist
    remaining = []
    for s in range(len(syndrome)):
        if unpaired[s] == True:
            remaining.append(s)
    
    remaining_edges = []
    for i in range(0, len(remaining)):
        for j in range(i+1, len(remaining)):
            remaining_edges.append([
                remaining[i],
                remaining[j],
                adjList[remaining[i]][remaining[j]]
            ])

    #print("remaining edges")
    #print(remaining_edges)

    G2 = nx.Graph()
    G2.add_weighted_edges_from(remaining_edges)
    ans2 = nx.min_weight_matching(G2)

    end = time.perf_counter() # BENCHMARK ---------------------
    t = end - start

    #print("remaining")
    #print(ans2)
    
    weight = 0
    DEBUG = []
    for edge in ans:
        DEBUG.append(edge)
        weight += abs(syndrome[edge[0]][0] - syndrome[edge[1]][0]) + abs(syndrome[edge[0]][1] - syndrome[edge[1]][1])
    
    for edge in ans2:
        DEBUG.append(edge)
        weight += abs(syndrome[edge[0]][0] - syndrome[edge[1]][0]) + abs(syndrome[edge[0]][1] - syndrome[edge[1]][1])

    #print(DEBUG)    

    return t, weight
    # Using the resulting list of edges, get a list of vertices that remain unpaired.
    # Construct all possible edges between them (use an adjList), and run blossom on it one last time
    
# OLD CODE: testing individual runs
def run():
    N=20
    p=0.25
    syndrome = distribute_errors(N, p)
    edges = makegraph(syndrome)

    #print(syndrome)
    #print(edges)

    # Blossom benchmark
    t, weight = blossom(syndrome, edges)
    print("Blossom:")
    print(f"   time={t}")
    print(f"   weight={weight}")
    print()

    # ------------------------------------
    # Greedy benchmark
    t, weight = greedy(syndrome, edges)
    print("Greedy:")
    print(f"   time={t}")
    print(f"   weight={weight}")
    print()

    # ------------------------------------
    # Serpentine curve benchmark
    t, weight = serpentine(N, syndrome, edges)
    print("Serpentine:")
    print(f"   time={t}")
    print(f"   weight={weight}")
    print()

    # ------------------------------------
    # Nearest neighbor benchmark
    t, weight = nearest_neighbor(syndrome, edges)
    print("Nearest neighbor:")
    print(f"   time={t}")
    print(f"   weight={weight}")
    print()


# Simulation of each heuristic
def simulate_blossom(N, p, trials):
    t_tot = 0
    w_tot = 0
    for t in range(trials):
        syndrome = distribute_errors(N, p)
        edges = makegraph(syndrome)
        t, weight = blossom(syndrome, edges)

        t_tot += t
        w_tot += weight

    # Return averages
    t_tot /= trials
    w_tot /= trials
    return t_tot, w_tot

def simulate_greedy(N, p, trials):
    t_tot = 0
    w_tot = 0
    for t in range(trials):
        syndrome = distribute_errors(N, p)
        edges = makegraph(syndrome)
        t, weight = greedy(syndrome, edges)

        t_tot += t
        w_tot += weight

    # Return averages
    t_tot /= trials
    w_tot /= trials
    return t_tot, w_tot

def simulate_serpentine(N, p, trials):
    t_tot = 0
    w_tot = 0
    for t in range(trials):
        syndrome = distribute_errors(N, p)
        edges = makegraph(syndrome)
        t, weight = serpentine(N, syndrome, edges)

        t_tot += t
        w_tot += weight

    # Return averages
    t_tot /= trials
    w_tot /= trials
    return t_tot, w_tot

def simulate_NN(N, p, trials):
    t_tot = 0
    w_tot = 0
    for t in range(trials):
        syndrome = distribute_errors(N, p)
        edges = makegraph(syndrome)
        t, weight = nearest_neighbor(syndrome, edges)

        t_tot += t
        w_tot += weight

    # Return averages
    t_tot /= trials
    w_tot /= trials
    return t_tot, w_tot

# Simulate each heuristic with random trials
def simulate():
    p = 0.02
    p_corr = 0.4
    TRIALS = 100 # Number of trials per value of N

    n_blossom = []
    t_blossom = []
    w_blossom = []
    t_greedy = []
    w_greedy = []
    t_serpentine = []
    w_serpentine = []
    t_NN = []
    w_NN = []

    for N in range(0, 41):
        t1, w1 = 0, 0
        t2, w2 = 0, 0
        t3, w3 = 0, 0
        t4, w4 = 0, 0

        # Trials
        for t in range(TRIALS):
            #syndrome = distribute_errors(N, p)
            syndrome = distribute_errors_corr(N, p, p_corr)
            edges = makegraph(syndrome)
            
            t1_test, w1_test = blossom(syndrome, edges)
            t2_test, w2_test = greedy(syndrome, edges)
            t3_test, w3_test = serpentine(N, syndrome, edges)
            t4_test, w4_test = nearest_neighbor(syndrome, edges)
            
            w1 += 1
            w2 += (w2_test/w1_test) if w1_test > 0 else 1
            w3 += (w3_test/w1_test) if w1_test > 0 else 1
            w4 += (w4_test/w1_test) if w1_test > 0 else 1

            t1 += t1_test
            t2 += t2_test
            t3 += t3_test
            t4 += t4_test
        
        # Get average
        t1 /= TRIALS
        t2 /= TRIALS
        t3 /= TRIALS
        t4 /= TRIALS
        w1 /= TRIALS
        w2 /= TRIALS
        w3 /= TRIALS
        w4 /= TRIALS

        n_blossom.append(N)
        t_blossom.append(t1)
        w_blossom.append(w1)

        t_greedy.append(t2)
        w_greedy.append(w2)

        t_serpentine.append(t3)
        w_serpentine.append(w3)
        
        t_NN.append(t4)
        w_NN.append(w4)

        print(f"{N} done")

    # N vs average running time
    plt.figure(1)
    plt.scatter(n_blossom, t_blossom, color='blue', marker='o', label='Blossom', s=25)
    plt.scatter(n_blossom, t_greedy, color='cyan', marker='s', label='Greedy', s=25)
    #plt.scatter(n_blossom, t_serpentine, color='green', marker='D', label='Serpentine', s=25)
    plt.scatter(n_blossom, t_NN, color='red', marker='H', label='Nearest Neighbor', s=25)
    
    plt.title("Running Time of Different Heuristics-Based MWPM Algorithms")
    plt.xlabel("Lattice Size (L)")
    plt.ylabel("Average Running Time (sec)")
    plt.legend()
    plt.show()


    # N vs ratio to optimal weight
    plt.figure(2)
    plt.scatter(n_blossom, w_blossom, color='blue', marker='o', label='Blossom', s=25)
    plt.scatter(n_blossom, w_greedy, color='cyan', marker='s', label='Greedy', s=25)
    #plt.scatter(n_blossom, w_serpentine, color='green', marker='D', label='Serpentine', s=25)
    plt.scatter(n_blossom, w_NN, color='red', marker='H', label='Nearest Neighbor', s=25)
    
    plt.title("Solution Weight Ratio of Different Heuristics-Based MWPM Algorithms")
    plt.xlabel("Lattice Size (L)")
    plt.ylabel("Solution Weight Ratio (actual/optimal)")
    plt.legend()
    plt.show()

    # Misc. data
    for i in range(len(n_blossom)):
        if i % 5 == 0:
            print(f"L = {n_blossom[i]}")
            print(f"Time (blossom) = {t_blossom[i]}")
            print(f"Weight (blossom) = {w_blossom[i]}")
            print(f"Time (greedy) = {t_greedy[i]}")
            print(f"Weight (greedy) = {w_greedy[i]}")


#run()
simulate()
