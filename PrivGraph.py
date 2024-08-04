
# This file includes code from the following GitHub repository:
# Repository Name: https://github.com/username/repository
# Licensed under: LICENSE NAME

import community
import networkx as nx
import time
import numpy as np

from numpy.random import laplace
from sklearn import metrics
import pandas as pd
from utils import *
import community
import comm
import time
import random

import itertools
from heapq import *
import os


def community_init(mat0, mat0_graph, epsilon, nr, t=1.0):
    # t1 = time.time()
    # Divide the nodes randomly
    g1 = list(np.zeros(len(mat0)))
    ind = -1

    for i in range(len(mat0)):
        if i % nr == 0:
            ind = ind + 1
        g1[i] = ind

    random.shuffle(g1)

    mat0_par3 = {}
    for i in range(len(mat0)):
        mat0_par3[i] = g1[i]

    gr1 = max(mat0_par3.values()) + 1

    # mat0_mod3 = community.modularity(mat0_par3,mat0_graph)
    # print('mat0_mod2=%.3f,gr1=%d'%(mat0_mod3,gr1))

    mat0_par3_pv = np.array(list(mat0_par3.values()))
    mat0_par3_pvs = []
    for i in range(gr1):
        pv = np.where(mat0_par3_pv == i)[0]
        pvs = list(pv)
        mat0_par3_pvs.append(pvs)
    mat_one_level = np.zeros([gr1, gr1])

    for i in range(gr1):
        pi = mat0_par3_pvs[i]
        mat_one_level[i, i] = np.sum(mat0[np.ix_(pi, pi)])
        for j in range(i + 1, gr1):
            pj = mat0_par3_pvs[j]
            mat_one_level[i, j] = np.sum(mat0[np.ix_(pi, pj)])
    # print('generate new matrix time:%.2fs'%(time.time()-t1))

    lap_noise = laplace(0, 1 / epsilon, gr1 * gr1).astype(np.int32)
    lap_noise = lap_noise.reshape(gr1, gr1)

    ga = get_uptri_arr(mat_one_level, ind=1)
    ga_noise = ga + laplace(0, 1 / epsilon, len(ga))
    ga_noise_pp = FO_pp(ga_noise)
    mat_one_level_noise = get_upmat(ga_noise_pp, gr1, ind=1)

    noise_diag = np.int32(mat_one_level.diagonal() + laplace(0, 2 / epsilon, len(mat_one_level)))

    # keep consistency
    noise_diag = FO_pp(noise_diag)

    mat_one_level_noise = np.triu(mat_one_level_noise, 1)
    mat_one_level_noise = mat_one_level_noise + np.transpose(mat_one_level_noise)

    row, col = np.diag_indices_from(mat_one_level_noise)
    mat_one_level_noise[row, col] = noise_diag
    mat_one_level_noise[mat_one_level_noise < 0] = 0

    mat_one_level_graph = nx.from_numpy_array(mat_one_level_noise, create_using=nx.Graph)

    # Apply the Louvain method
    mat_new_par = community.best_partition(mat_one_level_graph, resolution=t)
    gr2 = max(mat_new_par.values()) + 1
    mat_new_pv = np.array(list(mat_new_par.values()))
    mat_final_pvs = []
    for i in range(gr2):
        pv = np.where(mat_new_pv == i)[0]
        mat_final_pv = []
        for j in range(len(pv)):
            pvj = pv[j]
            mat_final_pv.extend(mat0_par3_pvs[pvj])
        mat_final_pvs.append(mat_final_pv)

    label1 = np.zeros([len(mat0)], dtype=np.int32)
    for i in range(len(mat_final_pvs)):
        label1[mat_final_pvs[i]] = i

    return label1


def get_uptri_arr(mat_init, ind=0):
    a = len(mat_init)
    res = []
    for i in range(a):
        dat = mat_init[i][i + ind:]
        res.extend(dat)
    arr = np.array(res)
    return arr


def get_upmat(arr, k, ind=0):
    mat = np.zeros([k, k], dtype=np.int32)
    left = 0
    for i in range(k):
        delta = k - i - ind
        mat[i, i + ind:] = arr[left:left + delta]
        left = left + delta

    return mat


# Post processing
def FO_pp(data_noise, type='norm_sub'):
    if type == 'norm_sub':
        data = norm_sub_deal(data_noise)

    if type == 'norm_mul':
        data = norm_mul_deal(data_noise)

    return data


def norm_sub_deal(data):
    data = np.array(data, dtype=np.int32)
    data_min = np.min(data)
    data_sum = np.sum(data)
    delta_m = 0 - data_min

    if delta_m > 0:
        dm = 100000000
        data_seq = np.zeros([len(data)], dtype=np.int32)
        for i in range(0, delta_m):
            data_t = data - i
            data_t[data_t < 0] = 0
            data_t_s = np.sum(data_t)
            dt = np.abs(data_t_s - data_sum)
            if dt < dm:
                dm = dt
                data_seq = data_t
                if dt == 0:
                    break

    else:
        data_seq = data
    return data_seq


# generate graph(intra edges) based on degree sequence
def generate_intra_edge(dd1, div=1):
    dd1 = np.array(dd1, dtype=np.int32)
    dd1[dd1 < 0] = 0
    dd1_len = len(dd1)
    dd1_p = dd1.reshape(dd1_len, 1) * dd1.reshape(1, dd1_len)
    s1 = np.sum(dd1)

    dd1_res = np.zeros([dd1_len, dd1_len], dtype=np.int8)
    if s1 > 0:
        batch_num = int(dd1_len / div)
        begin_id = 0
        for i in range(div):
            if i == div - 1:
                batch_n = dd1_len - begin_id
                dd1_r = np.random.randint(0, high=s1, size=(batch_n, dd1_len))
                res = dd1_p[begin_id:, :] - dd1_r
                res[res > 0] = 1
                res[res < 1] = 0
                dd1_res[begin_id:, :] = res
            else:
                dd1_r = np.random.randint(0, high=s1, size=(batch_num, dd1_len))
                res = dd1_p[begin_id:begin_id + batch_num, :] - dd1_r
                res[res > 0] = 1
                res[res < 1] = 0
                dd1_res[begin_id:begin_id + batch_num, :] = res
                begin_id = begin_id + batch_num

    # make sure the final adjacency matrix is symmetric
    dd1_out = np.triu(dd1_res, 1)
    dd1_out = dd1_out + np.transpose(dd1_out)
    return dd1_out
class PriorityQueue(object):
    def __init__(self):
        self.pq = []  # list of entries arranged in a heap
        self.entry_finder = {}  # mapping of tasks to entries
        self.REMOVED = '<removed-task>'  # placeholder for a removed task
        self.counter = itertools.count()  # unique sequence count

    def add_task(self, task, priority=0):
        'Add a new task or update the priority of an existing task'
        if task in self.entry_finder:
            self.remove_task(task)
        count = next(self.counter)
        entry = [priority, count, task]
        self.entry_finder[task] = entry
        heappush(self.pq, entry)

    def remove_task(self, task):
        'Mark an existing task as REMOVED.  Raise KeyError if not found.'
        entry = self.entry_finder.pop(task)
        entry[-1] = self.REMOVED

    def pop_item(self):
        'Remove and return the lowest priority task. Raise KeyError if empty.'
        while self.pq:
            priority, count, task = heappop(self.pq)
            if task is not self.REMOVED:
                del self.entry_finder[task]
                return task, priority
        raise KeyError('pop from an empty priority queue')

    def __str__(self):
        return str([entry for entry in self.pq if entry[2] != self.REMOVED])


def degreeDiscountIC(G, k, p=0.01):
    S = []
    dd = PriorityQueue()  # degree discount
    t = dict()  # number of adjacent vertices that are in S
    d = dict()  # degree of each vertex

    # initialize degree discount
    for u in G.nodes():
        d[u] = sum([G[u][v]['weight'] for v in G[u]])  # each edge adds degree 1
        # d[u] = len(G[u]) # each neighbor adds degree 1
        dd.add_task(u, -d[u])  # add degree of each node
        t[u] = 0

    # add vertices to S greedily
    for i in range(k):
        u, priority = dd.pop_item()  # extract node with maximal degree discount
        S.append(u)
        for v in G[u]:
            if v not in S:
                t[v] += G[u][v]['weight']  # increase number of selected neighbors
                priority = d[v] - 2 * t[v] - (d[v] - t[v]) * t[v] * p  # discount of degree
                dd.add_task(v, -priority)
    return S


def runIC(G, S, p=0.01):
    from copy import deepcopy
    from random import random
    T = deepcopy(S)  # copy already selected nodes

    i = 0
    while i < len(T):
        for v in G[T[i]]:  # for neighbors of a selected node
            if v not in T:  # if it wasn't selected yet
                w = G[T[i]][v]['weight']  # count the number of edges between two nodes
                if random() <= 1 - (1 - p) ** w:  # if at least one of edges propagate influence
                    # print (T[i], 'influences', v)
                    T.append(v)
        i += 1
    return T


def find_seed(graph_path, seed_size=20):
    # read in graph
    G = nx.Graph()
    with open(graph_path) as f:

        for line in f:
            u, v = map(int, line.split())
            try:
                G[u][v]['weight'] += 1
            except:
                G.add_edge(u, v, weight=1)

    S = degreeDiscountIC(G, seed_size)
    return S


def cal_spread(graph_path, S_all, p=0.01, seed_size=20, iterations=100):
    # read in graph
    G = nx.Graph()
    with open(graph_path) as f:

        for line in f:
            u, v = map(int, line.split())
            # print('u:%s,v:%s'%(u,v))
            try:
                G[u][v]['weight'] += 1
            except:
                G.add_edge(u, v, weight=1)

    # calculate initial set

    if seed_size <= len(S_all):
        S = S_all[:seed_size]
    else:
        print('seed_size is too large.')
        S = S_all

    avg = 0
    for i in range(iterations):
        T = runIC(G, S, p)
        avg += float(len(T)) / iterations

    avg_final = int(round(avg))

    return avg_final

def priv_graph(mat0, epsilon):
    e1_r = 1 / 3
    e2_r = 1 / 3
    N = 20
    t = 1.0
    e1 = e1_r * epsilon

    e2 = e2_r * epsilon
    e3_r = 1 - e1_r - e2_r

    e3 = e3_r * epsilon

    ed = e3
    ev = e3

    ev_lambda = 1 / ed
    dd_lam = 2 / ev
    mat0_graph = nx.from_numpy_array(mat0, create_using=nx.Graph)
    mat0_node = mat0_graph.number_of_nodes()
    mat1_pvarr1 = community_init(mat0, mat0_graph, epsilon=e1, nr=N, t=t)

    part1 = {}
    for i in range(len(mat1_pvarr1)):
        part1[i] = mat1_pvarr1[i]


    # Community Adjustment
    mat1_par1 = comm.best_partition(mat0_graph, part1, epsilon_EM=e2)
    mat1_pvarr = np.array(list(mat1_par1.values()))

    # Information Extraction
    mat1_pvs = []
    for i in range(max(mat1_pvarr) + 1):
        pv1 = np.where(mat1_pvarr == i)[0]
        pvs = list(pv1)
        mat1_pvs.append(pvs)

    comm_n = max(mat1_pvarr) + 1

    ev_mat = np.zeros([comm_n, comm_n], dtype=np.int64)

    # edge vector
    for i in range(comm_n):
        pi = mat1_pvs[i]
        ev_mat[i, i] = np.sum(mat0[np.ix_(pi, pi)])
        for j in range(i + 1, comm_n):
            pj = mat1_pvs[j]
            ev_mat[i, j] = int(np.sum(mat0[np.ix_(pi, pj)]))
            ev_mat[j, i] = ev_mat[i, j]

    ga = get_uptri_arr(ev_mat, ind=1)
    ga_noise = ga + laplace(0, ev_lambda, len(ga))

    ga_noise_pp = FO_pp(ga_noise)
    ev_mat = get_upmat(ga_noise_pp, comm_n, ind=1)

    # degree sequence
    dd_s = []
    for i in range(comm_n):
        dd1 = mat0[np.ix_(mat1_pvs[i], mat1_pvs[i])]
        dd1 = np.sum(dd1, 1)

        dd1 = (dd1 + laplace(0, dd_lam, len(dd1))).astype(int)
        dd1 = FO_pp(dd1)
        dd1[dd1 < 0] = 0
        dd1[dd1 >= len(dd1)] = len(dd1) - 1

        dd1 = list(dd1)
        dd_s.append(dd1)

    # Graph Reconstruction
    mat2 = np.zeros([mat0_node, mat0_node], dtype=np.int8)
    for i in range(comm_n):
        # Intra-community
        dd_ind = mat1_pvs[i]
        dd1 = dd_s[i]
        mat2[np.ix_(dd_ind, dd_ind)] = generate_intra_edge(dd1)

        # Inter-community
        for j in range(i + 1, comm_n):
            ev1 = ev_mat[i, j]
            pj = mat1_pvs[j]
            if ev1 > 0:
                c1 = np.random.choice(pi, ev1)
                c2 = np.random.choice(pj, ev1)
                for ind in range(ev1):
                    mat2[c1[ind], c2[ind]] = 1
                    mat2[c2[ind], c1[ind]] = 1

    mat2 = mat2 + np.transpose(mat2)
    mat2 = np.triu(mat2, 1)
    mat2 = mat2 + np.transpose(mat2)
    mat2[mat2 > 0] = 1
    return mat2




def main_func(dataset_name='Chamelon', eps=[0.5, 1, 1.5, 2, 2.5, 3, 3.5], e1_r=1 / 3, e2_r=1 / 3, N=20, t=1.0,
              exp_num=10, save_csv=True):
    t_begin = time.time()

    data_path = './data/' + dataset_name + '.txt'
    mat0, mid = get_mat(data_path)

    cols = ['eps', 'exper', 'num_node_RE','num_edge_RE', 'tria_count_RE','avg_deg_RE','deg_var_RE','deg_dsb_KL','diam_RE','SP_RE','dis_dsb_KL','GCC_RE','ACC_RE', 'CD_NMI','MOD_RE','Ass_RE', 'evc_MAE']

    all_data = pd.DataFrame(None, columns=cols)

    # original graph
    mat0_graph = nx.from_numpy_array(mat0, create_using=nx.Graph)

    mat0_edge = mat0_graph.number_of_edges()
    mat0_node = mat0_graph.number_of_nodes()
    print('Dataset:%s' % (dataset_name))
    print('Node number:%d' % (mat0_graph.number_of_nodes()))
    print('Edge number:%d' % (mat0_graph.number_of_edges()))

    mat0_par = community.best_partition(mat0_graph)
    mat0_degree = np.sum(mat0, 0)
    mat0_avg_degree = np.mean(mat0_degree)
    mat0_degree_variance = np.var(mat0_degree, ddof=1)
    mat0_deg_dist = np.bincount(np.int64(mat0_degree))  # degree distribution
    mat0_dis_dist = distance_distribution(mat0_graph)   # distance distribution
    mat0_evc = nx.eigenvector_centrality(mat0_graph, max_iter=10000)
    mat0_evc_a = dict(sorted(mat0_evc.items(), key=lambda x: x[1], reverse=True))
    mat0_evc_ak = list(mat0_evc_a.keys())
    mat0_evc_val = np.array(list(mat0_evc_a.values()))
    evc_kn = np.int64(0.01 * mat0_node)
    mat0_diam = cal_diam(mat0)
    mat0_cc = nx.transitivity(mat0_graph)
    mat0_mod = community.modularity(mat0_par, mat0_graph)
    mat0_total_triangle = sum(nx.triangles(mat0_graph).values()) // 3
    mat0_acc = nx.average_clustering(mat0_graph)
    mat0_ass = nx.degree_assortativity_coefficient(mat0_graph)
    mat0_SP = average_shortest_path_length_custom(mat0_graph)


    all_num_node_RE = []
    all_num_edge_RE = []
    all_tria_count_RE = []
    all_avg_deg_RE = []
    all_deg_var_RE = []
    all_deg_dsb_KL = []
    all_diam_RE = []
    all_SP_RE = []
    all_dis_dsb_KL = []
    all_GCC_RE = []
    all_ACC_RE = []
    all_CD_NMI = []
    all_MOD_RE = []
    all_Ass_RE = []
    all_evc_MAE = []

    for index, epsilon in enumerate(eps):
        ti = time.time()

        e1 = e1_r * epsilon

        e2 = e2_r * epsilon
        e3_r = 1 - e1_r - e2_r

        e3 = e3_r * epsilon

        ed = e3
        ev = e3

        ev_lambda = 1 / ed
        dd_lam = 2 / ev

        num_node_RE_arr = np.zeros([exp_num])
        num_edge_RE_arr = np.zeros([exp_num])
        tria_count_RE_arr = np.zeros([exp_num])
        avg_deg_RE_arr = np.zeros([exp_num])
        deg_var_RE_arr = np.zeros([exp_num])
        deg_dsb_KL_arr = np.zeros([exp_num])
        diam_RE_arr = np.zeros([exp_num])
        SP_RE_arr = np.zeros([exp_num])
        dis_dsb_KL_arr = np.zeros([exp_num])
        GCC_RE_arr = np.zeros([exp_num])
        ACC_RE_arr = np.zeros([exp_num])
        CD_NMI_arr = np.zeros([exp_num])
        MOD_RE_arr = np.zeros([exp_num])
        Ass_RE_arr = np.zeros([exp_num])
        evc_MAE_arr = np.zeros([exp_num])

        for exper in range(exp_num):
            print('-----------epsilon=%.1f,exper=%d/%d-------------' % (epsilon, exper + 1, exp_num))

            t1 = time.time()

            # Community Initialization
            mat1_pvarr1 = community_init(mat0, mat0_graph, epsilon=e1, nr=N, t=t)

            part1 = {}
            for i in range(len(mat1_pvarr1)):
                part1[i] = mat1_pvarr1[i]

            # Community Adjustment
            mat1_par1 = comm.best_partition(mat0_graph, part1, epsilon_EM=e2)
            mat1_pvarr = np.array(list(mat1_par1.values()))

            # Information Extraction
            mat1_pvs = []
            for i in range(max(mat1_pvarr) + 1):
                pv1 = np.where(mat1_pvarr == i)[0]
                pvs = list(pv1)
                mat1_pvs.append(pvs)

            comm_n = max(mat1_pvarr) + 1

            ev_mat = np.zeros([comm_n, comm_n], dtype=np.int64)

            # edge vector
            for i in range(comm_n):
                pi = mat1_pvs[i]
                ev_mat[i, i] = np.sum(mat0[np.ix_(pi, pi)])
                for j in range(i + 1, comm_n):
                    pj = mat1_pvs[j]
                    ev_mat[i, j] = int(np.sum(mat0[np.ix_(pi, pj)]))
                    ev_mat[j, i] = ev_mat[i, j]

            ga = get_uptri_arr(ev_mat, ind=1)
            ga_noise = ga + laplace(0, ev_lambda, len(ga))

            ga_noise_pp = FO_pp(ga_noise)
            ev_mat = get_upmat(ga_noise_pp, comm_n, ind=1)

            # degree sequence
            dd_s = []
            for i in range(comm_n):
                dd1 = mat0[np.ix_(mat1_pvs[i], mat1_pvs[i])]
                dd1 = np.sum(dd1, 1)

                dd1 = (dd1 + laplace(0, dd_lam, len(dd1))).astype(int)
                dd1 = FO_pp(dd1)
                dd1[dd1 < 0] = 0
                dd1[dd1 >= len(dd1)] = len(dd1) - 1

                dd1 = list(dd1)
                dd_s.append(dd1)

            # Graph Reconstruction
            mat2 = np.zeros([mat0_node, mat0_node], dtype=np.int8)
            for i in range(comm_n):
                # Intra-community
                dd_ind = mat1_pvs[i]
                dd1 = dd_s[i]
                mat2[np.ix_(dd_ind, dd_ind)] = generate_intra_edge(dd1)

                # Inter-community
                for j in range(i + 1, comm_n):
                    ev1 = ev_mat[i, j]
                    pj = mat1_pvs[j]
                    if ev1 > 0:
                        c1 = np.random.choice(pi, ev1)
                        c2 = np.random.choice(pj, ev1)
                        for ind in range(ev1):
                            mat2[c1[ind], c2[ind]] = 1
                            mat2[c2[ind], c1[ind]] = 1

            mat2 = mat2 + np.transpose(mat2)
            mat2 = np.triu(mat2, 1)
            mat2 = mat2 + np.transpose(mat2)
            mat2[mat2 > 0] = 1

            mat2_graph = nx.from_numpy_array(mat2, create_using=nx.Graph)

            # save the graph
            # file_name = './result/' +  'PrivGraph_%s_%.1f_%d.txt' %(dataset_name,epsilon,exper)
            # write_edge_txt(mat2,mid,file_name)

            # evaluate
            mat2_edge = mat2_graph.number_of_edges()
            mat2_node = mat2_graph.number_of_nodes()
            mat2_total_triangle = sum(nx.triangles(mat2_graph).values()) // 3
            mat2_degree = np.sum(mat2, 0)
            mat2_avg_degree = np.mean(mat2_degree)
            mat2_degree_variance = np.var(mat2_degree, ddof=1)
            mat2_SP = average_shortest_path_length_custom(mat2_graph)
            mat2_dis_dist = distance_distribution(mat2_graph)
            mat2_par = community.best_partition(mat2_graph)
            mat2_mod = community.modularity(mat2_par, mat2_graph)

            mat2_cc = nx.transitivity(mat2_graph)

            mat2_deg_dist = np.bincount(np.int64(mat2_degree))  # degree distribution
            mat2_acc = nx.average_clustering(mat2_graph)
            mat2_ass = nx.degree_assortativity_coefficient(mat2_graph)
            mat2_evc = nx.eigenvector_centrality(mat2_graph, max_iter=10000)
            mat2_evc_a = dict(sorted(mat2_evc.items(), key=lambda x: x[1], reverse=True))
            mat2_evc_ak = list(mat2_evc_a.keys())
            mat2_evc_val = np.array(list(mat2_evc_a.values()))

            mat2_diam = cal_diam(mat2)

            # calculate the metrics
            # clustering coefficent
            cc_rel = cal_rel(mat0_cc, mat2_cc)

            num_node_RE = cal_rel(mat0_node, mat2_node)
            num_edge_RE = cal_rel(mat0_edge, mat2_edge)
            tria_count_RE = cal_rel(mat0_total_triangle, mat2_total_triangle)
            avg_deg_RE = cal_rel(mat0_avg_degree, mat2_avg_degree)
            deg_var_RE = cal_rel(mat0_degree_variance, mat2_degree_variance)
            SP_RE = cal_rel(mat0_SP, mat2_SP)
            dis_dsb_KL = cal_kl(mat0_dis_dist, mat2_dis_dist)
            ACC_RE = cal_rel(mat0_acc, mat2_acc)
            Ass_RE = cal_rel(mat0_ass, mat2_ass)

            # degree distribution
            deg_kl = cal_kl(mat0_deg_dist, mat2_deg_dist)

            # modularity
            mod_rel = cal_rel(mat0_mod, mat2_mod)

            # NMI
            labels_true = list(mat0_par.values())
            labels_pred = list(mat2_par.values())
            nmi = metrics.normalized_mutual_info_score(labels_true, labels_pred)

            # MAE of EVC
            evc_MAE = cal_MAE(mat0_evc_val, mat2_evc_val, k=evc_kn)

            # diameter
            diam_rel = cal_rel(mat0_diam, mat2_diam)

            num_node_RE_arr[exper] = num_node_RE
            num_edge_RE_arr[exper] = num_edge_RE
            tria_count_RE_arr[exper] = tria_count_RE
            avg_deg_RE_arr[exper] = avg_deg_RE
            deg_var_RE_arr[exper] = deg_var_RE
            deg_dsb_KL_arr[exper] = deg_kl
            diam_RE_arr[exper] = diam_rel
            SP_RE_arr[exper] = SP_RE
            dis_dsb_KL_arr[exper] = dis_dsb_KL
            GCC_RE_arr[exper] = cc_rel
            ACC_RE_arr[exper] = ACC_RE
            CD_NMI_arr[exper] = nmi
            MOD_RE_arr[exper] = mod_rel
            Ass_RE_arr[exper] = Ass_RE
            evc_MAE_arr[exper] = evc_MAE

            print(
                'Nodes=%d,Edges=%d,num_node_RE=%.4f,num_edge_RE=%.4f,tria_count_RE=%.4f,avg_deg_RE=%.4f,deg_var_RE=%.4f,deg_dsb_KL=%.4f,diam_RE=%.4f,SP_RE=%.4f,dis_dsb_KL=%.4f,GCC_RE=%.4f,ACC_RE=%.4f,CD_NMI=%.4f,MOD_RE=%.4f,Ass_RE=%.4f,evc_MAE=%.4f' \
                % (
                mat2_node, mat2_edge, num_node_RE, num_edge_RE, tria_count_RE, avg_deg_RE, deg_var_RE, deg_kl, diam_rel,
                SP_RE, dis_dsb_KL, cc_rel, ACC_RE, nmi, mod_rel, Ass_RE, evc_MAE))

            data_col = [epsilon, exper, num_node_RE, num_edge_RE, tria_count_RE, avg_deg_RE, deg_var_RE, deg_kl,
                        diam_rel, SP_RE, dis_dsb_KL, cc_rel, ACC_RE, nmi, mod_rel, Ass_RE, evc_MAE]
            col_len = len(data_col)
            data_col = np.array(data_col).reshape(1, col_len)
            data1 = pd.DataFrame(data_col, columns=cols)
            all_data = all_data.append(data1)

        all_CD_NMI.append(np.mean(CD_NMI_arr))
        all_GCC_RE.append(np.mean(GCC_RE_arr))
        all_deg_dsb_KL.append(np.mean(deg_dsb_KL_arr))
        all_MOD_RE.append(np.mean(MOD_RE_arr))
        all_evc_MAE.append(np.mean(evc_MAE_arr))
        all_diam_RE.append(np.mean(diam_RE_arr))
        all_num_node_RE.append(np.mean(num_node_RE_arr))
        all_num_edge_RE.append(np.mean(num_node_RE_arr))
        all_tria_count_RE.append(np.mean(tria_count_RE_arr))
        all_avg_deg_RE.append(np.mean(avg_deg_RE_arr))
        all_deg_var_RE.append(np.mean(deg_var_RE_arr))
        all_SP_RE.append(np.mean(SP_RE_arr))
        all_dis_dsb_KL.append(np.mean(dis_dsb_KL_arr))
        all_ACC_RE.append(np.mean(ACC_RE_arr))
        all_Ass_RE.append(np.mean(Ass_RE_arr))

        print('all_index=%d/%d Done.%.2fs\n' % (index + 1, len(eps), time.time() - ti))
    res_path = './result'
    save_name = res_path + '/' + '%s_%d_PrivGraph_final_0.5.csv' % (dataset_name, exp_num)
    if not os.path.exists(res_path):
        os.mkdir(res_path)

    if save_csv == True:
        all_data.to_csv(save_name, index=False, sep=',')

    print('-----------------------------')

    print('dataset:', dataset_name)

    print('eps=', eps)
    print('all_num_node_RE=', all_num_node_RE)
    print('all_edge_RE=', all_num_edge_RE)
    print('triangle_RE=', all_tria_count_RE)
    print('all_avg_deg_RE=', all_avg_deg_RE)
    print('all_deg_var_RE=', all_deg_var_RE)
    print('all_deg_dsb_KL=', all_deg_dsb_KL)
    print('all_diam_RE=', all_diam_RE)
    print('all_SP_RE=', all_SP_RE)
    print('all_dis_dsb_KL=', all_dis_dsb_KL)
    print('all_GCC_RE=', all_GCC_RE)
    print('all_ACC_RE=', all_ACC_RE)
    print('all_CD_NMI=', all_CD_NMI)
    print('all_MOD_RE=', all_MOD_RE)
    print('all_Ass_RE=', all_Ass_RE)
    print('all_evc_MAE=', all_evc_MAE)

    print('All time:%.2fs' % (time.time() - t_begin))


if __name__ == '__main__':
    # set the dataset
    dataset_name = 'CA-HepPh'

    # set the privacy budget, list type
    eps = [0.1, 0.5, 1, 2, 5, 10]

    # set the ratio of the privacy budget
    e1_r = 1 / 3
    e2_r = 1 / 3

    # set the number of experiments
    exp_num = 5

    # set the number of nodes for community initialization
    n1 = 20

    # set the resolution parameter
    t = 1.0

    # run the function
    main_func(dataset_name=dataset_name, eps=eps, e1_r=e1_r, e2_r=e2_r, N=n1, t=t, exp_num=exp_num)



