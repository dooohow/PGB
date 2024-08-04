import numpy as np
import networkx as nx
from utils import *
import pandas as pd
from sklearn import metrics
import os

def add_laplace_noise(data, epsilon):
    """
    Add Laplace noise to the data for differential privacy.

    Parameters:
    data (list): The original data.
    epsilon (float): The privacy budget.

    Returns:
    noisy_data (list): The data with added Laplace noise.
    """
    noisy_data = []
    for value in data:
        noise = np.random.laplace(0, 1 / epsilon)
        noisy_value = max(0, round(value + noise))  # Ensure degrees are non-negative and integers
        noisy_data.append(noisy_value)
    return noisy_data


def ensure_even_degree_sum(degree_sequence):
    """
    Ensure the sum of the degree sequence is even by adjusting one degree if necessary.

    Parameters:
    degree_sequence (list): The degree sequence.

    Returns:
    adjusted_degree_sequence (list): The adjusted degree sequence with an even sum.
    """
    degree_sum = sum(degree_sequence)
    if degree_sum % 2 != 0:
        degree_sequence[0] += 1  # Adjust the first degree to make the sum even
    return degree_sequence


def create_bter_graph_from_adjacency_matrix(adj_matrix, epsilon):
    """
    Create a BTER graph given an adjacency matrix and epsilon for differential privacy.

    Parameters:
    adj_matrix (numpy.ndarray): Adjacency matrix of the input graph.
    epsilon (float): The privacy budget for differential privacy.

    Returns:
    numpy.ndarray: Adjacency matrix of the generated BTER graph.
    """
    # Get degree sequence from adjacency matrix
    degree_sequence = np.sum(adj_matrix, axis=1).tolist()

    # Add Laplace noise to the degree sequence
    noisy_degree_sequence = add_laplace_noise(degree_sequence, epsilon)

    # Ensure the degree sequence sum is even
    noisy_degree_sequence = ensure_even_degree_sum(noisy_degree_sequence)

    # Sort nodes by noisy degree
    sorted_nodes = sorted(range(len(noisy_degree_sequence)), key=lambda x: noisy_degree_sequence[x])
    noisy_degree_sequence = sorted(noisy_degree_sequence)

    # Phase 1: Create ER communities
    communities = []
    i = 0
    while i < len(noisy_degree_sequence):
        d = noisy_degree_sequence[i]
        community_size = int(d) + 1  # Ensure community size is an integer
        if i + community_size > len(noisy_degree_sequence):
            community_size = len(noisy_degree_sequence) - i
        community = sorted_nodes[i:i + community_size]
        communities.append(community)
        i += community_size

    # Create a graph
    G = nx.Graph()

    # Add nodes
    for node in range(len(noisy_degree_sequence)):
        G.add_node(node)

    # Connect nodes within each community
    for community in communities:
        p = np.log(len(community) + 1) / np.log(max(noisy_degree_sequence) + 1)
        for i in range(len(community)):
            for j in range(i + 1, len(community)):
                if np.random.rand() < p:
                    G.add_edge(community[i], community[j])

    # Phase 2: Connect communities using excess degrees
    excess_degrees = []
    for node in G.nodes():
        actual_degree = G.degree[node]
        target_degree = noisy_degree_sequence[node]
        excess_degree = max(target_degree - actual_degree, 0)
        excess_degrees.append(excess_degree)

    # Use the CL model to connect excess degrees
    # Ensure no node has an excess degree of 0
    non_zero_excess_degrees = [max(d, 0.1) for d in excess_degrees]
    if sum(non_zero_excess_degrees) > 0:
        cl_graph = nx.expected_degree_graph(non_zero_excess_degrees, selfloops=False)
        for u, v in cl_graph.edges():
            if not G.has_edge(u, v):
                G.add_edge(u, v)

    # Convert graph to adjacency matrix with integer values
    adj_matrix_bter = nx.to_numpy_array(G).astype(int)
    return adj_matrix_bter


def main_function(dataset_name='Facebook', eps=[0.5,1,1.5,2,2.5,3,3.5], generator=None, exp_num=10, save_csv=True):
    t_begin = time.time()
    data_path = './data/' + dataset_name + '.txt'
    mat0, mid = get_mat(data_path)

    cols = ['eps', 'exper', 'num_node_RE', 'num_edge_RE', 'tria_count_RE', 'avg_deg_RE', 'deg_var_RE', 'deg_dsb_KL',
            'diam_RE', 'SP_RE', 'dis_dsb_KL', 'GCC_RE', 'ACC_RE', 'CD_NMI', 'MOD_RE', 'Ass_RE', 'evc_MAE']

    all_data = pd.DataFrame(None, columns=cols)

    # original graph
    mat0_graph = nx.from_numpy_array(mat0, create_using=nx.Graph)

    mat0_edge = mat0_graph.number_of_edges()
    mat0_node = mat0_graph.number_of_nodes()
    print('Dataset:%s' % dataset_name)
    print('Node number:%d' % (mat0_graph.number_of_nodes()))
    print('Edge number:%d' % (mat0_graph.number_of_edges()))

    mat0_par = community.best_partition(mat0_graph)
    mat0_degree = np.sum(mat0, 0)
    mat0_avg_degree = np.mean(mat0_degree)
    mat0_degree_variance = np.var(mat0_degree, ddof=1)
    mat0_deg_dist = np.bincount(np.int64(mat0_degree))  # degree distribution
    mat0_dis_dist = distance_distribution(mat0_graph)  # distance distribution
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

        # print("Labels:", labels)
        # print("Groups:")
        # for i, group in enumerate(groups, start=1):
        #     print(f"Group {i}: {group}")

        for exper in range(exp_num):
            print('-----------epsilon=%.1f,exper=%d/%d-------------' % (epsilon, exper + 1, exp_num))
            mat2 = create_bter_graph_from_adjacency_matrix(mat0, epsilon)
            print('1')
            mat2_graph = nx.from_numpy_array(mat2, create_using=nx.Graph)
            print('2')

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
    save_name = res_path + '/' + '%s_%d_DGG.csv' % (dataset_name, exp_num)
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
    dataset_name = 'CA-HepPh'
    eps = [0.1, 0.5, 1, 2, 5, 10]
    exp_num = 10
    main_function(dataset_name=dataset_name,eps=eps,exp_num=exp_num)
