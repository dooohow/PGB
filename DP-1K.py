import numpy as np
import networkx as nx
from utils import *
import pandas as pd
from sklearn import metrics
import os
import tracemalloc


def laplace_noise(scale, size):
    """
    Generate Laplace noise.

    :param scale: Scale of the Laplace distribution (1/epsilon).
    :param size: Number of noise values to generate.
    :return: NumPy array of Laplace noise.
    """
    return np.random.laplace(0, scale, size)


def apply_laplace_noise(degree_distribution, epsilon):
    """
    Apply Laplace noise to the degree distribution for differential privacy.

    :param degree_distribution: List where index represents the degree and value represents the number of nodes with that degree.
    :param epsilon: Privacy parameter.
    :return: Noised degree distribution.
    """
    # Calculate the scale of the Laplace noise
    scale = 4.0 / epsilon  # 1K-distribution global sensitivity is 4

    # Generate Laplace noise
    noise = laplace_noise(scale, len(degree_distribution))

    # Apply noise to the degree distribution
    noised_distribution = degree_distribution + noise

    # Ensure all values are non-negative integers
    noised_distribution = np.maximum(0, np.round(noised_distribution)).astype(int)

    # Truncate too large values
    # max_allowed_degree = int(np.max(degree_distribution) * 1.5)  # Limit to 150% of the max original degree
    # noised_distribution = np.clip(noised_distribution, 0, max_allowed_degree)
    # original_total_degree = np.sum(np.arange(len(degree_distribution)) * degree_distribution)
    # max_total_degree = int(original_total_degree * 1.5)
    # total_degree = np.sum(np.arange(len(noised_distribution)) * noised_distribution)
    #
    # if total_degree > max_total_degree:
    #     scaling_factor = max_total_degree / total_degree
    #     noised_distribution = np.floor(noised_distribution * scaling_factor).astype(int)

    # Ensure the sum of degrees is even
    degree_sum = sum(degree * count for degree, count in enumerate(noised_distribution))
    if degree_sum % 2 != 0:
        for i in range(len(noised_distribution)):
            if noised_distribution[i] > 0:
                noised_distribution[i] += 1  # Increment by 1 instead of decrement to ensure it's even
                break

    return noised_distribution


def generate_graph_from_noised_distribution(noised_distribution):
    """
    Generate a graph from the noised degree distribution.

    :param noised_distribution: Noised degree distribution.
    :return: NetworkX graph.
    """
    # Create a degree sequence from the noised degree distribution
    degree_sequence = []
    for degree, count in enumerate(noised_distribution):
        degree_sequence.extend([degree] * count)
    print(len(degree_sequence))

    # Ensure the sum of degrees is even
    if sum(degree_sequence) % 2 != 0:
        degree_sequence[0] += 1

    # Use the degree sequence to generate a graph
    G = nx.havel_hakimi_graph(degree_sequence)

    return G


def get_adjacency_matrix(G):
    """
    Get the adjacency matrix of a graph.

    :param G: NetworkX graph.
    :return: Adjacency matrix as a NumPy array.
    """
    adjacency_matrix = nx.adjacency_matrix(G).todense()
    return np.array(adjacency_matrix)


def calculate_degree_distribution(adj_matrix):
    """
    Calculate the degree distribution from an adjacency matrix.

    :param adj_matrix: Adjacency matrix.
    :return: Degree distribution as a list.
    """
    G = nx.from_numpy_array(adj_matrix)
    degrees = [G.degree(n) for n in G.nodes()]
    max_degree = max(degrees)
    degree_distribution = [degrees.count(i) for i in range(max_degree + 1)]
    return degree_distribution



def generate_private_graph(adj_matrix, epsilon):
    """
    Generate a differentially private graph from the given adjacency matrix and epsilon.

    :param adj_matrix: Adjacency matrix of the original graph.
    :param epsilon: Privacy parameter.
    :return: Adjacency matrix of the generated private graph.
    """
    # Calculate the degree distribution from the adjacency matrix
    degree_distribution = calculate_degree_distribution(adj_matrix)

    # Apply Laplace noise to the degree distribution
    noised_distribution = apply_laplace_noise(np.array(degree_distribution), epsilon)

    # Generate graph from noised degree distribution
    G = generate_graph_from_noised_distribution(noised_distribution)

    # Get the adjacency matrix of the generated graph
    adj_matrix_private = get_adjacency_matrix(G)

    return adj_matrix_private


def main_function(dataset_name='Facebook', eps=[0.5,1,1.5,2,2.5,3,3.5], generator=None, exp_num=10, save_csv=True):
    t_begin = time.time()
    data_path = './data/' + dataset_name + '.txt'
    mat0, mid = get_mat(data_path)

    cols = ['eps', 'exper', 'num_node_RE','num_edge_RE', 'tria_count_RE','avg_deg_RE','deg_var_RE','deg_dsb_KL','diam_RE','SP_RE','dis_dsb_KL','GCC_RE','ACC_RE', 'CD_NMI','MOD_RE','Ass_RE', 'evc_MAE']

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
            mat2 = generate_private_graph(mat0, epsilon)
            print('1')
            mat2_graph = nx.from_numpy_array(mat2, create_using=nx.Graph)
            print('2')



            mat2_edge = mat2_graph.number_of_edges()
            mat2_node = mat2_graph.number_of_nodes()


            mat2 = pad_adj_matrix(mat2, mat0_graph.number_of_nodes())
            mat2_graph = nx.from_numpy_array(mat2, create_using=nx.Graph)
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
            mod_rel = cal_rel(mat0_mod, mat2_mod)
            diam_rel = cal_rel(mat0_diam, mat2_diam)
            ACC_RE = cal_rel(mat0_acc, mat2_acc)
            Ass_RE = cal_rel(mat0_ass, mat2_ass)

            dis_dsb_KL = cal_kl(mat0_dis_dist, mat2_dis_dist)
            deg_kl = cal_kl(mat0_deg_dist, mat2_deg_dist)
            labels_true = list(mat0_par.values())
            labels_pred = list(mat2_par.values())
            min_length = min(len(labels_true), len(labels_pred))
            labels_true = labels_true[:min_length]
            labels_pred = labels_pred[:min_length]
            nmi = metrics.normalized_mutual_info_score(labels_true, labels_pred)
            evc_MAE = cal_MAE(mat0_evc_val, mat2_evc_val, k=evc_kn)



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
                % (mat2_node, mat2_edge, num_node_RE,num_edge_RE,tria_count_RE,avg_deg_RE,deg_var_RE,deg_kl,diam_rel,SP_RE,dis_dsb_KL,cc_rel,ACC_RE,nmi,mod_rel,Ass_RE,evc_MAE))

            data_col = [epsilon,exper,num_node_RE,num_edge_RE,tria_count_RE,avg_deg_RE,deg_var_RE,deg_kl,diam_rel,SP_RE,dis_dsb_KL,cc_rel,ACC_RE,nmi,mod_rel,Ass_RE,evc_MAE]
            col_len = len(data_col)
            data_col = np.array(data_col).reshape(1,col_len)
            data1 = pd.DataFrame(data_col,columns=cols)
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
    save_name = res_path + '/' + '%s_%d_DP1k.csv' %(dataset_name, exp_num)
    if not os.path.exists(res_path):
        os.mkdir(res_path)

    if save_csv == True:
        all_data.to_csv(save_name,index=False,sep=',')

    print('-----------------------------')

    print('dataset:', dataset_name)

    print('eps=', eps)
    print('all_num_node_RE=',all_num_node_RE)
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
