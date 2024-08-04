from utils import *
from scipy.optimize import minimize
from sklearn import metrics
import pandas as pd
import os
def triangle_count(G):
    return sum(nx.triangles(G).values()) // 3

def laplace(value, sensitivity, epsilon):
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale)
    return value + noise


def cal_local_sensitivity(G):
    max_sensitivity = 0
    nodes = list(G.nodes())
    old_triangles = triangle_count(G)
    total_nodes = len(nodes)

    for i, node in enumerate(nodes, 1):
        G_prime = G.copy()
        G_prime.remove_node(node)
        new_triangles = triangle_count(G_prime)

        sensitivity = abs(old_triangles - new_triangles)

        if sensitivity > max_sensitivity:
            max_sensitivity = sensitivity

        print(f"Processing node {i}/{total_nodes}")

    return max_sensitivity



def sensitivity(beta, local_sensitivity):
    return local_sensitivity * np.exp(-beta * 1)

def sorted_degree_vector(G):
    """
    Calculate the degree vector of all nodes in the graph and sort the vector.

    Parameters:
    - G (networkx.Graph): The input NetworkX graph.

    Returns:
    - np.array: The sorted degree vector array.
    """

    degrees = np.array([d for n, d in G.degree()])

    sorted_degrees = np.sort(degrees)

    return sorted_degrees
def compute_s_bar(s_tilde):
    """
    Parameters:
    - s_tilde (np.array): The given s_tilde array.

    Returns:
    - np.array: The computed s_bar array with integer results.
    """

    n = len(s_tilde)
    J = []
    s_bar = np.zeros(n)

    def top(stack):
        return stack[-1] if stack else None

    def average_subarray(arr, start, end):
        if start > end:
            return float('inf')
        return np.mean(arr[start:end + 1])

    J.append(n - 1)

    for k in range(n - 1, -1, -1):
        j_star = k
        j = top(J)
        while J and average_subarray(s_tilde, j_star + 1, j) <= average_subarray(s_tilde, k, j_star):
            j_star = j
            J.pop()
            j = top(J)
        J.append(j_star)

    b = 0
    while J:
        j_star = J.pop()
        for k in range(b, j_star + 1):
            s_bar[k] = average_subarray(s_tilde, b, j_star)
        b = j_star + 1

    return np.round(s_bar).astype(int)


def DP_SKG(G, epsilon, local_sensitivity):
    epsilon1 = epsilon/2
    epsilon2 = epsilon/2
    sorted_degrees = sorted_degree_vector(G)
    noisy_array = laplace_mechanism(sorted_degrees, epsilon1)
    noisy_degree = compute_s_bar(noisy_array)
    E_tilde = 0.5 * np.sum(noisy_degree)
    print('fact',E_tilde)
    H_tilde = 0.5 * np.sum(noisy_degree * (noisy_degree - 1))
    print(H_tilde)
    T_tilde = (1 / 6) * np.sum(sorted_degrees * (noisy_degree - 1) * (noisy_degree - 2))
    print(T_tilde)
    delta = 0.01
    beta = epsilon2 / (2 * math.log(2 / delta))
    beta = math.floor(beta * 10) / 10
    ss_beta_f = sensitivity(beta, local_sensitivity)
    print(ss_beta_f)
    delta_tilde = laplace(triangle_count(G), 2 * ss_beta_f, epsilon2)
    while delta_tilde <= 0:
        delta_tilde = laplace(triangle_count(G), 2 * ss_beta_f, epsilon2)
    print(triangle_count(G))
    print(delta_tilde)
    n = G.number_of_nodes()
    r = np.floor(np.log2(n))
    print(r)

    def E(a, b, c, k):
        return 0.5 * (np.power(a + 2 * b + c, k) - np.power(a + c, k))

    def H(a, b, c, k):
        return 0.5 * (np.power((a + b) ** 2 + (b + c) ** 2, k) - 2 * np.power(a * (a + b) + c * (c + b), k)
                      - np.power(a ** 2 + 2 * b ** 2 + c ** 2, k) + 2 * np.power(a ** 2 + c ** 2, k))

    def delta(a, b, c, k):
        return (1 / 6) * (np.power(a ** 3 + 3 * b ** 2 * (a + c) + c ** 3, k) - 3 * np.power(
            a * (a ** 2 + b ** 2) + c * (b ** 2 + c ** 2), k) +
                          2 * np.power(a ** 3 + c ** 3, k))

    def T(a, b, c, k):
        return (1 / 6) * (np.power((a + b) ** 3 + (b + c) ** 3, k) - 3 * np.power(a * (a + b) ** 2 + c * (b + c) ** 2, k)
                          - 3 * np.power(a ** 3 + c ** 3 + b * (a ** 2 + c ** 2) + b ** 2 * (a + c) + 2 * b ** 3, k)
                          + 2 * np.power(a ** 3 + c ** 3 + 2 * b ** 3, k) + 5 * np.power(a ** 3 + c ** 3 + b ** 2 * (a + c), k)
                          + 4 * np.power(a ** 3 + c ** 3 + b * (a ** 2 + c ** 2), k) - 6 * np.power(a ** 3 + c ** 3, k))


    def dist_sq_log(x, y):
        return (x-y) ** 2

    def norm_F2(F, E):
        return F ** 2


    def objective(params, k, E_tilde, H_tilde, delta_tilde, T_tilde):
        a, b, c = params
        total_sum = 0
        total_sum += dist_sq_log(E(a, b, c, k), E_tilde) / norm_F2(E_tilde, E_tilde)
        total_sum += dist_sq_log(H(a, b, c, k), H_tilde) / norm_F2(H_tilde, H_tilde)
        total_sum += dist_sq_log(T(a, b, c, k), T_tilde) / norm_F2(T_tilde, T_tilde)
        total_sum += dist_sq_log(delta(a, b, c, k), delta_tilde) / norm_F2(delta_tilde, delta_tilde)
        return total_sum


    constraints = [{'type': 'ineq', 'fun': lambda x: x[0]},  # a >= 0
                   {'type': 'ineq', 'fun': lambda x: 1 - x[0]},  # a <= 1
                   {'type': 'ineq', 'fun': lambda x: x[1]},  # b >= 0
                   {'type': 'ineq', 'fun': lambda x: 1 - x[1]},  # b <= 1
                   {'type': 'ineq', 'fun': lambda x: x[2]},  # c >= 0
                   {'type': 'ineq', 'fun': lambda x: 1 - x[2]},  # c <= 1
                   {'type': 'ineq', 'fun': lambda x: x[0] - x[2]}]  # a > c


    k = int(r)
    results = []
    for _ in range(1000):
        initial_guess = np.random.rand(3)
        result = minimize(objective, initial_guess, args=(k, E_tilde, H_tilde, delta_tilde, T_tilde),
                          constraints=constraints, method='COBYLA', options={'disp': False})
        results.append(result)


    best_result = min(results, key=lambda res: res.fun)
    a = best_result.x[0]
    b = best_result.x[1]
    c = best_result.x[2]
    matrix0 = np.array([[a, b], [b, c]])
    matrix0_kronecker_k = matrix0
    for _ in range(k-1):
        matrix0_kronecker_k = np.kron(matrix0_kronecker_k, matrix0)
    sys_graph = nx.Graph()
    num_nodes = matrix0_kronecker_k.shape[0]
    sys_graph.add_nodes_from(range(num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            probability = matrix0_kronecker_k[i, j]
            if np.random.rand() < probability:
                sys_graph.add_edge(i, j)
    print('最优参数:', best_result.x)
    undirect_G = sys_graph.to_undirected()
    return undirect_G

def main_function(dataset_name='Facebook', eps=[0.5,1,1.5,2,2.5,3,3.5], generator=None, exp_num=10, save_csv=True):
    t_begin = time.time()
    data_path = './data/' + dataset_name + '.txt'
    mat0, mid = get_mat(data_path)

    cols = ['eps', 'exper', 'num_node_RE','num_edge_RE', 'tria_count_RE','avg_deg_RE','deg_var_RE','deg_dsb_KL','diam_RE','SP_RE','dis_dsb_KL','GCC_RE','ACC_RE', 'CD_NMI','MOD_RE','Ass_RE', 'evc_MAE']

    all_data = pd.DataFrame(None, columns=cols)

    # original graph
    mat0_graph = nx.from_numpy_array(mat0, create_using=nx.Graph)
    local_sen = cal_local_sensitivity(mat0_graph)
    print('local_sensitivity:', local_sen)

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

        # print("Labels:", labels)
        # print("Groups:")
        # for i, group in enumerate(groups, start=1):
        #     print(f"Group {i}: {group}")

        for exper in range(exp_num):
            print('-----------epsilon=%.1f,exper=%d/%d-------------' % (epsilon, exper + 1, exp_num))
            mat2_graph = DP_SKG(mat0_graph, epsilon, local_sen)
            print('1')
            adj_matrix = nx.adjacency_matrix(mat2_graph)

            # 将稀疏矩阵转换为稠密矩阵，并转换为整数类型
            dense_matrix = adj_matrix.todense()
            mat2 = np.array(dense_matrix, dtype=int)
            mat2 = pad_adj_matrix(mat2, mat0_graph.number_of_nodes())
            print('2')


            mat2_edge = mat2_graph.number_of_edges()
            mat2_node = mat2_graph.number_of_nodes()
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
    save_name = res_path + '/' + '%s_%d_SKG.csv' %(dataset_name, exp_num)
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



