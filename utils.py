import networkx as nx
import numpy as np

from sklearn.cluster import KMeans
import random
import math





def get_mat(data_path):
    # data_path = './data/' + dataset_name + '.txt'
    data = np.loadtxt(data_path)

    # initial statistics
    dat = (np.append(data[:, 0], data[:, 1])).astype(int)
    dat_c = np.bincount(dat)

    d = {}
    node = 0
    mid = []
    for i in range(len(dat_c)):
        if dat_c[i] > 0:
            d[i] = node
            mid.append(i)
            node = node + 1
    mid = np.array(mid, dtype=np.int32)

    # initial statistics
    Edge_num = data.shape[0]
    c = len(d)

    # genarated adjancent matrix
    mat0 = np.zeros([c, c], dtype=np.uint8)
    for i in range(Edge_num):
        mat0[d[int(data[i, 0])], d[int(data[i, 1])]] = 1

    # transfer direct to undirect
    mat0 = mat0 + np.transpose(mat0)
    mat0 = np.triu(mat0, 1)
    mat0 = mat0 + np.transpose(mat0)
    mat0[mat0 > 0] = 1
    return mat0, mid


def pad_adj_matrix(adj_matrix, n):
    """
    Pad the given adjacency matrix to be of size n x n.

    Parameters:
    adj_matrix (numpy.ndarray): The input adjacency matrix.
    n (int): The target size of the padded adjacency matrix.

    Returns:
    numpy.ndarray: The padded adjacency matrix of size n x n.
    """
    current_size = adj_matrix.shape[0]

    # If the current size is already equal to or larger than n, no padding is needed
    if current_size >= n:
        return adj_matrix

    # Create a new n x n matrix initialized with zeros
    padded_matrix = np.zeros((n, n), dtype=adj_matrix.dtype)

    # Copy the original matrix into the top-left corner of the new matrix
    padded_matrix[:current_size, :current_size] = adj_matrix

    return padded_matrix

def cal_diam(mat):
    """
    Calculate the diameter of a graph.

    Parameters:
    - mat: Adjacency matrix.

    Returns:
    - max_diam: An integer representing the diameter.
    """

    mat_graph = nx.from_numpy_array(mat,create_using=nx.Graph)
    max_diam = 0
    for com in nx.connected_components(mat_graph):
        com_list = list(com)
        mat_sub = mat[np.ix_(com_list,com_list)]
        sub_g = nx.from_numpy_array(mat_sub,create_using=nx.Graph)
        diam = nx.diameter(sub_g)
        if diam > max_diam:
            max_diam = diam
    return max_diam

def calculate_inter_group_edges(G, labels):
    """
    Calculate the number of edges from each node to nodes not in its own group.

    Parameters:
    - G: A networkx graph object.
    - labels: An array representing the clustering labels of each node.

    Returns:
    - inter_group_degrees: A dictionary where the key is the node and the value is the number of edges from that node to nodes in other groups.
    """

    inter_group_degrees = {}
    for node in G.nodes():
        degree = 0
        for neighbor in G.neighbors(node):
            # 如果邻居节点的标签与当前节点不同，计数加一
            if labels[node] != labels[neighbor]:
                degree += 1
        inter_group_degrees[node] = degree
    return inter_group_degrees


def average_shortest_path_length_custom(G):
    """
    Calculate the average shortest path length between all pairs of nodes in the graph, handling both connected and disconnected graphs.

    Parameters:
    - G: A networkx graph.

    Returns:
    - average_shortest_path_length: The average shortest path length.
    """

    if nx.is_connected(G):
        return nx.average_shortest_path_length(G)
    else:
        subgraphs = (G.subgraph(c).copy() for c in nx.connected_components(G))
        total_length = 0
        total_pairs = 0

        for subgraph in subgraphs:
            if subgraph.number_of_nodes() > 1:  # 忽略只有一个节点的子图
                length = nx.average_shortest_path_length(subgraph)
                num_nodes = subgraph.number_of_nodes()
                num_pairs = num_nodes * (num_nodes - 1) / 2
                total_length += length * num_pairs
                total_pairs += num_pairs

        return total_length / total_pairs if total_pairs > 0 else float('inf')


def distance_distribution(G):
    """
    Calculate the distance distribution of the graph, which is the frequency distribution of the shortest path lengths between all pairs of nodes.

    Parameters:
    - G: A networkx graph.

    Returns:
    - dist_dist: The frequency distribution of the shortest path lengths.
    """

    all_shortest_paths = []
    for source in G.nodes:
        lengths = nx.single_source_shortest_path_length(G, source)
        all_shortest_paths.extend(lengths.values())

    dist_dist = np.bincount(np.int64(all_shortest_paths))
    return dist_dist



def laplace_mechanism(data, epsilon):
    """
    Apply the Laplace mechanism to the data to achieve differential privacy.

    Parameters:
    - data: Original data, a NumPy array.
    - epsilon: Privacy budget.

    Returns:
    - privacy_data: Privacy-protected data with added Laplace noise.
    """

    scale = 1.0 / epsilon
    noise = np.random.laplace(0, scale, size=data.shape)
    privacy_data = data + noise
    privacy_data = np.maximum(privacy_data, 0)

    return privacy_data





def top_m_filter(adjacency_matrix, epsilon):
    """
    
    :param adjacency_matrix: original adjacency matrix
    :param epsilon: 
    :return: matirx
    """
    epsilon1 = epsilon/2
    epsilon2 = epsilon/2
    n = adjacency_matrix.shape[0]
    m = np.sum(adjacency_matrix) // 2  

    sanitized_matrix = np.zeros((n, n))


    m_tilde = laplace_mechanism(m, epsilon2)
    epsilon_t = np.log(n * (n - 1) / (2 * m_tilde) - 1)

    if epsilon1 < epsilon_t:
        theta = (1 / (2 * epsilon1)) * np.log(n * (n - 1) / (2 * m_tilde) - 1)
    else:
        theta = (1 / epsilon1) * np.log(n * (n - 1) / (4 * m_tilde) + 0.5 * (np.exp(epsilon1) - 1))


    n1 = 0
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency_matrix[i, j] == 1:
                A_ij = adjacency_matrix[i, j]
                A_tilde_ij = laplace_mechanism(A_ij, epsilon1)
                if A_tilde_ij > theta:
                    sanitized_matrix[i, j] = 1
                    sanitized_matrix[j, i] = 1
                    n1 += 1

    n0 = m_tilde - n1
    while n0 > 0:
        i, j = np.random.choice(n, 2, replace=False)
        if sanitized_matrix[i, j] == 0:
            sanitized_matrix[i, j] = 1
            sanitized_matrix[j, i] = 1
            n0 -= 1

    return sanitized_matrix


def cal_overlap(la,lb,k):
    la = la[:k]
    lb = lb[:k]
    la_s = set(la)
    lb_s = set(lb)
    num = len(la_s & lb_s)
    rate = num / k
    return rate
def cal_rel(A,B):
    eps = 0.000000000000001
    A = np.float64(A)
    B = np.float64(B)
    #eps = np.float64(eps)
    res = abs((A-B)/(A+eps))
    return res

def cal_MSE(A,B):
    res = np.mean((A-B)**2)
    return res

def cal_kl(A,B):
    p = A / sum(A)
    q = B / sum(B)
    if A.shape[0] > B.shape[0]:
        q = np.pad(q,(0,p.shape[0]-q.shape[0]),'constant',constant_values=(0,0))
    elif A.shape[0] < B.shape[0]:
        p = np.pad(p,(0,q.shape[0]-p.shape[0]),'constant',constant_values=(0,0))
    kl = p * np.log((p+np.finfo(np.float64).eps)/(q+np.finfo(np.float64).eps))
    kl = np.sum(kl)
    return kl
def cal_MAE(A,B,k=None):
    if k== None:
        res = np.mean(abs(A-B))
    else:
        a = np.array(A[:k])
        b = np.array(B[:k])
        res = np.mean(abs(a-b))
    return res
