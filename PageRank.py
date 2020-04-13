import numpy as np
import csv


def page_rank_score(A, alpha=0.7):
    """
    Compute the page rank score from the matrix A and the teleportation's parameter alpha.
    :param A: The adjacency matrix of the graph
    :param alpha: a parameter of teleportation
    :return: A vector who contains the node's rank for the graph
    """
    n = len(A)
    H = probabilistic_matrix(A)
    G = google_matrix(H, alpha, n)
    vector_result = power_method(G, n)
    return vector_result


def probabilistic_matrix(A):
    """
    This function compute the matrix of probability to go edge from another edge.
    :param A: The adjacency matrix of a directed graph
    :return: The matrix of probabilities
    """
    for i in range(len(A)):
        somme = sum(A[i, :])
        if somme != 0:
            A[i, :] = A[i, :] / somme
    return A


def dangling_node_vector(H):
    """
    Compute a dangling node vector.If a row of the probabilistic matrix is 0, so the same row of this vector is 1.
    Otherwise = 0.
    :param H: The probabilistic matrix of the starting matrix.
    :return: return the dangling node vector of H.
    """
    a = np.zeros((len(H), 1))
    for i in range(len(H)):
        if sum(H[i, :]) == 0:
            a[i, 0] = 1
    return a


def vector_e(n):
    """
    The vector e is a vector for what each element is 1.
    :param n: the len of the starting matrix.
    :return: A vector for what each element is 1.
    """
    return np.ones((n, 1))


def stochastic_matrix(H, n):
    """
    Compute the stochastic matrix from the probabilistic matrix.
    :param H: The probabilistic matrix.
    :param n: The len of the matrix.
    :return: The stochastic matrix from H.
    """
    a = dangling_node_vector(H)
    e = vector_e(n)
    S = H + a * ((1/n) * e.transpose())
    return S


def teleportation_matrix(n):
    """
    Compute the teleportation matrix.
    :param n: The len of matrix
    :return: The teleportation matrix
    """
    e = vector_e(n)
    E = (1/n) * (e * e.transpose())
    return E


def google_matrix(H, alpha, n):
    """
    Compute the Google's matrix following the formula :
    G = αH + (αa + (1 - α)e)1/n(e^T)
    :param H: The probabilistic matrix
    :param alpha: The parameter of teleportation
    :param n: The len of the matrix
    :return: The Google's matrix
    """
    S = stochastic_matrix(H, n)
    E = teleportation_matrix(n)
    G = alpha * S + (1 - alpha) * E
    return G


def power_method(G, n):
    """
    Compute the eigen vector of the Google's matrix with the power method
    :param G: The Google's matrix
    :param n: The len of the matrix
    :return: The eigen vector of the Google's matrix
    """
    x = (np.ones((n, 1)))
    x_T = x.transpose()
    for i in range(150):
        x_T = x_T.dot(G)
    return (1 / n) * x_T


if __name__ == '__main__':
    A = np.array([[0, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1, 1], [0, 0, 0, 1, 0, 1], [0, 0, 0, 1, 0, 0]], float)    # A est une matrice de tests
    with open('matrix.csv') as file:
        matrix = np.array(list(csv.reader(file, dialect="excel", delimiter=',', lineterminator='\n')), float)
    print(matrix)
    print(page_rank_score(matrix))
