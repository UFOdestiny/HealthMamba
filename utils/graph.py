import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg


def normalize_adj_mx(adj_mx, adj_type):
    methods = {
        "scalap": calculate_scaled_laplacian,
        "symadj": calculate_sym_adj,
    }
    if adj_type not in methods:
        return [np.eye(adj_mx.shape[0]).astype(np.float32)]
    result = methods[adj_type](adj_mx)
    if sp.issparse(result):
        return [np.asarray(result.todense(), dtype=np.float32)]
    return [np.asarray(result, dtype=np.float32)]


def calculate_scaled_laplacian(adj_mx, lambda_max=None):
    adj_mx = np.maximum(adj_mx, adj_mx.T)
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max = linalg.eigsh(L, k=1, which="LM")[0][0]
    L = sp.csr_matrix(L)
    I = sp.identity(L.shape[0], format="csr", dtype=L.dtype)
    return (2 / lambda_max * L) - I


def calculate_normalized_laplacian(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat = sp.diags(d_inv_sqrt)
    normalized = d_mat.dot(adj_mx).dot(d_mat).tocoo()
    return sp.eye(adj_mx.shape[0]) - normalized


def calculate_sym_adj(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    rowsum = np.array(adj_mx.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat = sp.diags(d_inv_sqrt)
    result = d_mat.dot(adj_mx).dot(d_mat)
    if sp.issparse(result):
        return np.asarray(result.todense(), dtype=np.float32)
    return result.astype(np.float32)
