import numpy as np
from fastdtw import fastdtw
import pandas as pd
import torch


def calculate_dtw_matrix(input, k=0.005, time_delay_matrix=None, sample_rate=1):
    """Calculate DTW similarity matrix between current sequence and historical sequence

    Args:
        x: Current sequence data with shape (batch, seq_len, num_nodes)
        x_his: Historical sequence data with shape (batch, his, num_nodes)
        k: Coefficient for exponential transformation, default 0.005
        last_dtw: DTW similarity matrix from previous timestep, shape (num_nodes, num_nodes)
        m: Threshold for skipping DTW calculation based on last_dtw, default 0.1

    Returns:
        dtw_matrices: DTW similarity matrices with shape (num_nodes, num_nodes)
    """

    batch_size, seq_len, num_nodes = input.shape
    device = input.device

    x = input.mean(dim=0)  # Average over the batch_size dimension

    # Initialize output matrix
    dtw_matrices = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if time_delay_matrix[i, j] != 0:
                time_delay = int(np.ceil(abs(time_delay_matrix[i, j] / sample_rate)))
                if seq_len >= 2 * time_delay:
                    if time_delay_matrix[i, j] < 0:
                        x_seq = x[seq_len - time_delay:, i]
                        x_his_seq = x[seq_len - 2 * time_delay:seq_len - time_delay, j]
                    else:
                        x_seq = x[seq_len - time_delay:, j]
                        x_his_seq = x[seq_len - 2 * time_delay:seq_len - time_delay, i]
                    x_seq_d = calculate_first_order_difference(x_seq)
                    x_his_seq_d = calculate_first_order_difference(x_his_seq)
                    distance, _ = fastdtw(x_seq_d.cpu().numpy(), x_his_seq_d.cpu().numpy(), radius=1)
                    similarity = torch.exp(-k * torch.tensor(distance, device=device))
                    if time_delay_matrix[i, j] < 0:
                        dtw_matrices[i, j] = similarity
                    else:
                        dtw_matrices[j, i] = similarity

    return dtw_matrices


class StandardScaler:
    def __init__(self, mean, scale):
        self.mean = mean
        self.scale = scale

    def transform(self, data):
        return (data - self.mean)/ self.scale
        
    def inverse_transform(self, data):
        return data * self.scale + self.mean
    

def load_adjacency_factors(distance_path, spectrum_path):
    df = pd.read_pickle(distance_path)
    distance_factor = df[-1]
    spectrum_factor = np.load(spectrum_path)
    
    return distance_factor, spectrum_factor


def calculate_first_order_difference(x):
    """Calculate the first order difference of the input data.

    Args:
        x (torch.Tensor): Input data of shape (batch, seq_len, num_nodes)

    Returns:
        torch.Tensor: First order difference of shape (batch, seq_len-1, num_nodes)
                      or original input if seq_len is 1.
    """
    if x.shape[0] == 1:
        return x  # Return original input if seq_len is 1

    return x[1:] - x[:-1]  # Calculate the difference along the seq_len dimension



def calculate_normalized_laplacian(adj_matrix):
    """Calculate normalized Laplacian matrix
    
    Args:
        adj_matrix: Adjacency matrix as a tensor with shape (num_nodes, num_nodes)
        
    Returns:
        normalized_adj: Normalized adjacency matrix as a tensor with shape (num_nodes, num_nodes)
    """
    # Check if diagonal elements are all zero
    if torch.all(torch.diag(adj_matrix) == 0):
        adj_matrix = adj_matrix + torch.eye(adj_matrix.shape[0], device=adj_matrix.device)
    
    # Calculate degree matrix
    degree_matrix = torch.sum(adj_matrix, dim=1)
    degree_matrix = torch.diag(degree_matrix)
    
    # Calculate inverse square root of degree matrix
    degree_inv_sqrt = torch.diag(1.0 / torch.sqrt(torch.diag(degree_matrix)))
    
    # Calculate normalized adjacency matrix: A_norm = D^(-1/2)AD^(-1/2)
    normalized_adj = degree_inv_sqrt @ adj_matrix @ degree_inv_sqrt
    
    return normalized_adj



def calculate_fft_similarity_matrices(df2):
    """
    Calculate FFT similarity, dominant frequency similarity, phase difference matrices and time delays for time series data.
    
    Args:
        df2 (pd.DataFrame): Input dataframe with time series data in columns
        
    Returns:
        tuple: (similarity_matrix, dominant_freq_matrix, phase_diff_matrix, dominant_freqs, time_delay_matrix)
            - similarity_matrix: Normalized FFT amplitude correlation matrix
            - dominant_freq_matrix: Binary matrix indicating same dominant frequencies
            - phase_diff_matrix: Phase difference matrix between signals
            - dominant_freqs: List of dominant frequencies for each signal
            - time_delay_matrix: Time delay matrix between signals with same dominant frequency
    """
    # Initialize matrices
    n_columns = df2.shape[1]
    similarity_matrix = np.zeros((n_columns, n_columns))
    dominant_freq_matrix = np.zeros((n_columns, n_columns))
    phase_diff_matrix = np.zeros((n_columns, n_columns))
    time_delay_matrix = np.zeros((n_columns, n_columns))

    # Calculate FFT and dominant frequency for each column
    fft_results = []
    dominant_freqs = []
    fft_phases = []  # Store phases for each signal
    for i in range(n_columns):
        signal = df2.iloc[:2016, i] - np.mean(df2.iloc[:2016, i])
        fft = np.fft.fft(signal)
        freq = np.fft.fftfreq(len(signal))
        
        # Store FFT results (positive frequencies only)
        fft_results.append(np.abs(fft[1:len(signal)//2]))
        
        # Find dominant frequency
        pos_freq_mask = freq > 0
        pos_freqs = freq[pos_freq_mask]
        pos_fft = np.abs(fft)[pos_freq_mask]
        dominant_freq_idx = np.argmax(pos_fft)
        dominant_freq = pos_freqs[dominant_freq_idx]
        dominant_freqs.append(dominant_freq)
        
        # Store phase at dominant frequency
        fft_phases.append(np.angle(fft)[pos_freq_mask][dominant_freq_idx])

    # Calculate FFT similarity, dominant frequency similarity, phase differences and time delays
    for i in range(n_columns):
        for j in range(n_columns):
            if i == j:
                cosine_similarity = 1  # Set cosine similarity to 0 if i equals j
            else:
                # Calculate cosine similarity between FFT amplitudes
                norm_i = np.linalg.norm(fft_results[i])
                norm_j = np.linalg.norm(fft_results[j])
                if norm_i > 0 and norm_j > 0:
                    cosine_similarity = np.dot(fft_results[i], fft_results[j]) / (norm_i * norm_j)
                else:
                    cosine_similarity = 0  # Handle case where one of the norms is zero
            similarity_matrix[i,j] = cosine_similarity
            
            # Compare dominant frequencies
            if np.isclose(dominant_freqs[i], dominant_freqs[j], rtol=1e-5):
                dominant_freq_matrix[i,j] = 1
            else:
                dominant_freq_matrix[i,j] = 0

    # Normalize FFT similarity matrix to [0,1] range
    similarity_matrix = (similarity_matrix - np.min(similarity_matrix)) / (np.max(similarity_matrix) - np.min(similarity_matrix))

    # Calculate phase difference and time delay for signals with same dominant frequency
    for i in range(n_columns):
        for j in range(n_columns):
            if dominant_freq_matrix[i,j] == 1 and similarity_matrix[i,j] > 0.9:
                phase_diff = fft_phases[i] - fft_phases[j]
                # Wrap phase difference to [-π, π]
                phase_diff = np.mod(phase_diff + np.pi, 2 * np.pi) - np.pi
                phase_diff_matrix[i,j] = phase_diff
                time_delay = phase_diff / (2 * np.pi * dominant_freqs[i])
                time_delay_matrix[i,j] = 0 if np.isclose(time_delay, 0, atol=1e-1) else time_delay

    return similarity_matrix, dominant_freq_matrix, phase_diff_matrix, dominant_freqs, time_delay_matrix


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels,
                                 null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)*100


def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return mae, mape, rmse