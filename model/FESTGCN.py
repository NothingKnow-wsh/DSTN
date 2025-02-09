import torch
import torch.nn as nn
from .util import calculate_dtw_matrix, calculate_normalized_laplacian
import pandas as pd
from torch.distributions import Bernoulli
import numpy as np

class GraphConvolution(nn.Module):
    """Graph convolution layer"""
    def __init__(self, adj, num_gru_units: int, output_dim: int, bias: float = 0.0, spectrum_similarity_matrix=None):
        super(GraphConvolution, self).__init__()
        self._num_gru_units = num_gru_units
        self._output_dim = output_dim
        self._bias_init_value = bias
        self.adj = torch.tensor(adj.values, dtype=torch.float32).cuda() if isinstance(adj, pd.DataFrame) else torch.tensor(adj, dtype=torch.float32).cuda()
        self.spectrum_similarity_matrix_laplacian = calculate_normalized_laplacian(spectrum_similarity_matrix)
        self.laplacian = calculate_normalized_laplacian(self.adj)
        self.weights = nn.Parameter(
            torch.zeros(self._num_gru_units + 1, self._output_dim)
        )
        self.biases = nn.Parameter(torch.zeros(self._output_dim))
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize parameters"""
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, self._bias_init_value)
    
    def forward(self, inputs, hidden_state, dtw, multi):
        """
        Args:
            inputs: Input features (batch_size, num_nodes)
            hidden_state: Hidden state from GRU (batch_size, num_nodes, num_gru_units) 
            dtw: DTW similarity matrix (num_nodes, num_nodes)
            multi: Boolean flag for combining with Laplacian
        """
        batch_size, num_nodes = inputs.shape
        
        # Process adjacency matrix based on multi flag
        if multi:
            laplacian = torch.tensor(self.laplacian, dtype=torch.float32).cuda()
            dtw = (dtw + torch.eye(num_nodes).cuda() + laplacian + self.spectrum_similarity_matrix_laplacian) / 3
        else:
            dtw = (dtw + self.spectrum_similarity_matrix_laplacian + torch.eye(num_nodes).cuda()) / 2
        
        # Reshape inputs and hidden state
        inputs = inputs.reshape((batch_size, num_nodes, 1))
        hidden_state = hidden_state.reshape((batch_size, num_nodes, self._num_gru_units))
        
        # Concatenate inputs and hidden state
        concatenation = torch.cat((inputs, hidden_state), dim=2)
        dtw = dtw.reshape(1, num_nodes, num_nodes)

        a_times_concat = torch.matmul(dtw.to(torch.float32), concatenation.to(torch.float32))
        a_times_concat = a_times_concat.reshape((batch_size * num_nodes, self._num_gru_units + 1))
        
        # Linear transformation
        outputs = a_times_concat.to(torch.float32) @ self.weights + self.biases
        
        # Reshape output
        outputs = outputs.reshape((batch_size, num_nodes, self._output_dim))
        outputs = outputs.reshape((batch_size, num_nodes * self._output_dim))
        
        return outputs


class FGCN_cell(nn.Module):
    """FGCN cell: Combines graph convolution and GRU"""
    def __init__(self, adj, input_dim: int, hidden_dim: int, scaler, spectrum_similarity_matrix, time_delay_matrix, sample_rate):
        super(FGCN_cell, self).__init__()
        self.adj = adj
        self.num_nodes = adj.shape[0]
        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim
        self.scaler = scaler
        self.spectrum_similarity_matrix = torch.tensor(spectrum_similarity_matrix, dtype=torch.float32).cuda()
        self.time_delay_matrix = time_delay_matrix
        self.sample_rate = sample_rate
        self.dtw_cache = []
        # self.spectrum_laplacian = calculate_normalized_laplacian(self.spectrum_similarity_matrix)
        self.graph_conv1 = GraphConvolution(self.adj, self.hidden_dim, self.hidden_dim*2, bias=1.0, spectrum_similarity_matrix=self.spectrum_similarity_matrix)
        self.graph_conv2 = GraphConvolution(self.adj, self.hidden_dim, self.hidden_dim, spectrum_similarity_matrix=self.spectrum_similarity_matrix)

    def inverse_norm(self, x):
        x_cpu = x.cpu()
        result_x = self.scaler.inverse_transform(x_cpu.detach().numpy())
        return torch.FloatTensor(result_x).to(x.device)
        
    def forward(self, inputs, states, observed):
        device = inputs.device
        input_inver = self.inverse_norm(inputs)
        batch_size = inputs.shape[0]
        times = inputs.shape[1]
        
        # 初始化在GPU上
        gcn_1 = torch.zeros(batch_size, self.input_dim * self.hidden_dim*2, device=device)
        gcn_2 = torch.zeros(batch_size, self.input_dim * self.hidden_dim, device=device)

        if observed == False:
            dtw = self.dtw_cache[-1]
        else:
            if times == 1:
                dtw = torch.zeros((self.num_nodes, self.num_nodes), dtype=torch.float32, device=device)
            else:
                dtw = calculate_dtw_matrix(
                            input_inver,
                            k=0.01,
                            time_delay_matrix=self.time_delay_matrix,
                            sample_rate= self.sample_rate).cuda()

        tm = torch.full((self.num_nodes, self.num_nodes), times).cuda()
        td = torch.from_numpy(np.ceil(np.abs(self.time_delay_matrix) / self.sample_rate).astype(int)).cuda()
        time_indicator = tm - td

        for t in range(times):
            if t == times - 1:
                multi = True
                if observed == True:
                    self.dtw_cache.append(dtw)
            else:
                multi = False

            dtwt = torch.where((dtw != 0) & ((t + 1) > time_indicator), dtw, 0)
            # dtwt = dtw

            input = inputs[:, t, :]
            hidden_state = states[t].to(device)

            # 所有运算都在GPU上进行
            gcn_1 = gcn_1 + self.graph_conv1(input, hidden_state, dtwt, multi)
            concatenation = torch.sigmoid(gcn_1)
            r, u = torch.chunk(concatenation, chunks=2, dim=1)
            gcn_2 = gcn_2 + self.graph_conv2(input, r * hidden_state, dtwt, multi)
            c = torch.tanh(gcn_2)
            h_new = u * hidden_state + (1.0 - u) * c
            # hidden_state = h_new

        return h_new, h_new

class FESTGCN(nn.Module):
    """Complete FGCN model"""
    def __init__(self, adj, hidden_dim, scaler, spectrum_similarity_matrix, time_delay_matrix, epsilon, sample_rate):
        super(FESTGCN, self).__init__()
        self.adj = adj
        self.hidden_dim = hidden_dim
        self.scaler = scaler
        self.spectrum_similarity_matrix = np.where(spectrum_similarity_matrix < 0.7, 0, spectrum_similarity_matrix)
        # self.spectrum_similarity_matrix = spectrum_similarity_matrix
        self.time_delay_matrix = time_delay_matrix
        self.num_nodes = adj.shape[0]
        self.epsilon = epsilon
        self.sample_rate = sample_rate

        self.fgcn_cell = FGCN_cell(self.adj, self.num_nodes, self.hidden_dim, self.scaler, self.spectrum_similarity_matrix, self.time_delay_matrix, self.sample_rate)
        self.fc_controller = nn.Linear(self.num_nodes*self.hidden_dim+1, self.num_nodes)
        self.fc_baseline = nn.Linear(self.num_nodes * self.hidden_dim + 1, self.num_nodes)
        self.linear = nn.Linear(self.hidden_dim, 1)

    def BaselineNetwork(self,x):
        b = self.fc_baseline(x.detach())
        return b
    
    def Controller(self, x,epsilon=0.0):
        probs = torch.sigmoid(self.fc_controller(x))
        probs = (1-epsilon)*probs + epsilon*torch.FloatTensor([0.05]).cuda(0)  # Explore/exploit
        m = Bernoulli(probs=probs)
        action = m.sample() # sample an action
        log_pi = m.log_prob(action) # compute log probability of sampled action
        return action.squeeze(0).cuda(0), log_pi.squeeze(0).cuda(0), -torch.log(probs).squeeze(0).cuda(0)
    
    def forward(self, inputs):
        # inputs: (batch_size, seq_len, num_nodes)
        batch_size, seq_len, num_nodes = inputs.shape
        device = inputs.device

        encoder_state = torch.zeros(batch_size, num_nodes * self.hidden_dim, device=device)
        encoder_states = []

        halt_points = torch.zeros((batch_size, self.num_nodes)).cuda(0)
        predictions = torch.zeros((batch_size, self.num_nodes)).cuda(0)
        actions=[]
        log_pi=[]
        halt_probs=[]
        baselines=[]

        for i in range(seq_len - 9):
            encoder_inputs = inputs[:,:i+1,:]
            encoder_states.append(encoder_state)
            output, encoder_state = self.fgcn_cell(encoder_inputs, encoder_states, observed = True)

            output = output.reshape((batch_size, num_nodes, self.hidden_dim))
            prediction = self.linear(output)

            decoder_state=encoder_state
            decoder_states=encoder_states+[decoder_state]
            decoder_input=inputs[:,i,:].unsqueeze(1)
            decoder_inputs=torch.cat((encoder_inputs,decoder_input),1)

            for j in range (seq_len-i-1):
                # print('current prediction time:', j)
                pred,decoder_state = self.fgcn_cell(decoder_inputs, decoder_states, observed = False)
                pred = pred.reshape((batch_size, num_nodes, self.hidden_dim))
                prediction=self.linear(pred)
                decoder_input=prediction.transpose(1,2)
                
                decoder_inputs=torch.cat((decoder_inputs,decoder_input),1)#torch(batch_size,i+j+2,num_nodes)
                decoder_states=decoder_states+[decoder_state]#torch(i+j+2,batch_size,num_nodes*hidden_dim)
            
            outputs_final = prediction.view(batch_size, self.num_nodes)

            # --------Contorller

            decoder_state = decoder_state.reshape(batch_size, num_nodes, self.hidden_dim)
            decoder_state = decoder_state.reshape(batch_size, num_nodes * self.hidden_dim)

            c_in = torch.cat((decoder_state.unsqueeze(0),torch.tensor([i], dtype=torch.float, requires_grad=False).view(1, 1,1).repeat(1,batch_size, 1).cuda(0)),dim=2)
            a_t, p_t, w_t = self.Controller(c_in,self.epsilon)
            b_t = self.BaselineNetwork(c_in)
            
            predictions = torch.where((a_t == 1) & (predictions == 0), outputs_final, predictions)
            halt_points = torch.where((halt_points == 0) & (a_t == 1), torch.tensor([i+1],dtype=torch.float, requires_grad=False).view(1, 1).repeat(batch_size,self.num_nodes).cuda(0), halt_points)
           
            baselines.append(b_t)
            actions.append(a_t)
            log_pi.append(p_t)
            halt_probs.append(w_t)
            if (halt_points == 0).sum() == 0:  # If no negative values, every class has been halted
                break

        self.fgcn_cell.dtw_cache = []
        predictions = torch.where(predictions == 0, outputs_final, predictions)
        halt_points = torch.where(halt_points == 0, torch.tensor([i + 1], dtype=torch.float, requires_grad=False).view(1, 1).repeat(batch_size, self.num_nodes).cuda(0), halt_points)

       
        baselines = torch.stack(baselines).squeeze(1)
        log_pi = torch.stack(log_pi)
        wait_penalty= torch.stack(halt_probs).sum(0).sum(1).mean() 
        grad_mask = torch.zeros_like(torch.stack(actions).transpose(0, 1))

        for b in range(batch_size):
            for n in range(self.num_nodes):
                grad_mask[b, :(halt_points[b, n]).long(),n] = 1
        
        
        return predictions, (halt_points).mean()/seq_len,grad_mask.transpose(1,0),baselines.float(),log_pi,wait_penalty
