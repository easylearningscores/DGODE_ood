# Time-evolving Prompt Learning with Graph ODE
import torch
torch.cuda.empty_cache()
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import numpy as np
import torch
from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchdiffeq as ode
import numpy as np
import torch
# device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LayerParams:
    def __init__(self, rnn_network: torch.nn.Module, layer_type: str):
        self._rnn_network = rnn_network
        self._params_dict = {}
        self._biases_dict = {}
        self._type = layer_type

    def get_weights(self, shape):
        if shape not in self._params_dict:
            nn_param = torch.nn.Parameter(torch.empty(*shape, device=device))
            torch.nn.init.xavier_normal_(nn_param)
            self._params_dict[shape] = nn_param
            self._rnn_network.register_parameter('{}_weight_{}'.format(self._type, str(shape)),
                                                 nn_param)
        return self._params_dict[shape]

    def get_biases(self, length, bias_start=0.0):
        if length not in self._biases_dict:
            biases = torch.nn.Parameter(torch.empty(length, device=device))
            torch.nn.init.constant_(biases, bias_start)
            self._biases_dict[length] = biases
            self._rnn_network.register_parameter('{}_biases_{}'.format(self._type, str(length)),
                                                 biases)

        return self._biases_dict[length]


class DCGRUCell(torch.nn.Module):
    def __init__(self, num_units, max_diffusion_step, num_nodes, nonlinearity='tanh',
                 filter_type="laplacian", use_gc_for_ru=True):
        """

        :param num_units:
        :param adj_mx:
        :param max_diffusion_step:
        :param num_nodes:
        :param nonlinearity:
        :param filter_type: "laplacian", "random_walk", "dual_random_walk".
        :param use_gc_for_ru: whether to use Graph convolution to calculate the reset and update gates.
        """

        super().__init__()
        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu
        # support other nonlinearities up here?
        self._num_nodes = num_nodes
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._supports = []
        self._use_gc_for_ru = use_gc_for_ru
        
        '''
        Option:
        if filter_type == "laplacian":
            supports.append(utils.calculate_scaled_laplacian(adj_mx, lambda_max=None))
        elif filter_type == "random_walk":
            supports.append(utils.calculate_random_walk_matrix(adj_mx).T)
        elif filter_type == "dual_random_walk":
            supports.append(utils.calculate_random_walk_matrix(adj_mx).T)
            supports.append(utils.calculate_random_walk_matrix(adj_mx.T).T)
        else:
            supports.append(utils.calculate_scaled_laplacian(adj_mx))
        for support in supports:
            self._supports.append(self._build_sparse_matrix(support))
        '''

        self._fc_params = LayerParams(self, 'fc')
        self._gconv_params = LayerParams(self, 'gconv')

    @staticmethod
    def _build_sparse_matrix(L):
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        # this is to ensure row-major ordering to equal torch.sparse.sparse_reorder(L)
        indices = indices[np.lexsort((indices[:, 0], indices[:, 1]))]
        L = torch.sparse_coo_tensor(indices.T, L.data, L.shape, device=device)
        return L

    def _calculate_random_walk_matrix(self, adj_mx):

        # tf.Print(adj_mx, [adj_mx], message="This is adj: ")

        adj_mx = adj_mx + torch.eye(int(adj_mx.shape[0])).to(device)
        d = torch.sum(adj_mx, 1)
        d_inv = 1. / d
        d_inv = torch.where(torch.isinf(d_inv), torch.zeros(d_inv.shape).to(device), d_inv)
        d_mat_inv = torch.diag(d_inv)
        random_walk_mx = torch.mm(d_mat_inv, adj_mx)
        return random_walk_mx

    def forward(self, inputs, hx, adj):
        """Gated recurrent unit (GRU) with Graph Convolution.
        :param inputs: (B, num_nodes * input_dim)
        :param hx: (B, num_nodes * rnn_units)

        :return
        - Output: A `2-D` tensor with shape `(B, num_nodes * rnn_units)`.
        """
        adj_mx = self._calculate_random_walk_matrix(adj).t()
        output_size = 2 * self._num_units
        if self._use_gc_for_ru:
            fn = self._gconv
        else:
            fn = self._fc
        value = torch.sigmoid(fn(inputs, adj_mx, hx, output_size, bias_start=1.0))
        value = torch.reshape(value, (-1, self._num_nodes, output_size)) #[batchsize,207,128]
        r, u = torch.split(tensor=value, split_size_or_sections=self._num_units, dim=-1)
        r = torch.reshape(r, (-1, self._num_nodes * self._num_units)) 
        u = torch.reshape(u, (-1, self._num_nodes * self._num_units))

        c = self._gconv(inputs, adj_mx, r * hx, self._num_units)
        if self._activation is not None:
            c = self._activation(c)

        new_state = u * hx + (1.0 - u) * c
        return new_state

    @staticmethod
    def _concat(x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)

    def _fc(self, inputs, state, output_size, bias_start=0.0):
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size * self._num_nodes, -1))
        state = torch.reshape(state, (batch_size * self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=-1)
        input_size = inputs_and_state.shape[-1]
        weights = self._fc_params.get_weights((input_size, output_size))
        value = torch.sigmoid(torch.matmul(inputs_and_state, weights))
        biases = self._fc_params.get_biases(output_size, bias_start)
        value += biases
        return value

    def _gconv(self, inputs, adj_mx, state, output_size, bias_start=0.0):
        
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1)) #[64,207,2]
        state = torch.reshape(state, (batch_size, self._num_nodes, -1)) #[64,207,64]
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.size(2)

        x = inputs_and_state
        x0 = x.permute(1, 2, 0)  # (num_nodes, total_arg_size, batch_size)
        x0 = torch.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
        x = torch.unsqueeze(x0, 0) # [1,207,64*66]

        if self._max_diffusion_step == 0:
            pass
        else:
            x1 = torch.mm(adj_mx, x0)
            x = self._concat(x, x1)

            for k in range(2, self._max_diffusion_step + 1):
                x2 = 2 * torch.mm(adj_mx, x1) - x0
                x = self._concat(x, x2)
                x1, x0 = x2, x1
            '''
            Option:
            for support in self._supports:
                x1 = torch.sparse.mm(support, x0)
                x = self._concat(x, x1)

                for k in range(2, self._max_diffusion_step + 1):
                    x2 = 2 * torch.sparse.mm(support, x1) - x0
                    x = self._concat(x, x2)
                    x1, x0 = x2, x1
            '''
        num_matrices = self._max_diffusion_step + 1  # Adds for x itself.
        x = torch.reshape(x, shape=[num_matrices, self._num_nodes, input_size, batch_size])
        x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, order)
        x = torch.reshape(x, shape=[batch_size * self._num_nodes, input_size * num_matrices])

        weights = self._gconv_params.get_weights((input_size * num_matrices, output_size))
        x = torch.matmul(x, weights)  # (batch_size * self._num_nodes, output_size)

        biases = self._gconv_params.get_biases(output_size, bias_start) # 128
        x += biases
        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
        return torch.reshape(x, [batch_size, self._num_nodes * output_size])


class ODEFunc_SGODE(nn.Module):  
    def __init__(self, hidden_size, dropout, num_nodes,embed_dim,Atype):
        super(ODEFunc_SGODE, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)

        self.x0 = torch.zeros(64, num_nodes, hidden_size)
          
        softmax = nn.Softmax(dim=0)
        self.node_embeddings1 = nn.Parameter(softmax(torch.rand(num_nodes, embed_dim)), requires_grad=True)
        self.node_embeddings2 = nn.Parameter(softmax(torch.rand(num_nodes, embed_dim)), requires_grad=True)
        initial_type=Atype
        if initial_type==1:
            self.node_embeddings3 = nn.Parameter(softmax(torch.rand(num_nodes, embed_dim)), requires_grad=True)
            self.node_embeddings4 = nn.Parameter(softmax(torch.rand(num_nodes, embed_dim)), requires_grad=True)
        elif initial_type==2:
            self.node_embeddings3 = nn.Parameter(1e-6*(torch.ones(num_nodes, embed_dim)), requires_grad=True)
            self.node_embeddings4 = nn.Parameter(1e-6*(torch.ones(num_nodes, embed_dim)), requires_grad=True)
        self.C = nn.Parameter(softmax(torch.rand(num_nodes)), requires_grad=True)                  
        self.relu = nn.ReLU()
        self.wt = nn.Linear(hidden_size, hidden_size)

    def forward(self, t, x):  
        """
        :param t:  end time tick, if t is not used, it is an autonomous system
        :param x:  initial value   N_node * N_dim   400 * hidden_size
        :return:
        """
        pos = torch.mm(self.node_embeddings1, self.node_embeddings2.transpose(0, 1))
        pos = self.relu(pos)
        
        neg = torch.mm(self.node_embeddings3, self.node_embeddings4.transpose(0, 1))    
        neg = self.relu(neg)               
        K_weight = pos - neg
        self_x = self.C.reshape(1,-1,1) * x
        x = torch.einsum("nm,bmc->bnc", K_weight, x) 
        x = x + self_x 
        x = self.wt(x)
        x = x + self.x0     
        x = self.dropout_layer(x)
        x = F.relu(x)
        return x

class ODEBlock(nn.Module):
    def __init__(self, odefunc, rtol=1e-2, atol=1e-3, method='euler', adjoint=False, terminal=False): #vt,         :param vt:
        """
        :param odefunc: X' = f(X, t, G, W)
        :param rtol: optional float64 Tensor specifying an upper bound on relative error,
            per element of `y`.
        :param atol: optional float64 Tensor specifying an upper bound on absolute error,
            per element of `y`.
        :param method:
            'explicit_adams': AdamsBashforth,
            'fixed_adams': AdamsBashforthMoulton,
            'adams': VariableCoefficientAdamsBashforth,
            'tsit5': Tsit5Solver,
            'dopri5': Dopri5Solver,
            'euler': Euler,
            'midpoint': Midpoint,
            'rk4': RK4,
        """

        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        # self.integration_time_vector = vt  # time vector
        self.rtol = rtol
        self.atol = atol
        self.method = method
        self.adjoint = adjoint
        self.terminal = terminal

    def forward(self, vt, x):
        integration_time_vector = vt.type_as(x)
        if self.adjoint:
            out = ode.odeint_adjoint(self.odefunc, x, integration_time_vector,
                                     rtol=self.rtol, atol=self.atol, method=self.method)
        else:
            out = ode.odeint(self.odefunc, x, integration_time_vector,
                             rtol=self.rtol, atol=self.atol, method=self.method)
        # return out[-1]
        return out[-1] if self.terminal else out  # 100 * 400 * 10
    
class LayerParams:
    def __init__(self, rnn_network: torch.nn.Module, layer_type: str):
        self._rnn_network = rnn_network
        self._params_dict = {}
        self._biases_dict = {}
        self._type = layer_type

    def get_weights(self, shape):
        if shape not in self._params_dict:
            nn_param = torch.nn.Parameter(torch.empty(*shape, device=device))
            torch.nn.init.xavier_normal_(nn_param)
            self._params_dict[shape] = nn_param
            self._rnn_network.register_parameter('{}_weight_{}'.format(self._type, str(shape)),
                                                 nn_param)
        return self._params_dict[shape]

    def get_biases(self, length, bias_start=0.0):
        if length not in self._biases_dict:
            biases = torch.nn.Parameter(torch.empty(length, device=device))
            torch.nn.init.constant_(biases, bias_start)
            self._biases_dict[length] = biases
            self._rnn_network.register_parameter('{}_biases_{}'.format(self._type, str(length)),
                                                 biases)

        return self._biases_dict[length]


class ODE_DCGRUCell(torch.nn.Module):
    def __init__(self, num_units, input_dim, max_diffusion_step, num_nodes, nonlinearity='tanh',
                 filter_type="laplacian", use_gc_for_ru=True,embed_dim=10,Atype=2):
        """
        :param num_units:
        :param adj_mx:
        :param max_diffusion_step:
        :param num_nodes:
        :param nonlinearity:
        :param filter_type: "laplacian", "random_walk", "dual_random_walk".
        :param use_ode_for_gru: whether to use ode Graph convolution to calculate the reset and update gates.
        """
        super().__init__()
        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu
        # support other nonlinearities up here?
        self._num_nodes = num_nodes
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._supports = []
        if input_dim:
            self._gode = ODEBlock(ODEFunc_SGODE(num_units+input_dim, 
                                            dropout=0.0, num_nodes=num_nodes, 
                                            embed_dim=embed_dim,Atype=Atype
                                            ),adjoint=False)  
        else:
            self._gode = ODEBlock(ODEFunc_SGODE(num_units, 
                                            dropout=0.0, num_nodes=num_nodes, 
                                            embed_dim=embed_dim,Atype=Atype
                                            ),adjoint=False)  
        self._use_ode_for_gru = True
 
        self.map = torch.nn.Sequential(
            torch.nn.Linear(num_units+input_dim,num_units+input_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(num_units+input_dim,num_units+input_dim)
        )

        self._fc_params = LayerParams(self, 'fc')
        self._gconv_params = LayerParams(self, 'gconv')

    @staticmethod
    def _build_sparse_matrix(L):
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        # this is to ensure row-major ordering to equal torch.sparse.sparse_reorder(L)
        indices = indices[np.lexsort((indices[:, 0], indices[:, 1]))]
        L = torch.sparse_coo_tensor(indices.T, L.data, L.shape, device=device)
        return L

    def _calculate_random_walk_matrix(self, adj_mx):

        # tf.Print(adj_mx, [adj_mx], message="This is adj: ")

        adj_mx = adj_mx + torch.eye(int(adj_mx.shape[0])).to(device)
        d = torch.sum(adj_mx, 1)
        d_inv = 1. / d
        d_inv = torch.where(torch.isinf(d_inv), torch.zeros(d_inv.shape).to(device), d_inv)
        d_mat_inv = torch.diag(d_inv)
        random_walk_mx = torch.mm(d_mat_inv, adj_mx)
        return random_walk_mx

    def forward(self, inputs, hx):
        """Gated recurrent unit (GRU) with Graph Convolution.
        :param inputs: (B, num_nodes * input_dim)
        :param hx: (B, num_nodes * rnn_units)

        :return
        - Output: A `2-D` tensor with shape `(B, num_nodes * rnn_units)`.
        """
        output_size = 2 * self._num_units
        
        fn = self._gconv
        
        value = torch.sigmoid(fn(inputs, hx, output_size, bias_start=1.0))
        value = torch.reshape(value, (-1, self._num_nodes, output_size)) #[batchsize,207,128]
        r, u = torch.split(tensor=value, split_size_or_sections=self._num_units, dim=-1)
        r = torch.reshape(r, (-1, self._num_nodes * self._num_units)) 
        u = torch.reshape(u, (-1, self._num_nodes * self._num_units))

        c = self._gconv(inputs,  r * hx, self._num_units)
        if self._activation is not None:
            c = self._activation(c)

        new_state = u * hx + (1.0 - u) * c
        return new_state

    @staticmethod
    def _concat(x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)

    def _fc(self, inputs, state, output_size, bias_start=0.0):
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size * self._num_nodes, -1))
        state = torch.reshape(state, (batch_size * self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=-1)
        input_size = inputs_and_state.shape[-1]
        weights = self._fc_params.get_weights((input_size, output_size))
        value = torch.sigmoid(torch.matmul(inputs_and_state, weights))
        biases = self._fc_params.get_biases(output_size, bias_start)
        value += biases
        return value

    def _gconv(self, inputs,  state, output_size, bias_start=0.0):
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1)) #[64,207,2]
        state = torch.reshape(state, (batch_size, self._num_nodes, -1)) #[64,207,64]
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.size(2)

        x = inputs_and_state
        x0 = x.permute(1, 2, 0)  # (num_nodes, total_arg_size, batch_size)
        x0 = torch.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
        x = torch.unsqueeze(x0, 0) # [1,207,64*66]

        num_matrices = self._max_diffusion_step + 1  # Adds for x itself.

        if self._use_ode_for_gru:
            if self._max_diffusion_step == 0:
                pass
            else:
                vtime = torch.linspace(start=0, end=1,steps=self._max_diffusion_step + 1).float()
                vtime = vtime.type_as(x)
                self._gode.odefunc.x0 = self.map(inputs_and_state)
                x = self._gode(vtime , inputs_and_state) #[num_matrices,batch_size, num_nodes, input_size]                
                x = x.permute(1,2,0,3)
                weights = self._gconv_params.get_weights((num_matrices,input_size, output_size))
                
                x = torch.einsum("bnkm,kmc->bnc", x, weights)
            
        biases = self._gconv_params.get_biases(output_size, bias_start) # 128
        x += biases
        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
        return torch.reshape(x, [batch_size, self._num_nodes * output_size])
    

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def cosine_similarity_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).to(device)
    return -torch.autograd.Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature, eps=1e-10):
    sample = sample_gumbel(logits.size(), eps=eps)
    y = logits + sample
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature, hard=False, eps=1e-10):
  """Sample from the Gumbel-Softmax distribution and optionally discretize.
  Args:
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
  Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
  """
  y_soft = gumbel_softmax_sample(logits, temperature=temperature, eps=eps)
  if hard:
      shape = logits.size()
      _, k = y_soft.data.max(-1)
      y_hard = torch.zeros(*shape).to(device)
      y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
      y = torch.autograd.Variable(y_hard - y_soft.data) + y_soft
  else:
      y = y_soft
  return y

class Seq2SeqAttrs:
    def __init__(self, **model_kwargs):

        self.max_diffusion_step = int(model_kwargs.get('max_diffusion_step', 2))
        self.cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        self.filter_type = model_kwargs.get('filter_type', 'laplacian')
        self.num_nodes = int(model_kwargs.get('num_nodes', 1))
        self.num_rnn_layers = int(model_kwargs.get('num_rnn_layers', 1))
        self.rnn_units = int(model_kwargs.get('rnn_units'))
        self.hidden_state_size = self.num_nodes * self.rnn_units


class EncoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, **model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, **model_kwargs)
        self.input_dim = int(model_kwargs.get('input_dim', 1))
        self.seq_len = int(model_kwargs.get('seq_len'))  # for the encoder
        self.use_ode_for_gru = bool(model_kwargs.get('use_ode_for_gru'))
        self.embed_dim = int(model_kwargs.get('embed_dim'))
        self.Atype = int(model_kwargs.get('Atype'))
        #use_ode_for_gru: True
        if self.use_ode_for_gru:
            self.dcgru_layers = nn.ModuleList(
                [ODE_DCGRUCell(self.rnn_units,self.input_dim, self.max_diffusion_step, self.num_nodes,
                       filter_type=self.filter_type,embed_dim=self.embed_dim,Atype=self.Atype) for _ in range(self.num_rnn_layers)])
        else:
            self.dcgru_layers = nn.ModuleList(
                [DCGRUCell(self.rnn_units, self.max_diffusion_step, self.num_nodes,
                        filter_type=self.filter_type) for _ in range(self.num_rnn_layers)])

    def forward(self, inputs, hidden_state=None):
        """
        Encoder forward pass.
        :param inputs: shape (batch_size, self.num_nodes * self.input_dim) # fourier 
        :param hidden_state: (num_layers, batch_size, self.hidden_state_size)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.hidden_state_size)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                 (lower indices mean lower layers)
        """
        batch_size, _ = inputs.size()
        if hidden_state is None:
            hidden_state = torch.zeros((self.num_rnn_layers, batch_size, self.hidden_state_size),
                                       device=device)
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        return output, torch.stack(hidden_states)  # runs in O(num_layers) so not too slow


class DecoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, **model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, **model_kwargs)
        self.output_dim = int(model_kwargs.get('output_dim', 1))
        self.horizon = int(model_kwargs.get('horizon', 1))  # for the decoder
        self.projection_layer = nn.Linear(self.rnn_units, self.output_dim)
        self.use_ode_for_gru = bool(model_kwargs.get('use_ode_for_gru'))
        self.embed_dim = int(model_kwargs.get('embed_dim'))
        self.Atype = int(model_kwargs.get('Atype'))
        #use_ode_for_gru: True
        if self.use_ode_for_gru:
            self.dcgru_layers = nn.ModuleList(
                [ODE_DCGRUCell(self.rnn_units, 1, self.max_diffusion_step, self.num_nodes,
                       filter_type=self.filter_type,embed_dim=self.embed_dim,Atype=self.Atype) for _ in range(self.num_rnn_layers)])
        else:
            self.dcgru_layers = nn.ModuleList(
                [DCGRUCell(self.rnn_units, self.max_diffusion_step, self.num_nodes,
                        filter_type=self.filter_type) for _ in range(self.num_rnn_layers)])
    

    def forward(self, inputs,  hidden_state=None):
        """
        :param inputs: shape (batch_size, self.num_nodes * self.output_dim)
        :param hidden_state: (num_layers, batch_size, self.hidden_state_size)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.num_nodes * self.output_dim)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                 (lower indices mean lower layers)
        """
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        projected = self.projection_layer(output.view(-1, self.rnn_units))
        output = projected.view(-1, self.num_nodes * self.output_dim)

        return output, torch.stack(hidden_states)


class Time_prompt_GraphODE(nn.Module, Seq2SeqAttrs):
    def __init__(self, temperature, logger, **model_kwargs):
        super().__init__()
        Seq2SeqAttrs.__init__(self, **model_kwargs)
        self.encoder_model = EncoderModel(**model_kwargs)
        self.decoder_model = DecoderModel(**model_kwargs)
        self.cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        self.use_curriculum_learning = bool(model_kwargs.get('use_curriculum_learning', False))
        self._logger = logger
        self.temperature = temperature
        self.dim_fc = int(model_kwargs.get('dim_fc', False))
        self.embedding_dim = 10
        self.relu=nn.ReLU()
        self.only_pos=1

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def encoder(self, inputs):
        """
        Encoder forward pass
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :return: encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        """
        encoder_hidden_state = None
        for t in range(self.encoder_model.seq_len):
            _, encoder_hidden_state = self.encoder_model(inputs[t], encoder_hidden_state)

        return encoder_hidden_state

    def decoder(self, encoder_hidden_state,labels=None, batches_seen=None):
        """
        Decoder forward pass
        :param encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        :param labels: (self.horizon, batch_size, self.num_nodes * self.output_dim) [optional, not exist for inference]
        :param batches_seen: global step [optional, not exist for inference]
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        batch_size = encoder_hidden_state.size(1)
        go_symbol = torch.zeros((batch_size, self.num_nodes * self.decoder_model.output_dim),
                                device=device)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol 

        outputs = []

        for t in range(self.decoder_model.horizon):
            decoder_output, decoder_hidden_state = self.decoder_model(decoder_input, 
                                                                      decoder_hidden_state)
            decoder_input = decoder_output
            outputs.append(decoder_output)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    decoder_input = labels[t]
        outputs = torch.stack(outputs)
        return outputs

    def forward(self, label, inputs, labels=None, batches_seen=None):
        # inputs,label, (batch_size, Heigth, width, seq_len)
        """
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim) # batch*seq_len,1, h,w  ---> fourier ----->
        :param labels: shape (horizon, batch_size, num_sensor * output)
        :param batches_seen: batches seen till now
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        encoder_hidden_state = self.encoder(inputs)
        outputs = self.decoder(encoder_hidden_state, labels, batches_seen=batches_seen)
        return outputs

if __name__ == "__main__":
    
    def simple_logger(message):
        print(message)

    model_kwargs = {
        'input_dim': 32, 
        'seq_len': 10, 
        'horizon': 10,  
        'num_nodes': 1024,  
        'rnn_units': 64, 
        'embed_dim': 10,
        'Atype': 2,  
        'max_diffusion_step': 2,  
        'cl_decay_steps': 1000, 
        'use_ode_for_gru': True, 
        'filter_type': 'laplacian',  
    }

    model = Time_prompt_GraphODE(temperature=1.0, logger=simple_logger, **model_kwargs)
    model.to(device)

    print(f"Total trainable parameters: {count_parameters(model)}")

    """
            :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
            :param labels: shape (horizon, batch_size, num_sensor * output)
            :param batches_seen: batches seen till now
            :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
    """
    seq_len = 10
    batch_size = 1
    num_sensor = 1024 
    input_dim = 32
    num_nodes = 1024
    output_dim = 32
    inputs = torch.randn(seq_len, batch_size, num_sensor * input_dim).to(device)
    labels =  torch.randn(seq_len, batch_size, num_sensor * input_dim).to(device)
    # print("inputs shape:", inputs.shape)
    # print("labels shape:", labels.shape)
    node_feas = torch.randn(seq_len, batch_size, num_sensor * input_dim).to(device)
    temp = 1.0
    gumbel_soft = False
    batches_seen = 1
    outputs = model.forward('without_regularization', inputs, labels, batches_seen)
    print("outputs shape:", outputs.shape)