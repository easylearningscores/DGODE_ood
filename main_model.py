import torch
from Codebank import *
from FrequencyNetwork import *
from GrapODE_Solver import *
from TemporalGNN import *
import torch
import torch.nn as nn
import torch.optim as optim
from module import *
# '''
# Let's take the Navier-Stokes equations as an example. The data dimensions are [B, T, C, H, W], representing Batch size, Time step, Channel, Height, and Width, respectively.
# '''

class DGODE(nn.Module):
    def __init__(self, shape_in, num_classes, batch_size=2):
        super(DGODE, self).__init__()
        T, C, H, W = shape_in
        self.Frequency_Network = FourierNet(input_len=10*1, modes1=12, modes2=12, pred_len=10*1, width=20)
        self.Temporal_GNN = Equivariant_Graph_Operator(in_node_nf=10*1, hidden_nf=32, out_node_nf=10*1, in_edge_nf=1)
        self.vqvae = VectorQuantizerEMA(num_embeddings=128, embedding_dim=64, commitment_cost=0.99)
        self.classifier = nn.Linear(T*C*H*W, num_classes)
        self.skip_conneciton = ConvolutionalNetwork.skip_connection(shape_in=shape_in)
        self.batch_size = batch_size

        
        self.model_kwargs = {
            'input_dim': 1, 
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
        
        self.GraphODE_solver = Time_prompt_GraphODE(temperature=1.0, logger=self.simple_logger, **self.model_kwargs)
    
    def simple_logger(self, message):
        print(message)
    
    def forward(self, inputs):
        B, T, C, H, W = inputs.shape
        skip_feature = self.skip_conneciton(inputs)
        h_f = self.Frequency_Network(inputs)
        
        batch_size = self.batch_size
        n_nodes = 1024 
        n_feat = 10
        x_dim = 2
        input_graph = inputs.view(B, T*C, H*W)
        input_graph = input_graph.permute(0,2,1).reshape(-1, n_feat)
        # h = torch.ones(batch_size * n_nodes, n_feat)
        #print('input_graph shape', input_graph.shape) # h shape torch.Size([1024, 10])
        x = torch.ones(batch_size * n_nodes, x_dim)
        edges, edge_attr = get_edges_batch(n_nodes, batch_size)
        h_s, _ = self.Temporal_GNN(input_graph, x, edges, edge_attr)
        h_s = h_s.reshape(batch_size, n_nodes, -1)

        features_extraction = h_f + h_s
        Node_representation = features_extraction
        f_class = Node_representation
        # print("Node_representation shape:", Node_representation.shape)


        Environmental_representation = features_extraction

        # Graph ODE Solver
        B, N, _ = Node_representation.shape
        Node_representation = Node_representation.reshape(B, N, T, C).permute(2, 0, 1, 3).reshape(T, B, -1)

        batches_seen = 1
        GraphODE_solveroutputs = self.GraphODE_solver.forward('without_regularization', Node_representation, Node_representation, batches_seen)
        solver_outputs = GraphODE_solveroutputs.permute(1, 0, 2)
        solver_outputs = solver_outputs.reshape(B,T,C,H,W) + skip_feature
        # VQ-VAE
        Environmental_representation = Environmental_representation.reshape(B, H, W, T)
        h_e = Environmental_representation.permute(0, 3, 1, 2)
        loss_vq, K_he, perplexity, encodings = self.vqvae(h_e)

        # classification
        f_class = f_class.view(B, -1)
        class_feature = self.classifier(f_class)
        return solver_outputs, K_he, class_feature, Node_representation, Environmental_representation



if __name__ == "__main__":
    mse_loss_fn = nn.MSELoss()
    classification_loss_fn = nn.CrossEntropyLoss()

    model = DGODE(shape_in=(10, 1, 32, 32), num_classes=8, batch_size=2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    inputs = torch.rand(2, 10, 1, 32, 32)    
    solver_outputs, K_he, class_feature, Node_representation, Environmental_representation = model(inputs)
    print(solver_outputs.shape, class_feature.shape)        
    # regression_loss = mse_loss_fn(solver_outputs, Physics_filed_labels)
    # #classification_loss = classification_loss_fn(class_feature, class_target)
    # classification_loss = 0
        
    # total_loss = regression_loss + classification_loss
        
 
 