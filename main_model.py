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
        self.Frequency_Network = FourierNet(input_len=T*C, modes1=12, modes2=12, pred_len=T*C, width=20)
        self.Temporal_GNN = TemporalGNNModel(in_features=C, out_features=T*C, num_layers=1, num_nodes=H*W)
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
        
        node_features = inputs.reshape(B,T,C,H*W).permute(0,1,3,2)
        h_s = self.Temporal_GNN(node_features)
        features_extraction = h_f + h_s
        print(features_extraction.shape)
        Node_representation = features_extraction
        f_class = Node_representation

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
        return solver_outputs, loss_vq, K_he, class_feature, Node_representation, Environmental_representation



if __name__ == "__main__":
    mse_loss_fn = nn.MSELoss()
    classification_loss_fn = nn.CrossEntropyLoss()

    model = DGODE(shape_in=(10, 1, 32, 32), num_classes=8, batch_size=2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    inputs = torch.rand(2, 10, 1, 32, 32)    
    solver_outputs, loss_vq, K_he, class_feature, Node_representation, Environmental_representation = model(inputs)
    print(solver_outputs.shape, class_feature.shape)        
    # regression_loss = mse_loss_fn(solver_outputs, Physics_filed_labels)
    # #classification_loss = classification_loss_fn(class_feature, class_target)
    # classification_loss = 0
        
    # total_loss = regression_loss + classification_loss
        
 
 
