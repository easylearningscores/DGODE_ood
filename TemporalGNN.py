import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalGNNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(TemporalGNNLayer, self).__init__()
        self.phi_e = nn.Linear(3 * in_features, out_features)
    
    def forward(self, h_i_t, h_i_t_1, N_i_t):
        N_i_t_agg = torch.mean(N_i_t, dim=2)  
        concat_features = torch.cat([h_i_t, h_i_t_1, N_i_t_agg], dim=-1)
        h_i_t_l = F.relu(self.phi_e(concat_features))
        return h_i_t_l

class TemporalGNN(nn.Module):
    def __init__(self, in_features, out_features, num_layers, num_nodes):
        super(TemporalGNN, self).__init__()
        self.layers = nn.ModuleList([TemporalGNNLayer(in_features if i == 0 else out_features, out_features) for i in range(num_layers)])
        self.num_layers = num_layers
        self.num_nodes = num_nodes

    def forward(self, node_features):
        h = node_features
        batch_size = h.shape[0]
        for l in range(self.num_layers):
            h_new = []
            for i in range(self.num_nodes):
                h_i_t = h[:, :, i, :]
                h_i_t_1 = torch.cat([torch.zeros(batch_size, 1, h_i_t.shape[2]).to(h.device), h_i_t[:, :-1, :]], dim=1)
                N_i_t = h
                h_i_t_l = self.layers[l](h_i_t, h_i_t_1, N_i_t)
                h_new.append(h_i_t_l.unsqueeze(2))
            h = torch.cat(h_new, dim=2)
        return h

class NodeFeatureSummarizer(nn.Module):
    def __init__(self):
        super(NodeFeatureSummarizer, self).__init__()

    def forward(self, temporal_node_representations):
        spatial_node_features = torch.sum(temporal_node_representations, dim=1)
        return spatial_node_features

class TemporalGNNModel(nn.Module):
    def __init__(self, in_features, out_features, num_layers, num_nodes):
        super(TemporalGNNModel, self).__init__()
        self.temporal_gnn = TemporalGNN(in_features, out_features, num_layers, num_nodes)
        self.summarizer = NodeFeatureSummarizer()

    def forward(self, node_features):
        temporal_node_representations = self.temporal_gnn(node_features)
        spatial_node_features = self.summarizer(temporal_node_representations)
        return spatial_node_features




if __name__ == "__main__":
    in_features = 10
    out_features = 10
    num_layers = 3
    num_nodes = 1024
    T_obs = 5
    batch_size = 2

    model = TemporalGNNModel(in_features, out_features, num_layers, num_nodes)

    # (batch_size, T_obs, num_nodes, in_features)
    node_features = torch.randn(batch_size, T_obs, num_nodes, in_features)

    spatial_node_features = model(node_features)
    print(spatial_node_features.shape)  # (batch_size, num_nodes, out_features)
