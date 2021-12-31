import torch.nn as nn
from torch.nn import LayerNorm
from module.layers import GATRConv, get_embs_graph


class GraphEncoder(nn.Module):
    def __init__(self, in_channels, config, vocab):
        super(GraphEncoder, self).__init__()
        self.graph_num_layers = config.graph_num_layers

        self.in_channels = in_channels
        self.out_channels = self.in_channels
        self.num_relations = vocab.type_size
        self.heads = 4

        self.gnns = nn.ModuleList(
            GATRConv(self.in_channels, self.out_channels, num_relations=self.num_relations, heads=self.heads, concat=False)
            for _ in range(self.graph_num_layers))
        self.rnn = nn.GRUCell(self.out_channels, self.out_channels)

        self.layer_norm = LayerNorm(self.out_channels)

    def forward(self, embeddings, edges):
        # embeddings: sen_num x sent_len x sent_rep_size
        # edges(edges_index, edges_type): sen_num x 2

        batch_geometric = get_embs_graph(edges, embeddings)
        memory_bank = batch_geometric.x

        for layer in self.gnns:
            new_memory_bank = layer(memory_bank, batch_geometric.edge_index, edge_type=batch_geometric.y)
            memory_bank = self.rnn(new_memory_bank, memory_bank)

        # for layer in self.gnns:
        #     memory_bank = layer(memory_bank, batch_geometric.edge_index, edge_type=batch_geometric.y)

        out = memory_bank.view(embeddings.shape)

        out = self.layer_norm(out)  # sen_num x sent_len x sent_rep_size

        return out
