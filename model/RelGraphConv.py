from helper import *
from .message_passing import MessagePassing


class RelGraphConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_rels, act=lambda x:x, params=None):
        super(self.__class__, self).__init__()

        self.p = params
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_rels = num_rels
        self.act = act
        self.device = None

        self.w_rel = get_param((in_channels, out_channels))
        self.w_loop = get_param((in_channels, out_channels))

        self.drop = torch.nn.Dropout(self.p.dropout)
        self.bn = torch.nn.BatchNorm1d(out_channels)

    def forward(self, x, edge_index, edge_type, edge_norm=None):
        if self.device is None:
            self.device = edge_index.device  

        num_edges = edge_index.size(1) // 2
        num_ent = x.size(0)

        self.edge_index = edge_index
        self.edge_type = edge_type

        self.loop_index = torch.stack([torch.arange(num_ent), torch.arange(num_ent)]).to(self.device)
        self.loop_type = torch.full((num_ent,), num_edges, dtype=torch.long).to(self.device)

        self.norm = self.compute_norm(self.edge_index, num_ent)
        res = self.propagate('add', self.edge_index, x=x, edge_type=self.edge_type, norm=self.norm, mode='in')
        loop_res = self.propagate('add', self.loop_index, x=x, edge_type=self.loop_type, norm=None, mode='loop')
        out = self.drop(res)*(1/2) + loop_res*(1/2)
        
        if self.p.bias: out = out + self.bias
        out = self.bn(out)
        
        return self.act(out)

    def message(self, x_j, norm, mode):
        if mode == 'in':
            weight = self.w_rel
        else:
            weight = self.w_loop       
        x_j = torch.mm(x_j, weight)
        return x_j if norm is None else x_j 
    
    def compute_norm(self, edge_index, num_ent):
        row, col = edge_index
        edge_weight = torch.ones_like(row).float()
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_ent)
        deg_inv = deg.pow(-0.5)
        deg_inv[deg_inv == float('inf')] = 0
        norm = deg_inv[row] * edge_weight * deg_inv[col]
        return norm


