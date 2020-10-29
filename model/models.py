from helper import *
from model.RelGraphConv import RelGraphConv

class ConvE(torch.nn.Module):
	def __init__(self, args, num_entities, num_relations):
		super(ConvE, self).__init__()
		self.emb_e = torch.nn.Embedding(num_entities, args.embed_dim, padding_idx=0)
		self.emb_rel = torch.nn.Embedding(num_relations, args.embed_dim, padding_idx=0)
		self.inp_drop = torch.nn.Dropout(args.input_drop)
		self.hidden_drop = torch.nn.Dropout(args.hidden_drop)
		self.feature_map_drop = torch.nn.Dropout2d(args.feat_drop)
		self.loss = torch.nn.BCELoss()
		self.emb_dim1 = args.k_h
		self.emb_dim2 = args.embed_dim // self.emb_dim1

		self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=args.bias)
		self.bn0 = torch.nn.BatchNorm2d(1)
		self.bn1 = torch.nn.BatchNorm2d(32)
		self.bn2 = torch.nn.BatchNorm1d(args.embed_dim)
		self.register_parameter('b', Parameter(torch.zeros(num_entities)))
		self.fc = torch.nn.Linear(args.hidden_size, args.embed_dim)
		print(num_entities, num_relations)

	def init(self):
		xavier_normal_(self.emb_e.weight.data)
		xavier_normal_(self.emb_rel.weight.data)

	def forward(self, e1, rel):
		e1_embedded = self.emb_e(e1).view(-1, 1, self.emb_dim1, self.emb_dim2)
		rel_embedded = self.emb_e(rel).view(-1, 1, self.emb_dim1, self.emb_dim2)

		stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)

		stacked_inputs = self.bn0(stacked_inputs)
		x = self.inp_drop(stacked_inputs)
		x = self.conv1(x)
		x = self.bn1(x)
		x = F.relu(x)
		x = self.feature_map_drop(x)
		x = x.view(x.shape[0], -1)
		x = self.fc(x)
		x = self.hidden_drop(x)
		x = self.bn2(x)
		x = F.relu(x)
		x = torch.mm(x, self.emb_e.weight.transpose(1, 0))
		x += self.b.expand_as(x)
		pred = torch.sigmoid(x)

		return pred


class BaseModel(torch.nn.Module):
	def __init__(self, params):
		super(BaseModel, self).__init__()

		self.p = params
		self.act = torch.tanh
		self.bceloss = torch.nn.BCELoss()

	def loss(self, pred, true_label):
		return self.bceloss(pred, true_label)

class RGCNBase(BaseModel):
	def __init__(self, edge_index, edge_type, num_rel, params=None):
	 super(RGCNBase, self).__init__(params)

	 self.edge_index = edge_index
	 self.edge_type = edge_type
	 self.p.gcn_dim		= self.p.embed_dim if self.p.gcn_layer == 1 else self.p.gcn_dim
	 self.init_embed = get_param((self.p.num_ent, self.p.init_dim))
	 self.device = self.edge_index.device

	 self.init_rel = get_param((num_rel*2, self.p.gcn_dim))
	 self.conv1 = RelGraphConv(self.p.init_dim, self.p.gcn_dim, num_rel, act=self.act, params=self.p)
	 self.conv2 = RelGraphConv(self.p.gcn_dim, self.p.embed_dim, num_rel, act=self.act, params=self.p)

	 self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))

	def forward_base(self, sub, rel, drop1, drop2):
		r = self.init_rel
		x = self.conv1(self.init_embed, self.edge_index, self.edge_type)
		x = drop1(x)
		x = self.conv2(x, self.edge_index, self.edge_type)
		x = drop2(x)

		sub_emb = torch.index_select(x, 0, sub)
		rel_emb = torch.index_select(r, 0, rel)

		return sub_emb, rel_emb, x


class RelGCN_ConvE(RGCNBase):
	def __init__(self, edge_index, edge_type, params=None):
		super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params=params)
		self.bn0 = torch.nn.BatchNorm2d(1)
		self.bn1 = torch.nn.BatchNorm2d(self.p.num_filt)
		self.bn2 = torch.nn.BatchNorm1d(self.p.embed_dim)
		self.emb_e = torch.nn.Embedding(self.p.num_ent, self.p.gcn_dim, padding_idx=0)
		self.hidden_drop	= torch.nn.Dropout(self.p.hid_drop)
		self.hidden_drop2	= torch.nn.Dropout(self.p.hid_drop2)
		self.feature_drop	= torch.nn.Dropout(self.p.feat_drop)
		self.m_conv1		= torch.nn.Conv2d(1, out_channels=self.p.num_filt, kernel_size=(self.p.ker_sz, self.p.ker_sz), stride=1, padding=0, bias=self.p.bias)

		flat_sz_h		= int(2*self.p.k_w) - self.p.ker_sz + 1
		flat_sz_w		= self.p.k_h 	    - self.p.ker_sz + 1
		self.flat_sz		= flat_sz_h*flat_sz_w*self.p.num_filt
		self.fc			= torch.nn.Linear(self.flat_sz, self.p.embed_dim)

	def concat(self, e1_embed, rel_embed):
		e1_embed	= e1_embed. view(-1, 1, self.p.embed_dim)
		rel_embed	= rel_embed.view(-1, 1, self.p.embed_dim)
		stack_inp	= torch.cat([e1_embed, rel_embed], 1)
		stack_inp	= torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2*self.p.k_w, self.p.k_h))
		return stack_inp	

	def forward(self, sub, rel):
		sub_emb, rel_emb, all_ent	= self.forward_base(sub, rel, self.hidden_drop, self.feature_drop)
		rel_emb = self.emb_e(rel)
		stk_inp				= self.concat(sub_emb, rel_emb)
		x				= self.bn0(stk_inp)
		x				= self.m_conv1(x)
		x				= self.bn1(x)
		x				= F.relu(x)
		x				= self.feature_drop(x)
		x				= x.view(-1, self.flat_sz)
		x				= self.fc(x)
		x				= self.hidden_drop2(x)
		x				= self.bn2(x)
		x				= F.relu(x)

		x = torch.mm(x, all_ent.transpose(1,0))
		x += self.bias.expand_as(x)

		score = torch.sigmoid(x)
		return score