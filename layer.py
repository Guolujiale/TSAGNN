import torch
import torch.nn as nn
import torch.nn.functional as F


class Attn_head(nn.Module):
    def __init__(self, in_channel, out_sz, in_drop, coef_drop, activation, residual):
        super(Attn_head, self).__init__()
        self.in_drop = nn.Dropout(in_drop)
        self.coef_drop = nn.Dropout(coef_drop)
        self.activation = activation
        self.residual = residual

        self.conv_seq = nn.Conv1d(in_channel, out_sz, 1, bias=False)
        self.conv_f1 = nn.Conv1d(out_sz, 1, 1)
        self.conv_f2 = nn.Conv1d(out_sz, 1, 1)
        self.layer_norm = nn.LayerNorm(out_sz)
        self.bias = nn.Parameter(torch.zeros(out_sz))

        if self.residual:
            if in_channel != out_sz:
                self.residual_conv = nn.Conv1d(in_channel, out_sz, 1)
            else:
                self.residual_conv = None

    def custom_scatter_add(self, weighted_feats, edge_index, dim_size):
        output = torch.zeros((dim_size, weighted_feats.shape[-1]), device=weighted_feats.device)
        for i in range(edge_index.shape[1]):
            output[edge_index[0, i]] += weighted_feats[i]
            # 这里加入对第二个节点的加权特征
            output[edge_index[1, i]] += weighted_feats[i]
        return output

    def forward(self, seq, edge_index):

        seq = seq.transpose(0, 1)
        seq_fts = self.conv_seq(seq)
        f_1 = self.conv_f1(seq_fts)
        f_2 = self.conv_f2(seq_fts)

        num_splits = 1
        #print('edge_index.shape:', edge_index.shape)
        total_edges = edge_index.shape[1]
        split_size = edge_index.shape[1] // num_splits
        #print('split_size:', split_size)

        results = []
        for i in range(num_splits):
            start_idx = i * split_size
            end_idx = (i + 1) * split_size if i != num_splits - 1 else total_edges
            sub_edge_index = edge_index[:, start_idx:end_idx]
            start_features = f_1[0, sub_edge_index[0]]
            #print('start_features:', start_features)
            end_features = f_2[0, sub_edge_index[1]]
            #print('end_features:', end_features)
            sum_features = start_features + end_features
            #print('sum_features:', sum_features)
            #print('sum_features:', sum_features.shape)
            e_sub = F.selu(sum_features)
            #print('e_sub:', e_sub)
            #print('e_sub:', e_sub.shape)
            results.append(e_sub)
        e = torch.cat(results, dim=0)
        #print('e:', e)
        #print('e:', e.shape)
        coefs = F.softmax(e, dim=0)
        #print('coefs:', coefs)
        #print('coefs:', coefs.shape)

        weighted_feats_list = []
        total_edges = edge_index.shape[1]
        split_size = total_edges // num_splits
        for i in range(num_splits):
            start_idx = i * split_size
            end_idx = (i + 1) * split_size if i != num_splits - 1 else total_edges
            sub_coefs = coefs[start_idx:end_idx]
            sub_seq_fts = seq_fts[:, edge_index[1][start_idx:end_idx]].transpose(0, 1)
            sub_weighted_feats = sub_coefs.unsqueeze(-1) * sub_seq_fts
            weighted_feats_list.append(sub_weighted_feats)
        #print('weighted_feats_list:', weighted_feats_list)
        weighted_feats = torch.cat(weighted_feats_list, dim=0)
        #print('weighted_feats:', weighted_feats)
        #print('edge_index:', edge_index)
        #print('seq.size(1):', seq.size(1))

        vals = self.custom_scatter_add(weighted_feats, edge_index, seq.size(1))
        #print('vals:', vals)
        #print('vals shape:', vals.shape)

        ret = vals + self.bias
        #print('ret0:', ret)
        #print('ret0 shape:', ret.shape)

        if self.residual:
            if self.residual_conv is not None:
                seq_transposed = seq.transpose(0, 1)
                seq_with_extra_dim = seq_transposed.unsqueeze(2)
                convoluted_seq = self.residual_conv(seq_with_extra_dim)
                convoluted_seq_squeezed = convoluted_seq.squeeze(2)
                ret = ret + convoluted_seq_squeezed
            else:
                if ret.shape == seq.shape:
                    ret = ret + seq
                else:
                    ret = ret + seq.transpose(0, 1)
        #print('ret:', ret)
        #print('ret shape:', ret.shape)
        ret = self.activation(ret)
        #print('ret act:', ret)
        #print('ret act shape:', ret.shape)
        return ret


class SimpleAttLayer(nn.Module):
    def __init__(self, attention_size, time_major=False, return_alphas=False):
        super(SimpleAttLayer, self).__init__()
        self.time_major = time_major
        self.return_alphas = return_alphas
        self.attention_size = attention_size
        self.initialized = False

        self.w_omega = nn.Parameter(torch.empty(0))
        self.b_omega = nn.Parameter(torch.empty(0))
        self.u_omega = nn.Parameter(torch.empty(0))

    def initialize_parameters(self, hidden_size, device):
        self.w_omega = nn.Parameter(torch.randn(hidden_size, self.attention_size, device=device) * 0.1)
        self.b_omega = nn.Parameter(torch.randn(self.attention_size, device=device) * 0.1)
        self.u_omega = nn.Parameter(torch.randn(self.attention_size, device=device) * 0.1)
        self.initialized = True

    def forward(self, inputs):
        if not self.initialized:
            hidden_size = inputs.shape[2]
            device = inputs.device
            self.initialize_parameters(hidden_size, device)

        v = torch.tanh(torch.matmul(inputs, self.w_omega) + self.b_omega)
        vu = torch.matmul(v, self.u_omega)
        alphas = F.softmax(vu, dim=1)
        output = (inputs * alphas.unsqueeze(-1)).sum(dim=1)

        if not self.return_alphas:
            return output
        else:
            return output, alphas
