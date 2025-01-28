import torch
import torch.nn as nn
from layer3 import Attn_head, SimpleAttLayer


class Model(nn.Module):
    def __init__(self, residual, n_layer):
        super(Model, self).__init__()
        self.n_layer = n_layer - 1
        self.activation = nn.SELU()
        self.activation2 = nn.LeakyReLU(negative_slope=0.01)
        self.in_drop = 0.1
        self.coef_drop = 0.1
        self.attention_size = 128
        self.attn_head_input = Attn_head(7, 128, self.in_drop, self.coef_drop, self.activation, residual)
        self.simple_att_layer_initial = SimpleAttLayer(self.attention_size)
        self.attn_head_final = nn.ModuleList([
            Attn_head(128 , 128, self.in_drop, self.coef_drop, self.activation, residual) for _ in range(n_layer)
        ])
        self.simple_att_layer_final = nn.ModuleList([
            SimpleAttLayer(self.attention_size) for _ in range(n_layer)
        ])
        self.fc = nn.Linear(128, 1)  # 全连接层，将输出映射到1维

    def forward(self, inputs_input, edge_index_list):

        # Step 1: Attn_head processing for the input graph
        out_input = self.attn_head_input(inputs_input, edge_index_list[0])
        #print('out_input shape', out_input.shape)

        # Step 2: Apply SimpleAttLayer
        combined_out = self.simple_att_layer_initial(out_input.unsqueeze(1))

        # Step 3: Apply n_layer times Attn_head and SimpleAttLayer
        final_out = combined_out
        #print('combined_out shape', combined_out.shape)
        for i in range(self.n_layer):
            layer_out = self.attn_head_final[i](final_out.squeeze(1), edge_index_list[0])
            layer_out = self.simple_att_layer_final[i](layer_out.unsqueeze(1))
            final_out = final_out + layer_out  # Add residual connection

        # Step 4: Apply the final fully connected layer and ELU activation
        final_out = final_out.squeeze(1)
        final_out_last_node = final_out[-1]
        final_out_last_node = self.fc(final_out_last_node)
        final_out_last_node = self.activation2(final_out_last_node)

        return final_out_last_node
