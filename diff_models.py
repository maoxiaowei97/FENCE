import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def get_torch_trans(heads=8, layers=1, channels=64):  #Transformer Encoder
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def Conv1d_with_init(in_channels, out_channels, kernel_size): #一维卷积
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]#self.embedding 是一个通过 self.register_buffer 注册的预计算嵌入表
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class diff_CSDI(nn.Module):
    def __init__(self, config, inputdim=2):
        super().__init__()
        self.channels = int(config["channels"])

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=int(config["num_steps"]),
            embedding_dim=int(config["diffusion_embedding_dim"]),
        )
        '''
        使用 1D 卷积层将输入数据的维度从 inputdim 投影到 self.channels。
        这样做的目的是将输入数据扩展到指定的通道数，以便后续层处理。
        '''

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        # self.output_projection1 = Conv1d_with_init(self.channels, 1, 1)
        '''
        使用 1D 卷积层将通道数从 self.channels 投影到 1。
        这样的目的是将中间表示压缩回单一通道，以便输出结果
        '''
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=int(config["side_dim"]),
                    channels=self.channels,
                    diffusion_embedding_dim=int(config["diffusion_embedding_dim"]),
                    nheads=int(config["nheads"]),
                )
                for _ in range(int(config["layers"]))
            ]
        )
        '''
        初始化多层残差块，每个残差块的通道数为self.channels
        '''

    def forward(self, x, cond_info, diffusion_step):
        B, inputdim, K, L = x.shape
        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)
        
        diffusion_emb = self.diffusion_embedding(diffusion_step)
        
        skip = []
        attn_weights = None # 新增：初始化注意力权重变量
        for layer in self.residual_layers:
            # 修改：接收注意力权重
            x, skip_connection, attn_weights = layer(x, cond_info, diffusion_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x)
        x = F.relu(x)
        x = self.output_projection2(x)
        x = x.reshape(B, K, L)
        
        # 修改：模型前向过程返回最后一层的注意力权重
        return x, attn_weights


class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.feature_attn = nn.MultiheadAttention(channels, nheads, batch_first=True)
        self.feature_ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )
        self.feature_norm1 = nn.LayerNorm(channels)
        self.feature_norm2 = nn.LayerNorm(channels)
        
        # self.time_layer and self.feature_layer can be removed if get_torch_trans is not used elsewhere. For now we keep it.
        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)


    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y, None
            
        y = y.reshape(B, channel, K, L).permute(0, 3, 2, 1).reshape(B * L, K, channel)
        
        # 手动执行Transformer Encoder层的操作
        y_norm = self.feature_norm1(y)
        attn_output, attn_weights = self.feature_attn(y_norm, y_norm, y_norm, need_weights=True)
        y = y + attn_output
        
        ffn_output = self.feature_ffn(self.feature_norm2(y))
        y = y + ffn_output
        
        # 将维度恢复并返回权重
        y = y.reshape(B, L, K, channel).permute(0, 3, 2, 1).reshape(B, channel, K * L)
        
        # 对L维度上的注意力权重取平均，得到(B, K, K)
        attn_weights = attn_weights.view(B, L, K, K).mean(dim=1)
        
        return y, attn_weights

    def forward(self, x, cond_info, diffusion_emb):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)
        y = x + diffusion_emb

        y = self.forward_time(y, base_shape)
        # 修改：在残差块前向中传递注意力权重
        y, attn_weights = self.forward_feature(y, base_shape)
        y = self.mid_projection(y)

        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_info = self.cond_projection(cond_info)
        y = y + cond_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)

        # 修改：返回注意力权重
        return (x + residual) / math.sqrt(2.0), skip, attn_weights
