import torch
import torch.nn as nn
import torch.nn.functional as F


class STGCNATT(nn.Module):
    """MSTGANet: Multi-Scale Spatio-Temporal Graph Attention Network"""

    def __init__(self, grid_height, grid_width, input_steps=24, output_steps=24,
                 input_channels=2, encoder_channels1=64, encoder_channels2=128,
                 gcn_hidden_dim=64, attention_heads=4, dropout_rate=0.2):
        super().__init__()

        self.grid_height = grid_height
        self.grid_width = grid_width
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.num_nodes = grid_height * grid_width
        self.channels_per_branch = encoder_channels2 // 4

        # ---- 1. Multi-Scale 3D Convolutional Encoder (4 branches) ----
        self.encoder_scale1 = self._make_encoder_branch(
            input_channels, encoder_channels1//4, self.channels_per_branch,
            kernel_size=(1, 3, 3), padding=(0, 1, 1), dropout_rate=dropout_rate)

        self.encoder_scale2 = self._make_encoder_branch(
            input_channels, encoder_channels1//4, self.channels_per_branch,
            kernel_size=(3, 5, 5), padding=(1, 2, 2), dropout_rate=dropout_rate)

        self.encoder_scale3 = self._make_encoder_branch(
            input_channels, encoder_channels1//4, self.channels_per_branch,
            kernel_size=(5, 7, 7), padding=(2, 3, 3), dropout_rate=dropout_rate)

        self.encoder_temporal = self._make_encoder_branch(
            input_channels, encoder_channels1//4, self.channels_per_branch,
            kernel_size=(7, 1, 1), padding=(3, 0, 0), dropout_rate=dropout_rate)

        # Feature fusion after concatenation
        fusion_input_channels = 4 * self.channels_per_branch
        self.feature_fusion = nn.Sequential(
            nn.Conv3d(fusion_input_channels, encoder_channels2, kernel_size=1),
            nn.ReLU(), nn.BatchNorm3d(encoder_channels2), nn.Dropout3d(dropout_rate))

        # ---- 2. Dual-Branch Adaptive Graph Learning ----
        self.node_embedding_local = nn.Parameter(torch.randn(self.num_nodes, 16))
        self.node_embedding_global = nn.Parameter(torch.randn(self.num_nodes, 16))
        nn.init.xavier_uniform_(self.node_embedding_local)
        nn.init.xavier_uniform_(self.node_embedding_global)

        # ---- 3. Parallel GCN with Feature Fusion ----
        self.gcn_local = nn.Sequential(
            nn.Linear(encoder_channels2, gcn_hidden_dim), nn.ReLU(),
            nn.Dropout(dropout_rate), nn.Linear(gcn_hidden_dim, gcn_hidden_dim))

        self.gcn_global = nn.Sequential(
            nn.Linear(encoder_channels2, gcn_hidden_dim), nn.ReLU(),
            nn.Dropout(dropout_rate), nn.Linear(gcn_hidden_dim, gcn_hidden_dim))

        self.gcn_fusion = nn.Sequential(
            nn.Linear(gcn_hidden_dim * 2, gcn_hidden_dim), nn.ReLU(),
            nn.Dropout(dropout_rate), nn.Linear(gcn_hidden_dim, gcn_hidden_dim))

        # ---- 4. Spatio-Temporal Attention ----
        self.spatial_attention = nn.Sequential(
            nn.Linear(gcn_hidden_dim, gcn_hidden_dim // 2), nn.ReLU(),
            nn.Linear(gcn_hidden_dim // 2, 1), nn.Sigmoid())

        self.time_attention = nn.MultiheadAttention(
            gcn_hidden_dim, num_heads=attention_heads, batch_first=True, dropout=dropout_rate)

        # ---- 5. Decoder ----
        self.decoder = nn.Sequential(
            nn.Linear(gcn_hidden_dim, gcn_hidden_dim * 2), nn.ReLU(),
            nn.Dropout(dropout_rate), nn.Linear(gcn_hidden_dim * 2, output_steps * 2))

    def _make_encoder_branch(self, in_ch, mid_ch, out_ch, kernel_size, padding, dropout_rate):
        """Helper: build one multi-scale encoder branch"""
        return nn.Sequential(
            nn.Conv3d(in_ch, mid_ch, kernel_size=kernel_size, padding=padding),
            nn.ReLU(), nn.BatchNorm3d(mid_ch), nn.Dropout3d(dropout_rate),
            nn.Conv3d(mid_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.ReLU(), nn.BatchNorm3d(out_ch), nn.Dropout3d(dropout_rate))

    def forward(self, x):
        batch_size = x.shape[0]

        # -- Multi-scale encoding --
        x_input = x.permute(0, 1, 4, 2, 3)  # [B, C, T, H, W]
        f1 = self.encoder_scale1(x_input)
        f2 = self.encoder_scale2(x_input)
        f3 = self.encoder_scale3(x_input)
        ft = self.encoder_temporal(x_input)
        x_encoded = self.feature_fusion(torch.cat([f1, f2, f3, ft], dim=1))

        # -- Reshape to graph format --
        x_reshaped = x_encoded.permute(0, 2, 3, 4, 1).reshape(
            batch_size, self.input_steps, self.num_nodes, -1)

        # -- Adaptive graph construction --
        A_local = F.relu(self.node_embedding_local @ self.node_embedding_local.t())
        A_local = self._normalize(A_local) * self._local_mask()
        A_global = F.relu(self.node_embedding_global @ self.node_embedding_global.t()) + 0.1
        A_global = self._normalize(A_global)

        # -- Parallel GCN + Spatial Attention --
        spatial_features = []
        for t in range(self.input_steps):
            x_t = x_reshaped[:, t, :, :]
            h_local = self.gcn_local(torch.bmm(A_local.expand(batch_size, -1, -1), x_t))
            h_global = self.gcn_global(torch.bmm(A_global.expand(batch_size, -1, -1), x_t))
            fused = self.gcn_fusion(torch.cat([h_local, h_global], dim=-1))
            attn_w = self.spatial_attention(fused)
            spatial_features.append((fused * attn_w).unsqueeze(1))
        spatial_features = torch.cat(spatial_features, dim=1)

        # -- Temporal Attention --
        temp_input = spatial_features.permute(0, 2, 1, 3).reshape(
            batch_size * self.num_nodes, self.input_steps, -1)
        attended, _ = self.time_attention(temp_input, temp_input, temp_input)
        last_step = attended[:, -1, :]

        # -- Decode --
        output = self.decoder(last_step)
        output = output.view(batch_size, self.num_nodes, self.output_steps, 2)
        output = output.permute(0, 3, 1, 2).view(
            batch_size, 2, self.grid_height, self.grid_width, self.output_steps)
        return output

    def _normalize(self, A):
        """Symmetric Laplacian normalization"""
        A_self = A + torch.eye(A.size(0), device=A.device)
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(A_self.sum(dim=1)))
        return D_inv_sqrt @ A_self @ D_inv_sqrt

    def _local_mask(self):
        """Grid mask based on Manhattan distance (3×3 neighborhood)"""
        mask = torch.zeros(self.num_nodes, self.num_nodes)
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                idx = i * self.grid_width + j
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.grid_height and 0 <= nj < self.grid_width:
                            nidx = ni * self.grid_width + nj
                            mask[idx, nidx] = 1.0 / (1.0 + abs(di) + abs(dj))
        return mask