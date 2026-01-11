import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import reduce
from typing import List, Optional

# =============================================================================
# 1. Global Constants & Helper Functions (Math & Formatting)
# =============================================================================

S_VALUE = 1

def active_scales(n_scales, epoch, total_epochs):
    if epoch < 0.2 * total_epochs:
        return 1
    else:
        return n_scales

def _inner_product(f1, f2, h):
    """    
    f1 - (B, J) : B functions, observed at J time points,
    f2 - (B, J) : same as f1
    h  - (J-1,1): weights used in the trapezoidal rule
    pay attention to dimension
    <f1, f2> = sum (h/2) (f1(t{j}) + f2(t{j+1}))
    """
    prod = f1 * f2 # (B, J = len(h) + 1)
    return torch.matmul((prod[:, :-1] + prod[:, 1:]), h.unsqueeze(dim=-1))/2

def _l1(f, h):
    # f dimension : ( B bases, J )
    B, J = f.size()
    return _inner_product(torch.abs(f), torch.ones((B, J), device=f.device), h)

def _l2(f, h):
    # f dimension : ( B bases, J )
    # output dimension - ( B bases, 1 )
    return torch.sqrt(_inner_product(f, f, h)) 

# =============================================================================
# 2. Basic Neural Network Blocks (Layers, Norms, Embeddings)
# =============================================================================

class LayerNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        # d is the normalization dimension
        self.d = d
        self.eps = eps
        self.alpha = nn.Parameter(torch.randn(d))
        self.beta = nn.Parameter(torch.randn(d))

    def forward(self, x):
        # x is a torch.Tensor
        # avg is the mean value of a layer
        avg = x.mean(dim=-1, keepdim=True)
        # std is the standard deviation of a layer (eps is added to prevent dividing by zero)
        std = x.std(dim=-1, keepdim=True) + self.eps
        return (x - avg) / std * self.alpha + self.beta

class FourierFeatureEmbedding(nn.Module):
    """
    Fourier Feature Embedding for periodic basis learning.
    Maps time t to [sin(2πf₁t), cos(2πf₁t), sin(2πf₂t), cos(2πf₂t), ...]
    """
    def __init__(self, n_frequencies=32, freq_range=(0.67, 4.17), learnable=True):
        super().__init__()
        self.n_frequencies = n_frequencies
        
        # Initialize frequencies linearly spaced in the range
        freqs = torch.linspace(freq_range[0], freq_range[1], n_frequencies)
        
        if learnable:
            self.frequencies = nn.Parameter(freqs)
        else:
            self.register_buffer('frequencies', freqs)
        
        # Output dimension: 2 * n_frequencies (sin + cos for each freq)
        self.out_dim = 2 * n_frequencies
    
    def forward(self, t):
        # t shape: (..., 1), frequencies shape: (n_frequencies,)
        # Compute 2π * f * t for each frequency
        angles = 2 * np.pi * t * self.frequencies  # (..., n_frequencies)
        
        # Concatenate sin and cos features
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

class PeriodicFeedForward(nn.Module):
    """
    FeedForward network with Fourier Feature Embedding for learning periodic basis functions.
    Architecture: t -> FourierFeatures -> MLP -> output
    """
    def __init__(self, hidden=[64, 64, 64], dropout=0.1, activation=F.relu,
                 n_frequencies=32, freq_range=(0.67, 4.17), learnable_freq=True):
        super().__init__()
        
        # Fourier feature embedding
        self.fourier_embed = FourierFeatureEmbedding(
            n_frequencies=n_frequencies,
            freq_range=freq_range,
            learnable=learnable_freq
        )
        
        self.sigma = activation
        
        # MLP layers: input is Fourier features (2 * n_frequencies)
        in_d = self.fourier_embed.out_dim
        dim = [in_d] + hidden + [1]
        self.layers = nn.ModuleList([nn.Linear(dim[i-1], dim[i]) for i in range(1, len(dim))])
        self.ln = nn.ModuleList([LayerNorm(k) for k in hidden])
        self.dp = nn.ModuleList([nn.Dropout(dropout) for _ in range(len(hidden))])
    
    def forward(self, t):
        # Apply Fourier feature embedding
        x = self.fourier_embed(t)  # (..., 2 * n_frequencies)
        
        # Pass through MLP
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = x + self.ln[i](x)  # Skip connection
            x = self.sigma(x)
            x = self.dp[i](x)
        
        # Linear activation at the last layer
        return self.layers[-1](x)

class FeedForward(nn.Module):
    """Standard FeedForward (Optional, kept for compatibility if needed)"""
    def __init__(self, in_d=1, hidden=[4,4,4], dropout=0.1, activation=F.relu):
        super().__init__()
        self.sigma = activation
        dim = [in_d] + hidden + [1]
        self.layers = nn.ModuleList([nn.Linear(dim[i-1], dim[i]) for i in range(1, len(dim))])
        self.ln = nn.ModuleList([LayerNorm(k) for k in hidden])
        self.dp = nn.ModuleList([nn.Dropout(dropout) for _ in range(len(hidden))])

    def forward(self, t):
        for i in range(len(self.layers)-1):
            t = self.layers[i](t)
            t = t + self.ln[i](t)
            t = self.sigma(t)
            t = self.dp[i](t)
        return self.layers[-1](t)

# =============================================================================
# 3. AdaFNN (Basis Generation Module)
# =============================================================================

class AdaFNN(nn.Module):
    def __init__(self, 
                 n_base=4, 
                 base_hidden=[64, 64, 64], 
                 grid=(0, 1),
                 sub_hidden=[128, 128, 128], 
                 dropout=0.1, 
                 lambda1=0.0, 
                 lambda2=0.0,
                 device=None,
                 n_frequencies=32, 
                 freq_range=(0.67, 4.17)
                ):

        super().__init__()
        self.n_base = n_base
        self.device = device
        self.lambda1 = lambda1   # sparsity
        self.lambda2 = lambda2   # orthogonality

        # ===== time grid =====
        # grid should include both end points
        grid = np.array(grid)
        self.J = len(grid)
        # send the time grid tensor to device
        self.t = torch.tensor(grid).to(device).float()
        self.h = torch.tensor(grid[1:] - grid[:-1]).to(device).float()

        # ===== basis MLPS =====
        self.BL = nn.ModuleList([
                PeriodicFeedForward(
                    hidden=base_hidden, 
                    dropout=dropout, 
                    activation=F.selu,
                    n_frequencies=n_frequencies,
                    freq_range=freq_range,
                    learnable_freq=True
                ) for _ in range(n_base)
            ])

        # Sub Network: PPG segment -> coefficients
        sub_layers = []
        sub_dims = [self.J] + sub_hidden + [n_base]
        for i in range(len(sub_dims) - 1):
            sub_layers.append(nn.Linear(sub_dims[i], sub_dims[i+1]))
            if i < len(sub_dims) - 2:  # No activation/dropout on last layer
                sub_layers.append(nn.ReLU())
                sub_layers.append(nn.Dropout(dropout))
        sub_layers.append(nn.Tanh())
        self.SubNet = nn.Sequential(*sub_layers)

        self.aggregator = nn.Linear(self.J, 1)

    # =====================================================
    # Forward: return reconstructed signal
    # =====================================================
    def forward(self, x):
        B, J = x.size()
        
        assert J == self.J, f"Expected {self.J} time points, got {J}"

        # --- Evaluate basis ---
        T = self.t.unsqueeze(-1)                 # (J,1)
        
        self.bases = [basis(T).transpose(-1, -2) for basis in self.BL]  # list of (1,J)
        bases_tensor = torch.cat(self.bases, dim=0)     # (K, J)

        l2_norm = _l2(bases_tensor, self.h).detach()  # (n_base, 1)
        self.normalized_bases = [self.bases[i] / (l2_norm[i, 0] + 1e-6) for i in range(self.n_base)]
        normalized_bases_tensor = torch.cat(self.normalized_bases, dim=0)

        # Step 2: SubNet predicts coefficients from PPG
        coefficients = self.SubNet(x)

        reconstruction = torch.matmul(coefficients, normalized_bases_tensor)                    # (B, K)

        return reconstruction, coefficients

    # =====================================================
    # R1: Sparsity
    # =====================================================
    def R1(self, l1_k):
        if self.lambda1 == 0: return torch.zeros(1).to(self.device)
        # sample l1_k basis nodes to regularize
        selected = np.random.choice(self.n_base, min(l1_k, self.n_base), replace=False)
        selected_bases = torch.cat([self.normalized_bases[i] for i in selected], dim=0) # (k, J)
        return self.lambda1 * torch.mean(_l1(selected_bases, self.h))

    # =====================================================
    # R2: Orthogonality
    # =====================================================
    def R2(self, l2_pairs):
        if self.lambda2 == 0 or self.n_base == 1: return torch.zeros(1).to(self.device)
        k = min(l2_pairs, self.n_base * (self.n_base - 1) // 2)
        f1, f2 = [None] * k, [None] * k
        for i in range(k):
            a, b = np.random.choice(self.n_base, 2, replace=False)
            f1[i], f2[i] = self.normalized_bases[a], self.normalized_bases[b]
        return self.lambda2 * torch.mean(torch.abs(_inner_product(torch.cat(f1, dim=0),
                                                                  torch.cat(f2, dim=0),
                                                                  self.h)))

# =============================================================================
# 4. Video Encoder and Estimator Components
# =============================================================================

def _get_convolution_block(in_channels, hidden_channels, out_channels, pool_size=(2, 2, 2)):
    return nn.Sequential(
        nn.AvgPool3d(kernel_size=pool_size, stride=pool_size, padding=0),
        nn.Conv3d(in_channels, hidden_channels, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
        nn.BatchNorm3d(hidden_channels),
        nn.ELU(),
        nn.Conv3d(hidden_channels, out_channels, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
        nn.BatchNorm3d(out_channels),
        nn.ELU()
    )

def _get_estimator_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 1, 1), stride=1,
                  padding=(1, 0, 0)),
        nn.BatchNorm3d(out_channels),
        nn.ELU(),
    )

class Estimator(nn.Module):
    def __init__(self, ):
        super(Estimator, self).__init__()
        self.S = S_VALUE

        self.estimator_blocks = nn.ModuleList([
            _get_estimator_block(64, 64),
            _get_estimator_block(64, 64),
        ])

        self.final = nn.Sequential(
            nn.AdaptiveAvgPool3d((None, self.S, self.S)),
            nn.Conv3d(in_channels=64, out_channels=1, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))
        )

    def forward(self, x, parity):

        for block, t_size in zip(self.estimator_blocks, parity):
            x = F.interpolate(x, scale_factor=(2, 1, 1))
            x = F.pad(x, (0, 0, 0, 0, 0, t_size), mode='replicate')
            x = block(x)

        x = self.final(x)
        x = reduce(x, 'b c t s1 s2 -> b c t', 'mean')[:,-1]
        return x

class CoefficientEstimator(nn.Module):
    """
    Predict coefficients from Encoder features
    Input: (B, 64, T, H, W) - Encoder features
    Output: (B, n_base) - Coefficients
    """
    def __init__(self, in_channels=64, n_base=4, hidden_dim=128, dropout=0.1):
        super().__init__()
        
        # Spatial global pooling (H, W) -> (1, 1)
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        
        # Temporal feature extraction
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Global temporal pooling T -> 1
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        
        # MLP coefficient prediction
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_base),
            nn.Tanh()
        )
    
    def forward(self, x):
        """
        x: (B, C=64, T, H, W)
        return: (B, n_base)
        """
        B = x.size(0)
        
        # Spatial pooling: (B, 64, T, H, W) -> (B, 64, T, 1, 1)
        x = self.spatial_pool(x)  # (B, 64, T, 1, 1)
        
        # Remove spatial dimension: (B, 64, T)
        x = x.squeeze(-1).squeeze(-1)  # (B, 64, T)
        
        # Temporal convolution
        x = self.temporal_conv(x)  # (B, hidden_dim, T)
        
        # Temporal pooling: (B, hidden_dim, T) -> (B, hidden_dim, 1)
        x = self.temporal_pool(x).squeeze(-1)  # (B, hidden_dim)
        
        # Predict coefficients
        coefficients = self.fc(x)  # (B, n_base)
        
        return coefficients

class Encoder(nn.Module):
    """
    Video Encoder
    """
    def __init__(self):
        super().__init__()
        stem  = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=32, kernel_size=(1, 5, 5), stride=1, padding=(0, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ELU()
        )
        self.encoder_blocks = nn.ModuleList([
            stem,
            _get_convolution_block(32, 64, 64, (1, 2, 2)),
            _get_convolution_block(64, 64, 64),
            _get_convolution_block(64, 64, 64),
        ])

    def forward(self, x):
        parity = []
        for i, block in enumerate(self.encoder_blocks):
            x = block(x)
            if i != 0: 
                parity.append(x.size(2) % 2)
        return x, parity

# =============================================================================
# 5. Main Reconstruction Models
# =============================================================================

class BasisReconstruction(nn.Module):
    """
    Complete Basis Reconstruction Model
    Encoder -> Predict coefficients -> Combine with Basis to reconstruct rPPG
    """
    def __init__(self, adafnn: AdaFNN, freeze_basis=True, hidden_dim_estimator=128, dropout_estimator=0.1):
        super().__init__()
        
        # Encoder
        self.encoder = Encoder()

        # estimator
        self.estimator_head = Estimator()
        
        # Coefficient estimator
        self.coefficient_estimator = CoefficientEstimator(
            in_channels=64,
            n_base=adafnn.n_base,
            hidden_dim=hidden_dim_estimator,
            dropout=dropout_estimator
        )
        
        # Save AdaFNN (for basis)
        self.adafnn = adafnn
        
        # Freeze basis parameters
        if freeze_basis:
            for param in self.adafnn.BL.parameters():
                param.requires_grad = False
        
        # Precompute and register normalized bases
        self._precompute_bases()

    def _precompute_bases(self):
        """Precompute normalized basis functions"""
        with torch.no_grad():
            T = self.adafnn.t.unsqueeze(-1)  # (J, 1)
            bases_list = [basis(T).transpose(-1, -2) for basis in self.adafnn.BL]  # list of (1, J)
            bases_tensor = torch.cat(bases_list, dim=0)  # (n_base, J)
            
            # L2 Normalization
            l2_norm = _l2(bases_tensor, self.adafnn.h).detach()  # (n_base, 1)
            normalized_bases = bases_tensor / (l2_norm + 1e-6)  # (n_base, J)
            
            # Register as buffer (not involved in gradient update)
            self.register_buffer('normalized_bases', normalized_bases)
    
    def forward(self, video_clip, epoch=None, total_epochs=None):
        """
        video_clip: (B, C, T, H, W)
        return: rppg (B, J)
        """
        # Encoder extracts features
        feature, parity = self.encoder(video_clip)  # (B, 64, T', H', W')
        
        # Predict coefficients
        coefficients = self.coefficient_estimator(feature)  # (B, n_base)
        
        # Combine with basis to reconstruct rPPG
        # coefficients: (B, n_base), normalized_bases: (n_base, J)
        rppg = torch.matmul(coefficients, self.normalized_bases)  # (B, J)

        # estimator result
        coarse_rppg = self.estimator_head(feature, parity)  # (B,J)

        # feat = torch.mean(feature, dim=[2,3,4])  # (B,64) (unused but kept from original)

        final_rppg = rppg + coarse_rppg
        return final_rppg, rppg, coarse_rppg


class BasisReconstruction_aug(nn.Module):
    """
    Complete Basis Reconstruction Model (With Augmentation / Scaling)
    Encoder -> Predict coefficients -> Combine with Basis to reconstruct rPPG
    """
    def __init__(self, adafnn: AdaFNN, freeze_basis=True, hidden_dim_estimator=128, dropout_estimator=0.1, scales: Optional[List] = None):
        super().__init__()
        
        # Encoder
        self.encoder = Encoder()

        # estimator
        self.estimator_head = Estimator()
        
        # num of basis
        self.n_scales = len(scales) if scales else 1
        
        # Coefficient estimator
        self.coefficient_estimator = CoefficientEstimator(
            in_channels=64,
            n_base=adafnn.n_base * self.n_scales,
            hidden_dim=hidden_dim_estimator,
            dropout=dropout_estimator
        )
        
        # Save AdaFNN (for basis)
        self.adafnn = adafnn
        
        # Freeze basis parameters
        if freeze_basis:
            for param in self.adafnn.BL.parameters():
                param.requires_grad = False
        
        # used for basis augmentation
        self.scales = scales
        
        # Precompute and register normalized bases
        self._precompute_bases()

    def _precompute_bases(self):
        """Precompute normalized basis functions"""
        with torch.no_grad():
            T = self.adafnn.t.unsqueeze(-1)  # (J, 1)
            all_bases = []

            if self.scales:
                for s in self.scales:
                    # T/s -> scaling time axis to change frequency
                    scaled_bases = [
                        basis(T / s).transpose(-1, -2)   # (1, J)
                        for basis in self.adafnn.BL
                    ]
                    all_bases += scaled_bases
            else:
                 # Fallback if no scales provided
                scaled_bases = [basis(T).transpose(-1, -2) for basis in self.adafnn.BL]
                all_bases += scaled_bases

            # concat -> shape (n_base * len(scales), J)
            bases_tensor = torch.cat(all_bases, dim=0)

            # L2 Normalization
            l2_norm = _l2(bases_tensor, self.adafnn.h).detach()  # (n_base, 1)
            normalized_bases = bases_tensor / (l2_norm + 1e-6)  # (n_base, J)
            
            # Register as buffer (not involved in gradient update)
            self.register_buffer('normalized_bases', normalized_bases)
    
    def forward(self, video_clip, epoch=None, total_epochs=None):
        """
        video_clip: (B, C, T, H, W)
        return: rppg (B, J)
        """
        # Encoder extracts features
        feature, parity = self.encoder(video_clip)  # (B, 64, T', H', W')
        
        # Predict coefficients
        coefficients = self.coefficient_estimator(feature)  # (B, n_base)
        
        # Test progressively adding bases for prediction
        if epoch is not None and total_epochs is not None and self.scales:
            n_active = active_scales(self.n_scales, epoch, total_epochs)

            K = self.adafnn.n_base            # Number of bases per scale
            active_dim = n_active * K         # Active coefficient dimensions

            # Create mask
            mask = torch.zeros_like(coefficients)
            mask[:, :active_dim] = 1.0

            coefficients = coefficients * mask
        
        # Combine with basis to reconstruct rPPG
        # coefficients: (B, n_base), normalized_bases: (n_base, J)
        rppg = torch.matmul(coefficients, self.normalized_bases)  # (B, J)

        # estimator result
        coarse_rppg = self.estimator_head(feature, parity)  # (B,J)

        # feat = torch.mean(feature, dim=[2,3,4])  # (B,64)

        final_rppg = rppg
        return final_rppg, rppg, coarse_rppg