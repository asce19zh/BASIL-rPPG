import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
class FeedForward(nn.Module):

    def __init__(self, in_d=1, hidden=[4,4,4], dropout=0.1, activation=F.relu):
        # in_d      : input dimension, integer
        # hidden    : hidden layer dimension, array of integers
        # dropout   : dropout probability, a float between 0.0 and 1.0
        # activation: activation function at each layer
        super().__init__()
        self.sigma = activation
        dim = [in_d] + hidden + [1]
        self.layers = nn.ModuleList([nn.Linear(dim[i-1], dim[i]) for i in range(1, len(dim))])
        self.ln = nn.ModuleList([LayerNorm(k) for k in hidden])
        self.dp = nn.ModuleList([nn.Dropout(dropout) for _ in range(len(hidden))])

    def forward(self, t):
        for i in range(len(self.layers)-1):
            t = self.layers[i](t)
            # skipping connection
            t = t + self.ln[i](t)
            t = self.sigma(t)
            # apply dropout
            t = self.dp[i](t)
        # linear activation at the last layer
        return self.layers[-1](t)


class FourierFeatureEmbedding(nn.Module):
    """
    Fourier Feature Embedding for periodic basis learning.
    Maps time t to [sin(2πf₁t), cos(2πf₁t), sin(2πf₂t), cos(2πf₂t), ...]
    
    This helps the network learn periodic functions more easily.
    For PPG signals, typical heart rate range is 40-250 BPM (0.67-4.17 Hz).
    """
    
    def __init__(self, n_frequencies=32, freq_range=(0.67, 4.17), learnable=True):
        """
        Parameters:
        -----------
        n_frequencies : int
            Number of frequency components
        freq_range : tuple
            (min_freq, max_freq) in Hz for PPG-relevant frequencies
        learnable : bool
            If True, frequencies are learnable parameters
        """
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
        """
        Args:
            t: Time tensor of shape (..., 1)
        Returns:
            Fourier features of shape (..., 2 * n_frequencies)
        """
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
        """
        Parameters:
        -----------
        hidden : list
            Hidden layer dimensions for the MLP
        dropout : float
            Dropout probability
        activation : function
            Activation function
        n_frequencies : int
            Number of Fourier frequency components
        freq_range : tuple
            (min_freq, max_freq) in Hz
        learnable_freq : bool
            Whether frequencies are learnable
        """
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
        """
        Args:
            t: Time tensor of shape (..., 1)
        Returns:
            Basis value of shape (..., 1)
        """
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
        '''
        if self.training:
            # time augmentation
            s = torch.rand(1, device=self.device) * (3.6 - 0.3) + 0.3
            t_aug = T / s
        else:
            # no augmentation during inference
            t_aug = T
        '''
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
    # R1: Sparsity (same as original AdaFNN)
    # =====================================================
    def R1(self, l1_k):
        """
        L1 regularization
        l1_k : number of basis nodes to regularize, integer        
        """
        if self.lambda1 == 0: return torch.zeros(1).to(self.device)
        # sample l1_k basis nodes to regularize
        selected = np.random.choice(self.n_base, min(l1_k, self.n_base), replace=False)
        selected_bases = torch.cat([self.normalized_bases[i] for i in selected], dim=0) # (k, J)
        return self.lambda1 * torch.mean(_l1(selected_bases, self.h))


    # =====================================================
    # R2: Orthogonality (same as original AdaFNN)
    # =====================================================
    def R2(self, l2_pairs):
        """
        L2 regularization
        l2_pairs : number of pairs to regularize, integer  
        """
        if self.lambda2 == 0 or self.n_base == 1: return torch.zeros(1).to(self.device)
        k = min(l2_pairs, self.n_base * (self.n_base - 1) // 2)
        f1, f2 = [None] * k, [None] * k
        for i in range(k):
            a, b = np.random.choice(self.n_base, 2, replace=False)
            f1[i], f2[i] = self.normalized_bases[a], self.normalized_bases[b]
        return self.lambda2 * torch.mean(torch.abs(_inner_product(torch.cat(f1, dim=0),
                                                                  torch.cat(f2, dim=0),
                                                                  self.h)))
