import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from .bases import AdaFNN, _l2

S_VALUE = 1

def active_scales(n_scales, epoch, total_epochs):
    if epoch < 0.2 * total_epochs:
        return 1
    else:
        return n_scales


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
    從 Encoder 特徵預測係數
    輸入: (B, 64, T, H, W) - Encoder 特徵
    輸出: (B, n_base) - 係數
    """
    def __init__(self, in_channels=64, n_base=4, hidden_dim=128, dropout=0.1):
        super().__init__()
        
        # 空間全局池化 (H, W) -> (1, 1)
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        
        # 時間特徵提取
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 全局時間池化 T -> 1
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        
        # MLP 預測係數
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
        
        # 空間池化: (B, 64, T, H, W) -> (B, 64, T, 1, 1)
        x = self.spatial_pool(x)  # (B, 64, T, 1, 1)
        
        # 移除空間維度: (B, 64, T)
        x = x.squeeze(-1).squeeze(-1)  # (B, 64, T)
        
        # 時間卷積
        x = self.temporal_conv(x)  # (B, hidden_dim, T)
        
        # 時間池化: (B, hidden_dim, T) -> (B, hidden_dim, 1)
        x = self.temporal_pool(x).squeeze(-1)  # (B, hidden_dim)
        
        # 預測係數
        coefficients = self.fc(x)  # (B, n_base)
        
        return coefficients

class Encoder(nn.Module):
    """
    Video Encoder (與參考程式碼相同)
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

class BasisReconstruction(nn.Module):
    """
    完整的 Basis Reconstruction 模型
    Encoder -> 預測係數 -> 與 Basis 結合重建 rPPG
    """
    def __init__(self, adafnn: AdaFNN, freeze_basis=True,hidden_dim_estimator=128, dropout_estimator=0.1):
        super().__init__()
        
        # Encoder
        self.encoder = Encoder()

        #estimator
        self.estimator_head = Estimator()
        
        # 係數預測器
        self.coefficient_estimator = CoefficientEstimator(
            in_channels=64,
            n_base=adafnn.n_base,
            hidden_dim=hidden_dim_estimator,
            dropout=dropout_estimator
        )
        
        # 保存 AdaFNN（用於 basis）
        self.adafnn = adafnn
        
        # 凍結 basis 參數
        if freeze_basis:
            for param in self.adafnn.BL.parameters():
                param.requires_grad = False
        
        # 預先計算並註冊 normalized bases
        self._precompute_bases()


    
    def _precompute_bases(self):
        """預先計算正規化的 basis functions"""
        with torch.no_grad():
            T = self.adafnn.t.unsqueeze(-1)  # (J, 1)
            bases_list = [basis(T).transpose(-1, -2) for basis in self.adafnn.BL]  # list of (1, J)
            bases_tensor = torch.cat(bases_list, dim=0)  # (n_base, J)
            
            # L2 正規化
            l2_norm = _l2(bases_tensor, self.adafnn.h).detach()  # (n_base, 1)
            normalized_bases = bases_tensor / (l2_norm + 1e-6)  # (n_base, J)
            
            # 註冊為 buffer（不參與梯度更新）
            self.register_buffer('normalized_bases', normalized_bases)
    
    def forward(self, video_clip, epoch=None, total_epochs=None):
        """
        video_clip: (B, C, T, H, W)
        return: rppg (B, J)
        """
        # Encoder 提取特徵
        feature, parity = self.encoder(video_clip)  # (B, 64, T', H', W')
        
        # 預測係數
        coefficients = self.coefficient_estimator(feature)  # (B, n_base)
        
        # 與 basis 結合重建 rPPG
        # coefficients: (B, n_base), normalized_bases: (n_base, J)
        rppg = torch.matmul(coefficients, self.normalized_bases)  # (B, J)

        #estimator result
        coarse_rppg = self.estimator_head(feature, parity)  # (B,J)

        feat = torch.mean(feature, dim=[2,3,4])  # (B,64)

        final_rppg = rppg + coarse_rppg
        #final_rppg =  (rppg + coarse_rppg ) / 2  # wei divide by 2
        return final_rppg, rppg, coarse_rppg



class BasisReconstruction_aug(nn.Module):
    """
    完整的 Basis Reconstruction 模型
    Encoder -> 預測係數 -> 與 Basis 結合重建 rPPG
    """
    def __init__(self, adafnn: AdaFNN, freeze_basis=True,hidden_dim_estimator=128, dropout_estimator=0.1, scales: list | None = None):
        super().__init__()
        
        # Encoder
        self.encoder = Encoder()

        #estimator
        self.estimator_head = Estimator()
        
        #num of basis
        self.n_scales = len(scales)
        # 係數預測器
        self.coefficient_estimator = CoefficientEstimator(
            in_channels=64,
            n_base=adafnn.n_base*self.n_scales,
            hidden_dim=hidden_dim_estimator,
            dropout=dropout_estimator
        )
        
        # 保存 AdaFNN（用於 basis）
        self.adafnn = adafnn
        
        # 凍結 basis 參數
        if freeze_basis:
            for param in self.adafnn.BL.parameters():
                param.requires_grad = False
        # used for basis augmentation
        self.scales = scales
        # 預先計算並註冊 normalized bases
        self._precompute_bases()

    
    def _precompute_bases(self):
        """預先計算正規化的 basis functions"""
        with torch.no_grad():
            T = self.adafnn.t.unsqueeze(-1)  # (J, 1)
            all_bases = []

            for s in self.scales:
                # T/s → scaling time axis to change frequency
                scaled_bases = [
                    basis(T / s).transpose(-1, -2)   # (1, J)
                    for basis in self.adafnn.BL
                ]
                all_bases += scaled_bases

            # concat → shape (n_base * len(scales), J)
            bases_tensor = torch.cat(all_bases, dim=0)

            # L2 正規化
            l2_norm = _l2(bases_tensor, self.adafnn.h).detach()  # (n_base, 1)
            normalized_bases = bases_tensor / (l2_norm + 1e-6)  # (n_base, J)
            
            # 註冊為 buffer（不參與梯度更新）
            self.register_buffer('normalized_bases', normalized_bases)
    
    #def forward(self, video_clip):
    def forward(self, video_clip, epoch=None, total_epochs=None):
        """
        video_clip: (B, C, T, H, W)
        return: rppg (B, J)
        """
        # Encoder 提取特徵
        feature, parity = self.encoder(video_clip)  # (B, 64, T', H', W')
        
        # 預測係數
        coefficients = self.coefficient_estimator(feature)  # (B, n_base)
        
        
        #測試逐步加入bases預測
        if epoch is not None and total_epochs is not None:
            n_active = active_scales(self.n_scales, epoch, total_epochs)

            K = self.adafnn.n_base            # 每個 scale 的 basis 數
            active_dim = n_active * K         # 啟用的 coefficient 維度

            # 建立 mask
            mask = torch.zeros_like(coefficients)
            mask[:, :active_dim] = 1.0

            coefficients = coefficients * mask
        
        # 與 basis 結合重建 rPPG
        # coefficients: (B, n_base), normalized_bases: (n_base, J)
        rppg = torch.matmul(coefficients, self.normalized_bases)  # (B, J)

        #estimator result
        coarse_rppg = self.estimator_head(feature, parity)  # (B,J)

        feat = torch.mean(feature, dim=[2,3,4])  # (B,64)

        final_rppg = rppg
        return final_rppg, rppg, coarse_rppg