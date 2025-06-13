# models/projection.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import config
import math

class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        # projection shortcut (dimension match)
        self.shortcut = nn.Linear(input_dim, output_dim)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.fc1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        out = F.relu(out)
        
        out = self.fc3(out)
        
        return out + identity

def extract_latent(eegpt: nn.Module, projection: nn.Module, X: torch.Tensor):
    eegpt.eval()
    feats = eegpt(X)  # (batch, N, 1, 512)
    feats = feats[:, :, 0, :]  # (batch, N, 512)
    feats = feats.reshape(-1, feats.size(-1))  # (batch*N, 512)
    latent_z = projection(feats)  # (batch*N, 128)
    return latent_z

def modality_aware_nt_xent_loss(z, labels, modalities, temperature=0.07, alpha=0.7):
    """
    z: Tensor of shape [N, d]
    labels: Tensor of shape [N], indicating semantic class
    modalities: Tensor of shape [N], indicating modality (e.g., 0 for EEG, 1 for MEG)
    temperature: Temperature parameter for softmax (0.1에서 0.07로 감소하여 더 민감하게)
    alpha: Weight for modality-aware positive pairs (0.5에서 0.7로 증가하여 모달리티 간 균형 개선)
    """
    z = F.normalize(z, dim=1)
    
    # Compute similarity matrix
    sim_matrix = torch.matmul(z, z.T)  # [N, N]
    sim_matrix = sim_matrix / temperature
    
    # Create masks for different types of positive pairs
    labels = labels.contiguous().view(-1, 1)
    modalities = modalities.contiguous().view(-1, 1)
    
    # Same class, same modality
    same_class_same_mod = torch.eq(labels, labels.T).float() * torch.eq(modalities, modalities.T).float()
    same_class_same_mod.fill_diagonal_(0)
    
    # Same class, different modality
    same_class_diff_mod = torch.eq(labels, labels.T).float() * (1 - torch.eq(modalities, modalities.T).float())
    
    # Combined positive pairs mask
    pos_mask = same_class_same_mod + alpha * same_class_diff_mod
    
    # For numerical stability
    logits = sim_matrix - torch.max(sim_matrix, dim=1, keepdim=True)[0].detach()
    
    # Compute loss
    exp_logits = torch.exp(logits) * (1 - torch.eye(z.shape[0], device=z.device))
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
    
    mean_log_prob_pos = (pos_mask * log_prob).sum(1) / (pos_mask.sum(1) + 1e-8)
    
    loss = -mean_log_prob_pos.mean()
    return loss

def semantic_contrastive_loss(z, labels, modalities, temperature=0.07, cross_modal_weight=None):
    """Enhanced supervised contrastive loss emphasizing cross-modality pairs."""
    if cross_modal_weight is None:
        cross_modal_weight = config.CONTRASTIVE_CROSS_MODAL_WEIGHT

    z = F.normalize(z, dim=1)
    sim = torch.matmul(z, z.T) / temperature
    sim = sim - torch.max(sim, dim=1, keepdim=True)[0].detach()

    labels = labels.view(-1, 1)
    modalities = modalities.view(-1, 1)

    # positive pairs: same class
    pos_mask = torch.eq(labels, labels.T).float()
    pos_mask.fill_diagonal_(0)

    # emphasize cross modality pairs
    cross_mask = torch.ne(modalities, modalities.T).float()
    pos_mask = pos_mask * (1 + cross_modal_weight * cross_mask)

    exp_sim = torch.exp(sim) * (1 - torch.eye(z.size(0), device=z.device))
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
    mean_log_prob_pos = (pos_mask * log_prob).sum(1) / (pos_mask.sum(1) + 1e-8)
    return -mean_log_prob_pos.mean()

class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, device):
        super(CenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))

    def forward(self, features, labels):
        batch_size = features.size(0)
        centers_batch = self.centers[labels]  # [B, D]
        loss = (features - centers_batch).pow(2).sum() / batch_size
        return loss

class ArcFaceHead(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # ArcFace 파라미터 설정
        self.margin = config.ARCFACE_PARAMS['margin']
        self.scale = config.ARCFACE_PARAMS['scale']
        
        # 미리 계산해두기
        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
    def forward(self, features, labels):
        # L2 정규화
        features = F.normalize(features, p=2, dim=1)
        weights = F.normalize(self.weight, p=2, dim=1)
        
        # 코사인 유사도 계산
        cosine = F.linear(features, weights)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        # ArcFace 마진 적용
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > 0, phi, cosine)
        
        # 원-핫 인코딩
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        # 최종 출력
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        
        return output

class SemanticDecoder(nn.Module):
    def __init__(self, latent_dim: int, num_classes: int, hidden_dim: int = 256):
        super().__init__()
        self.class_embeddings = nn.Embedding(num_classes, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.class_embeddings.weight, mean=0.0, std=0.02)
        
    def forward(self, z, labels):
        # Get target semantic embeddings
        target_emb = self.class_embeddings(labels)
        
        # Decode latent representation
        pred_emb = self.decoder(z)
        
        return pred_emb, target_emb

def semantic_reconstruction_loss(pred_emb, target_emb):
    """
    Compute MSE loss between predicted and target semantic embeddings
    """
    return F.mse_loss(pred_emb, target_emb)
