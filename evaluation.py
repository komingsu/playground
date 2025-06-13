from models.diffe import *
from utils import *
import config

import pickle
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from ema_pytorch import EMA
from tqdm import tqdm
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
    accuracy_score
)

# set the device to use for evaluation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# create an argument parser for the data loader path
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, help='path to the data loader file')
parser.add_argument('--data_loader_path', type=str, help='path to the data loader file')

n_T = 1000
ddpm_dim = 128
encoder_dim = 256
fc_dim = 512
# Define model
num_classes = config.NUM_CLASSES
channels = 64

encoder = Encoder(in_channels=channels, dim=encoder_dim).to(device)
decoder = Decoder(
    in_channels=channels, n_feat=ddpm_dim, encoder_dim=encoder_dim
).to(device)
fc = LinearClassifier(encoder_dim, fc_dim, emb_dim=num_classes).to(device)
diffe = DiffE(encoder, decoder, fc).to(device)

# load the data loader from the file
args = parser.parse_args()
# load the pre-trained model from the file
diffe.load_state_dict(torch.load(args.model_path))

# load the data loader from the file
with open(args.data_loader_path, 'rb') as f:
    data_loader = pickle.load(f)

def evaluate_with_ablation(model, test_loader, device):
    """
    각 loss 컴포넌트를 개별적으로 비활성화하면서 모델 성능을 평가합니다.
    """
    results = {}
    
    # 기본 설정 (모든 loss 활성화)
    original_settings = {
        'USE_CENTER_LOSS': config.USE_CENTER_LOSS,
        'USE_CONTRASTIVE_LOSS': config.USE_CONTRASTIVE_LOSS,
        'USE_SEMANTIC_RECONSTRUCTION': config.USE_SEMANTIC_RECONSTRUCTION,
        'USE_ARCFACE_CLASSIFIER': config.USE_ARCFACE_CLASSIFIER,
        'USE_FEATURE_MATCHING': config.USE_FEATURE_MATCHING
    }
    
    # 각 loss 컴포넌트별 ablation 실험
    ablation_settings = {
        'baseline': original_settings,
        'no_center_loss': {**original_settings, 'USE_CENTER_LOSS': False},
        'no_contrastive_loss': {**original_settings, 'USE_CONTRASTIVE_LOSS': False},
        'no_semantic_reconstruction': {**original_settings, 'USE_SEMANTIC_RECONSTRUCTION': False},
        'no_arcface': {**original_settings, 'USE_ARCFACE_CLASSIFIER': False},
        'no_feature_matching': {**original_settings, 'USE_FEATURE_MATCHING': False}
    }
    
    for setting_name, settings in ablation_settings.items():
        # 설정 적용
        for key, value in settings.items():
            setattr(config, key, value)
        
        # 평가 수행
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for x, y, m in test_loader:
                x, y, m = x.to(device), y.to(device), m.to(device)
                outputs = model(x)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        
        # 메트릭 계산
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        results[setting_name] = {
            'accuracy': accuracy,
            'f1_score': f1
        }
        
        print(f"\n=== {setting_name} ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
    
    # 원래 설정으로 복구
    for key, value in original_settings.items():
        setattr(config, key, value)
    
    return results

diffe.eval()
with torch.no_grad():
    labels = np.arange(0, num_classes)
    Y = []
    Y_hat = []
    for x, y in data_loader:
        x, y = x.to(device), y.type(torch.LongTensor).to(device)
        encoder_out = diffe.encoder(x)
        y_hat = diffe.fc(encoder_out[1])
        y_hat = F.softmax(y_hat, dim=1)

        Y.append(y.detach().cpu())
        Y_hat.append(y_hat.detach().cpu())

    # List of tensors to tensor to numpy
    Y = torch.cat(Y, dim=0).numpy()  # (N, )
    Y_hat = torch.cat(Y_hat, dim=0).numpy()  # (N, 13): has to sum to 1 for each row

    # Accuracy and Confusion Matrix
    accuracy = top_k_accuracy_score(Y, Y_hat, k=1, labels=labels)

    print(f'Test accuracy: {accuracy:.2f}%')

    # Evaluate with ablation
    ablation_results = evaluate_with_ablation(diffe, data_loader, device)
    print("\nAblation Results:")
    for setting, metrics in ablation_results.items():
        print(f"{setting}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")