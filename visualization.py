from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import umap.umap_ as umap
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import torch
import torch.nn.functional as F
import os
import numpy as np
import pandas as pd
import config
from pandas.api.types import CategoricalDtype

# Times New Roman으로 전체 폰트 설정
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 12  # 기본 폰트 크기
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] = 10
matplotlib.rcParams['legend.fontsize'] = 10
matplotlib.rcParams['figure.titlesize'] = 13

def print_logs():
    print("<< INFO >>")
    print("SESSION: ", config.SESSION)
    print("Num Class: ", config.NUM_CLASSES)
    print("")
    print("Is LOSO?: ", config.LOSO)
    print("Is Finetuning?: ", config.FINETUNE)
    if config.FINETUNE:
      print("Finetuning Percent: ", config.FINETUNE_PERCENT)
    print("")
    print("Root Directory: ", config.ROOT_DIR)
    print("Data Directory: ", config.DATA_DIR)
    print("Data Folder: ", config.DATA_FOLDER)
    print("\n=====\n")
    
    print("Model Type: ", config.MODEL_TYPE)
    print("\n=====\n")
    
    print("Memo: ", config.MEMO)
    
    print("\n=====\n")

def plot_tsne_umap(embeddings, labels, save_dir, epoch=None, prefix="visual"):
    os.makedirs(save_dir, exist_ok=True)
    tsne_perplexity = max(5, min(15, len(embeddings) - 1))
    if tsne_perplexity >= len(embeddings):
        print(f"[!] Skipping t-SNE: not enough samples (n={len(embeddings)}) for perplexity={tsne_perplexity}")
        return
    tsne = TSNE(n_components=2, perplexity=tsne_perplexity, init="pca",random_state=42)
    umap_model = umap.UMAP(n_components=2, n_neighbors=15, random_state=42)

    tsne_result = tsne.fit_transform(embeddings)
    umap_result = umap_model.fit_transform(embeddings)

    label_names = np.array([config.LABEL.get(int(l), str(l)) for l in labels])
    unique_labels = np.unique(label_names)

    def plot_scatter(result, method):
        plt.figure(figsize=(6, 5))
        for label in unique_labels:
            idx = label_names == label
            plt.scatter(result[idx, 0], result[idx, 1], label=label, alpha=0.7, s=10)
        plt.legend()
        title = f"{method} Visualization"
        if epoch is not None:
            title += f" - Epoch {epoch+1}"
        plt.title(title)
        fname = f"{prefix}_{method.lower()}_epoch_{epoch+1}.png" if epoch is not None else f"{prefix}_{method.lower()}.png"
        plt.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches='tight')
        plt.close()

    plot_scatter(tsne_result, "t-SNE")
    plot_scatter(umap_result, "UMAP")

def plot_confusion_matrix_custom(y_true, y_pred, save_dir, epoch=None):
    os.makedirs(save_dir, exist_ok=True)

    # 유효한 라벨만 필터링
    valid_labels = set(config.LABEL.keys())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = np.isin(y_true, list(valid_labels)) & np.isin(y_pred, list(valid_labels))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    # 혼동 행렬 계산
    cm = confusion_matrix(y_true, y_pred, labels=list(valid_labels), normalize='true')
    cm = np.nan_to_num(cm) * 100  # NaN 제거 + 백분율 변환

    tick_labels = [config.LABEL.get(l, str(l)) for l in valid_labels]

    # 시각화
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=".1f", cmap="Blues", cbar=False,
                xticklabels=tick_labels, yticklabels=tick_labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    title = f"Confusion Matrix - Epoch {epoch+1} (% of True Label)" if epoch is not None else "Confusion Matrix"
    ax.set_title(title)

    fname = f"confusion_epoch_{epoch+1}.png" if epoch is not None else "confusion_matrix.png"
    plt.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches='tight')
    plt.close()


def plot_feature_map_1d(feature_map, save_dir, epoch=None, prefix="fmap"):
    os.makedirs(save_dir, exist_ok=True)
    fmap = feature_map.detach().squeeze(0).cpu().numpy()
    num_channels = min(fmap.shape[0], 8)
    fig, axes = plt.subplots(num_channels, 1, figsize=(10, 2 * num_channels))
    if num_channels == 1:
        axes = [axes]
    for i in range(num_channels):
        axes[i].plot(fmap[i])
        axes[i].set_title(f"Feature Channel {i}")
    fig.suptitle(f"Feature Map (Epoch {epoch+1})" if epoch is not None else "Feature Map")
    plt.tight_layout()
    fname = f"{prefix}_epoch_{epoch+1}.png" if epoch is not None else f"{prefix}.png"
    plt.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.close()

def extract_latent_features(model, dataloader, device):
    model.eval()
    z_all, y_all = [], []
    with torch.no_grad():
        for x, y, m in dataloader:
            x = x.to(device)
            z = model.encoder(x)[1]  # encoder_out[1] is z
            z_all.append(z.cpu())
            y_all.append(y.cpu())
    return torch.cat(z_all).numpy(), torch.cat(y_all).numpy()

def extract_semantic_embeddings(model, ddpm, dataloader, device):
    model.eval()
    ddpm.eval()
    sem_out_all, y_all = [], []
    with torch.no_grad():
        for x, y, m in dataloader:
            x = x.to(device)
            x_hat, down, up, _, t = ddpm(x)  # noise 무시
            decoder_out, fc_out, sem_out = model(x, (x_hat, down, up, t))
            sem_out_all.append(sem_out.cpu())
            y_all.append(y.cpu())
    return torch.cat(sem_out_all).numpy(), torch.cat(y_all).numpy()

def plot_feature_vector_tsne(model, dataloader, device, save_dir, epoch=None, prefix="down1_tsne"):
    model.eval()
    vecs, labels = [], []
    with torch.no_grad():
        for x, y, m in dataloader:
            x = x.to(device)
            feat = model.encoder.down1(x)[0]  # (B, C, T)
            pooled = F.adaptive_avg_pool1d(feat, 1).squeeze(-1)  # (B, C)
            vecs.append(pooled.cpu())
            labels.append(y)
    vecs = torch.cat(vecs).detach().cpu().numpy().reshape(-1, vecs[0].shape[-1])
    labels = torch.cat(labels).numpy()
    plot_tsne_umap(vecs, labels, save_dir=save_dir, epoch=epoch, prefix=prefix)
    
def make_group(sem, mod):
    if sem == 'rest' and mod == 'rest':
        return 'rest_rest'
    elif sem != 'rest' and mod != 'rest':
        return f'{sem}_{mod}'
    else:
        return None  # 잘못된 조합은 제외

def plot_eegpt_feature(latent_z, modality_labels, semantic_class, save_dir, title='t-SNE of Latent Space'):
    if not latent_z:
        print("[!] Warning: latent_z list is empty. Skipping visualization.")
        return
    save_dir = os.path.join(save_dir, 'contrastive_loss')
    os.makedirs(save_dir, exist_ok=True)

    # CPU로 이동 + numpy 변환
    z_np = torch.cat(latent_z, dim=0).cpu().numpy()
    m_np = torch.cat(modality_labels, dim=0).cpu().numpy()
    s_np = torch.cat(semantic_class, dim=0).cpu().numpy()

    # 1. 각 라벨 개수 확인
    unique_y, count_y = np.unique(s_np, return_counts=True)
    unique_m, count_m = np.unique(m_np, return_counts=True)

    print("Semantic class (y) distribution:")
    for u, c in zip(unique_y, count_y):
        print(f"  y={u}: {c} samples")

    print("Modality (m) distribution:")
    for u, c in zip(unique_m, count_m):
        print(f"  m={u}: {c} samples")
    
    # 길이 맞추기
    min_len = min(len(z_np), len(m_np), len(s_np))
    z_np = z_np[:min_len]
    m_np = m_np[:min_len]
    s_np = s_np[:min_len]

    # t-SNE 실행 (2D로 축소)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    z_tsne = tsne.fit_transform(z_np)

    # 데이터프레임 구성
    if config.SESSION == 'all':
      modality_map = {0: 'word', 1: 'sentence', 2: 'visual', 3: 'rest'}
      modality_order = ['word', 'sentence', 'visual', 'rest']
      color_palette = ['#FFD586', '#FFB22C', '#F97A00',
                       '#FFDCDC', '#FFAAAA', '#CD5656',
                       '#9EC6F3', '#6096B4', '#0C359E',
                       '#735557']
    elif config.SESSION == 'word':
      modality_map = {0: 'word', 1: 'rest'}
      modality_order = ['word', 'rest']
      color_palette = ['#FFD586', '#FFDCDC', '#9EC6F3', '#735557']
    elif config.SESSION == 'is':
      modality_map = {0: 'sentence', 1: 'rest'}
      modality_order = ['sentence', 'rest']
      color_palette = ['#FFB22C', '#FFAAAA', '#6096B4', '#735557']
    elif config.SESSION == 'vi':
      modality_map = {0: 'visual', 1: 'rest'}
      modality_order = ['visual', 'rest']
      color_palette = ['#F97A00', '#CD5656', '#0C359E', '#735557']
    semantic_map = {0: 'clock', 1: 'toilet', 2: 'water', 3: 'rest'}
    semantic_order = ['clock', 'toilet', 'water', 'rest']
    
    m_str = np.vectorize(modality_map.get)(m_np)
    s_str = np.vectorize(semantic_map.get)(s_np)

    df = pd.DataFrame({
        'tSNE-1': z_tsne[:, 0],
        'tSNE-2': z_tsne[:, 1],
        'Modality': m_str,
        'Semantic': s_str
    })
    df['Group'] = [make_group(s, m) for s, m in zip(df['Semantic'], df['Modality'])]
    df = df.dropna(subset=['Group'])
    
    # 원하는 정렬 순서 정의
    group_order = []
    for sem in semantic_order:
        if sem == 'rest':
            group_order.append('rest_rest')  # rest는 rest와만 매칭
        else:
            for mod in modality_order:
                if mod != 'rest':            # 나머지 semantic은 rest를 제외한 modality와만 매칭
                    group_order.append(f"{sem}_{mod}")

    # Group 열을 순서가 있는 카테고리형으로 변환
    cat_type = CategoricalDtype(categories=group_order, ordered=True)
    df["Group"] = df["Group"].astype(cat_type)

    # 시각화
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df,
        x='tSNE-1', y='tSNE-2',
        hue='Group',
        palette=color_palette,
        s=30, alpha=0.7
    )

    plt.title(title)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(title='Semantic_Modality', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, 'semantic_modality.png'), dpi=150, bbox_inches='tight')
    plt.close()

def plot_eegpt_modality_feature(latent_z, modality_labels, save_dir, title='t-SNE of Latent Space'):
    if not latent_z:
        print("[!] Warning: latent_z list is empty. Skipping visualization.")
        return
    save_dir = os.path.join(save_dir, 'contrastive_loss')
    os.makedirs(save_dir, exist_ok=True)

    # CPU로 이동 + numpy 변환
    z_np = torch.cat(latent_z, dim=0).cpu().numpy()
    m_np = torch.cat(modality_labels, dim=0).cpu().numpy()

    unique_m, count_m = np.unique(m_np, return_counts=True)
    print("Modality (m) distribution:")
    for u, c in zip(unique_m, count_m):
        print(f"  m={u}: {c} samples")
    
    # 길이 맞추기
    min_len = min(len(z_np), len(m_np))
    z_np = z_np[:min_len]
    m_np = m_np[:min_len]

    # t-SNE 실행 (2D로 축소)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    z_tsne = tsne.fit_transform(z_np)

    # 데이터프레임 구성
    if config.SESSION == 'all':
      modality_map = {0: 'word', 1: 'sentence', 2: 'visual', 3: 'rest'}
      modality_order = ['word', 'sentence', 'visual', 'rest']
    elif config.SESSION == 'word':
      modality_map = {0: 'word', 1: 'rest'}
      modality_order = ['word', 'rest']
    elif config.SESSION == 'is':
      modality_map = {0: 'sentence', 1: 'rest'}
      modality_order = ['sentence', 'rest']
    elif config.SESSION == 'vi':
      modality_map = {0: 'visual', 1: 'rest'}
      modality_order = ['visual', 'rest']
    color_palette_modality = ['#FFD586', '#FFDCDC', '#9EC6F3', '#735557']

    m_str = np.vectorize(modality_map.get)(m_np)
    
    df = pd.DataFrame({
        'tSNE-1': z_tsne[:, 0],
        'tSNE-2': z_tsne[:, 1],
        'Modality': m_str
    })

    cat_type = CategoricalDtype(categories=modality_order, ordered=True)
    df["Modality"] = df["Modality"].astype(cat_type)
    
    # 시각화
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df,
        x='tSNE-1', y='tSNE-2',
        hue='Modality',
        palette=color_palette_modality,
        s=30, alpha=0.7
    )

    plt.title(title)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(title='Semantic_Modality', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, 'modality.png'), dpi=150, bbox_inches='tight')
    plt.close()
    