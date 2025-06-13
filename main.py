from models import *
from utils import *
from visualization import *

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
)

from torch.utils.tensorboard import SummaryWriter
import os
import csv
import sys

import config

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import pytz
from datetime import datetime
from sklearn.metrics import precision_score

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.loss")

# Evaluate function
def evaluate(encoder, fc, generator, device):
  # 라벨 인덱스
  labels = np.arange(0, config.NUM_CLASSES)
  Y = []
  Y_hat = []
  for x, y, m in generator:
      x, y, m = x.to(device), y.type(torch.LongTensor).to(device), m.to(device)
      encoder_out = encoder(x)
      y_hat = fc(encoder_out[1])
      y_hat = F.softmax(y_hat, dim=1)

      Y.append(y.detach().cpu())
      Y_hat.append(y_hat.detach().cpu())

  Y = torch.cat(Y, dim=0).numpy()
  Y_hat = torch.cat(Y_hat, dim=0).numpy()

  accuracy = top_k_accuracy_score(Y, Y_hat, k=1, labels=labels)
  f1 = f1_score(Y, Y_hat.argmax(axis=1), average="macro", labels=labels)
  recall = recall_score(Y, Y_hat.argmax(axis=1), average="macro", labels=labels)
  precision = precision_score(Y, Y_hat.argmax(axis=1), average="macro", labels=labels)
  auc = roc_auc_score(Y, Y_hat, average="macro", multi_class="ovo", labels=labels)
  
  
  metrics = {
      "accuracy": accuracy,
      "f1": f1,
      "recall": recall,
      "precision": precision,
      "auc": auc,
  }
  return metrics

def train(args):
  # 장치 설정
  device = args.device
  subjects = args.subjects
  
  if config.LOSO:
    test_sub_idx = args.test_sub_idx
  batch_size = config.BATCH_SIZE
  batch_size2 = config.BATCH_SIZE2
  seed = config.SEED
  random.seed(seed)
  torch.manual_seed(seed)

  # 한국 시간대 설정
  kst = pytz.timezone('Asia/Seoul')
  current_time_kst = datetime.now(kst)
  formatted_time_kst = current_time_kst.strftime("%y%m%d_%H%M")
  
  root_dir = config.ROOT_DIR
  data_dir = config.DATA_DIR
  data_folder = config.DATA_FOLDER
  
  # 모델 저장 디렉토리 세팅
  model_dir = ''
  if config.LOSO:
    model_dir_tmp = "LOSO"
    if config.FINETUNE:
      finetune_value = int(round((1 - config.FINETUNE_PERCENT) * 100))
      model_dir_tmp += f'_finetune_{finetune_value}'    
    model_dir = os.path.join(model_dir, model_dir_tmp)
  else:
    model_dir = os.path.join(model_dir, 'Dependent')

  if config.LOSO:
    model_dir = os.path.join(model_dir, f'test_subject_{subjects[test_sub_idx]}')
  else:
    model_dir = os.path.join(model_dir, f'sub{subjects[0]}')
  model_dir = os.path.join(model_dir, config.MODEL_TYPE)
  model_dir = os.path.join(model_dir, config.SESSION)
  model_dir_tmp = str(formatted_time_kst)
  model_dir = os.path.join(model_dir, model_dir_tmp)
  os.makedirs(f'{root_dir}/DiffE/{config.NUM_CLASSES}class/{model_dir}', exist_ok=True)
  
  metrics_csv_path = f'{root_dir}/DiffE/{config.NUM_CLASSES}class/{model_dir}/metrics.csv'
  writer = SummaryWriter(log_dir=f"{root_dir}/DiffE/{config.NUM_CLASSES}class/{model_dir}")
  if not os.path.exists(metrics_csv_path):
      with open(metrics_csv_path, mode="w", newline="") as f:
          csv.writer(f).writerow(["Epoch", "Accuracy", "F1", "Recall", "Precision", "AUC"])
  
  log_file = open(f'{root_dir}/DiffE/{config.NUM_CLASSES}class/{model_dir}/{formatted_time_kst}.log', 'w')
  sys.stdout = log_file
  sys.stderr = log_file
  print_logs()
  
  # Load data
  X, Y, M = load_data(root_dir=root_dir, data_dir=data_dir, data_folder=data_folder, subjects=subjects, session=config.SESSION, is_datacollection=True)
  # X2, Y2 = load_data(root_dir=root_dir, data_dir=data_dir, data_folder=data_folder, subjects=subjects, session=config.SESSION, is_datacollection=False)
  # Dataloader
  if config.LOSO:
    # if config.FINETUNE:
    #     train_loader, fine_tuning_loader, test_loader = get_dataloader(
    #         X, Y, batch_size, batch_size2, test_sub_idx=test_sub_idx, seed=seed, shuffle=True
    #     )
    # else:
      train_loader, test_loader = get_dataloader(
          X, Y, M, batch_size, batch_size2, test_sub_idx=test_sub_idx, seed=seed, shuffle=True
      )
  else:
      train_loader, test_loader = get_dataloader(
          X, Y, M, batch_size, batch_size2, seed=seed, shuffle=True
      )

  # Define model
  channels = X.shape[1]
  print("\n=====\n")
  print("Random Seed: ", seed)
  print("subject: ", subjects)
  print("device: ", device)
  print("X shape: ", X.shape)
  print("Y shape: ", Y.shape)
  print("M shape: ", M.shape)

  n_T = 1000
  ddpm_dim = 128
  if config.IS_BASELINE: encoder_dim = 256
  else:  encoder_dim = 512
  fc_dim = 512

  ddpm_model = ConditionalUNet(in_channels=channels, n_feat=ddpm_dim).to(device)
  ddpm = DDPM(nn_model=ddpm_model, betas=(1e-6, 1e-2), n_T=n_T, device=device).to(device)
  encoder = Encoder(in_channels=channels, dim=encoder_dim).to(device)
  decoder = Decoder(in_channels=channels, n_feat=ddpm_dim, encoder_dim=encoder_dim).to(device)
  fc = LinearClassifier(encoder_dim, fc_dim, emb_dim=config.NUM_CLASSES).to(device)
  diffe = DiffE(encoder, decoder, fc).to(device)

  if not config.IS_BASELINE:
      # (1) 채널 인덱스 준비
      channel_indices = match_channels(config.channels_names, config.use_channels_names)
      channel_indices = torch.tensor(channel_indices).to(device)  # GPU에 올려서 빠르게
      num_channels = len(channel_indices)

      # EEGPT 모델 구조 선언
      eegpt_teacher = EEGTransformer(
          img_size=(num_channels, 1250),  # 58채널짜리 input을 받기 때문에 img_size[0]=58
          patch_size=16,  # 혹은 네가 쓰는 값
          embed_dim=512,  # DiffE encoder output과 맞춰야 해 (예: 512)
          depth=6,  # 레이어 수 (네 pretrained 모델에 맞춰야 함)
          num_heads=8,  # 헤드 수
          patch_module=PatchEmbed,  # 기본
      )

      # Pretrained weight 로드
      pretrained_path = "/exHDD/eyko/concept/EEGPT/checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt"
      full_state = torch.load(pretrained_path, map_location='cpu', weights_only=False)
      if 'state_dict' in full_state:
          full_state = full_state['state_dict']

      # EEGTransformer 모델에 맞게 key filtering
      eegpt_state = {k.replace('model.', ''): v for k, v in full_state.items() if 'model.' in k}
      missing_keys, unexpected_keys = eegpt_teacher.load_state_dict(eegpt_state, strict=False)

      print(f"✅ Pretrained EEGPT loaded as teacher.")

      # 학습 중에는 freeze
      eegpt_teacher = eegpt_teacher.to(device)
      
      if config.FINETUNE:
          eegpt_teacher.train()
          for p in eegpt_teacher.parameters():
                p.requires_grad = True
      else:
          eegpt_teacher.eval()
          for p in eegpt_teacher.parameters():
              p.requires_grad = False
          if config.USE_Modality:
              projection_head = ProjectionHead(input_dim=512, output_dim=128).to(device)
              arcface_head = ArcFaceHead(in_features=128, out_features=config.NUM_CLASSES).to(device)
              for p in projection_head.parameters():
                  p.requires_grad = True
              for p in arcface_head.parameters():
                  p.requires_grad = True
              projection_optimizer = torch.optim.Adam(projection_head.parameters(), lr=1e-3)
              arcface_optimizer = torch.optim.Adam(arcface_head.parameters(), lr=1e-3)
              center_criterion = CenterLoss(num_classes=config.NUM_CLASSES, feat_dim=128, device=device)
              center_optimizer = torch.optim.SGD(center_criterion.parameters(), lr=0.5)

  
  print("ddpm size: ", sum(p.numel() for p in ddpm.parameters()))
  print("encoder size: ", sum(p.numel() for p in encoder.parameters()))
  print("decoder size: ", sum(p.numel() for p in decoder.parameters()))
  print("fc size: ", sum(p.numel() for p in fc.parameters()))

  criterion = nn.L1Loss()
  criterion_class = nn.MSELoss()

  # Loss 가중치 설정
  alpha = config.LOSS_WEIGHTS['alpha']  # classification loss 가중치
  beta = config.LOSS_WEIGHTS['beta']    # contrastive/feature matching loss 가중치
  center_lambda = config.LOSS_WEIGHTS['beta_center']  # Center loss 가중치
  gamma = config.LOSS_WEIGHTS['gamma']  # semantic reconstruction loss 가중치

  # Define optimizer
  base_lr, lr = 9e-5, 1.5e-3
  optim1 = optim.RMSprop(ddpm.parameters(), lr=base_lr)
  if config.IS_BASELINE:
      optim2 = optim.RMSprop(diffe.parameters(), lr=base_lr)
  else:
      if config.FINETUNE:
          optim2 = optim.RMSprop([
                  {'params': diffe.parameters(), 'lr': base_lr},
                  {'params': eegpt_teacher.parameters(), 'lr': base_lr * 0.1},  # EEGPT는 10배 작게
              ])
      else:
          optim2 = optim.RMSprop(diffe.parameters(), lr=base_lr)

  # Gradient clipping 설정
  max_grad_norm = 1.0

  # EMAs
  fc_ema = EMA(diffe.fc, beta=0.95, update_after_step=100, update_every=10,)

  step_size = 150
  scheduler1 = optim.lr_scheduler.CyclicLR(
    optimizer=optim1,
    base_lr=base_lr,
    max_lr=lr,
    step_size_up=step_size,
    mode="exp_range",
    cycle_momentum=False,
    gamma=0.9998,
  )
  scheduler2 = optim.lr_scheduler.CyclicLR(
    optimizer=optim2,
    base_lr=base_lr,
    max_lr=lr,
    step_size_up=step_size,
    mode="exp_range",
    cycle_momentum=False,
    gamma=0.9998,
  )
  
  # step_size = 10
  # scheduler1 = optim.lr_scheduler.CosineAnnealingLR(
  #   optimizer=optim1, T_max=step_size, eta_min=1e-6
  # )
  # scheduler2 = optim.lr_scheduler.CosineAnnealingLR(
  #   optimizer=optim2, T_max=step_size, eta_min=1e-6
  # )
  
  # scheduler1 = optim.lr_scheduler.CosineAnnealingWarmRestarts(
  #   optimizer=optim1,
  #   T_0=30,       # 첫 번째 주기(epoch)의 길이 (31 에폭 근처에서 재시작 유도)
  #   T_mult=1,     # 다음 주기의 길이를 두 배로 증가시킴
  #   eta_min=1e-6  # 학습률의 최소값
  # )
  # scheduler2 = optim.lr_scheduler.CosineAnnealingWarmRestarts(
  #   optimizer=optim2,
  #   T_0=30,       # 첫 번째 주기(epoch)의 길이 (31 에폭 근처에서 재시작 유도)
  #   T_mult=1,     # 다음 주기의 길이를 두 배로 증가시킴
  #   eta_min=1e-6  # 학습률의 최소값
  # )
  
  # Train & Evaluate
  num_epochs = config.EPOCHS
  test_period = 1
  start_test = test_period

  best_acc = 0
  best_f1 = 0
  best_recall = 0
  best_precision = 0
  best_auc = 0

  all_latents = []
  all_modalities = []
  all_semantic = []
  
  with tqdm(
    total=num_epochs, desc=f"Method ALL - Processing subject {subjects}"
  ) as pbar:
    for epoch in range(num_epochs):
      ddpm.train()
      diffe.train()
      ############################## Train ###########################################
      for x, y, m in train_loader:
        # print("Train batch x shape:", x.shape)  # 추가
        x, y, m = x.to(device), y.type(torch.LongTensor).to(device), m.to(device)
        y_cat = F.one_hot(y, num_classes=config.NUM_CLASSES).float().to(device)

        # --- DDPM update (64채널 그대로) ---
        optim1.zero_grad()
        x_hat, down, up, noise, t = ddpm(x)
        loss_ddpm = F.l1_loss(x_hat, x, reduction="none")
        loss_ddpm.mean().backward()
        optim1.step()
        ddpm_out = x_hat, down, up, t

        if not config.IS_BASELINE:
            # --- 잘라내기 ---
            x_diffe = torch.index_select(x, dim=1, index=channel_indices)
            if config.FINETUNE:
                z_pretrain = eegpt_teacher(x_diffe)
            else:
                # 1. EEGPT → Projection → Latent
                if config.USE_Modality:
                    feats = eegpt_teacher(x_diffe.to(device))  # (batch, N, 1, 512)
                    feats = feats[:, :, 0, :]                  # (batch, N, 512)
                    batch, N, D = feats.shape                  # batch, N, 512
                    feats = feats.reshape(-1, D)               # (batch*N, 512)
                    latent_z = projection_head(feats)          # (batch*N, 128)

                    Y = y.to(device)
                    M = m.to(device)
                    # Y, M shape: (batch,) → (batch*N,)
                    Y = Y.unsqueeze(1).repeat(1, N).reshape(-1)
                    M = M.unsqueeze(1).repeat(1, N).reshape(-1)

                    loss_contrastive = modality_aware_nt_xent_loss(latent_z, Y, M)
                else:
                    # --- EEGPT Teacher (58채널 input) ---
                    with torch.no_grad():
                        z_pretrain_all = eegpt_teacher(x_diffe)  # (batch, 78, 1, 512)
                        z_pretrain = z_pretrain_all[:, -1, 0, :]  # 🔥 마지막 Summary Token만 가져오기
                    # print("eegpt_teacher(x_diffe) output shape:", z_pretrain.shape)

        # --- DiffE update ---
        optim2.zero_grad()
        if config.IS_BASELINE:
            decoder_out, fc_out, encoder_out = diffe(x, x, ddpm_out)
            loss_gap = criterion(decoder_out, loss_ddpm.detach())
            loss_c = criterion_class(fc_out, y_cat)
            loss = loss_gap
        else:
            # DiffE forward
            decoder_out, fc_out, encoder_out = diffe(x, x_diffe, ddpm_out)

            # Feature Matching Loss
            if not config.USE_Modality:
                # z_diff: DiffE encoder가 만든 latent
                z_diff = encoder_out[1]  # DiffE에서 나온 encoder output 바로 쓰기
                # Feature Matching Loss
                if not config.FINETUNE:
                    loss_feature = F.mse_loss(z_diff, z_pretrain)

            # 기존 loss
            loss_gap = criterion(decoder_out, loss_ddpm.detach())
            loss_c = criterion_class(fc_out, y_cat)

            # 최종 loss
            if config.FINETUNE:
                loss = loss_gap
            else:
                if config.USE_Modality:
                    # ArcFace loss 추가
                    arcface_out = arcface_head(latent_z, Y)
                    loss_arcface = F.cross_entropy(arcface_out, Y)
                    # Center loss 추가
                    loss_center = center_criterion(latent_z, Y)
                    
                    # 모든 loss를 합산
                    loss = loss_gap
                    if config.USE_ARCFACE_CLASSIFIER:
                        loss += alpha * loss_c
                    if config.USE_CONTRASTIVE_LOSS:
                        loss += beta * loss_contrastive
                    if config.USE_CENTER_LOSS:
                        loss += center_lambda * loss_center
                    if config.USE_SEMANTIC_RECONSTRUCTION:
                        loss += gamma * loss_arcface
                    
                    # 각 모듈별 optimizer 업데이트
                    optim2.zero_grad()
                    projection_optimizer.zero_grad()
                    arcface_optimizer.zero_grad()
                    center_optimizer.zero_grad()
                    
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(diffe.parameters(), max_grad_norm)
                    if config.FINETUNE:
                        torch.nn.utils.clip_grad_norm_(eegpt_teacher.parameters(), max_grad_norm)
                    
                    optim2.step()
                    projection_optimizer.step()
                    arcface_optimizer.step()
                    center_optimizer.step()
                else:
                    loss = loss_gap
            
        scheduler1.step()
        scheduler2.step()
        fc_ema.update()

      if (epoch + 1) % config.EPOCHS == 0:
        viz_dir = f"{root_dir}/DiffE/{config.NUM_CLASSES}class/{model_dir}/visual"
      
        # 1. Feature embedding 시각화
        z_feat, z_labels = extract_latent_features(diffe, test_loader, device)
        plot_tsne_umap(z_feat, z_labels, save_dir=os.path.join(viz_dir, 'feature_embedding'), epoch=epoch)
        plot_feature_vector_tsne(
                                  model=diffe, 
                                  dataloader=test_loader,
                                  device=device,
                                  save_dir=os.path.join(viz_dir, 'feature_vector'),
                                  epoch=epoch,
                                  prefix="down1_tsne"
                              )

        # 2. Confusion matrix
        y_true, y_pred = [], []
        diffe.eval()
        with torch.no_grad():
            for x_val, y_val, m_val in test_loader:
                x_val = x_val.to(device)
                pred = fc_ema(diffe.encoder(x_val)[1])
                y_hat = pred.argmax(dim=1).cpu()
                y_true.append(y_val)
                y_pred.append(y_hat)
        plot_confusion_matrix_custom(
            torch.cat(y_true), torch.cat(y_pred), save_dir=os.path.join(viz_dir, 'matrix'), epoch=epoch
        )

        # 3. 중간 feature map 시각화
        sample_x, _, _ = next(iter(test_loader))
        sample_x = sample_x.to(device)
        fmap = diffe.encoder.down1(sample_x)[0]
        plot_feature_map_1d(fmap, save_dir=os.path.join(viz_dir, 'feature_map'), epoch=epoch)
        
        if not config.IS_BASELINE and not config.FINETUNE and config.USE_Modality:
          if len(all_latents) > 0:
              plot_eegpt_feature(all_latents, all_modalities, all_semantic, viz_dir, title="t-SNE of EEG latent z")
              plot_eegpt_modality_feature(all_latents, all_modalities, viz_dir, title="t-SNE of EEG latent z by modality")

      ############################## Fine-tuning ##############################
      # Fine-tuning
      # if config.FINETUNE:
      #   for x, y in fine_tuning_loader:
      #     x, y = x.to(device), y.type(torch.LongTensor).to(device)
      #     y_cat = F.one_hot(y, num_classes=config.NUM_CLASSES).type(torch.FloatTensor).to(device)
      #     optim2.zero_grad()
      #     x_hat, down, up, noise, t = ddpm(x)  # DDPM을 통해 ddpm_out을 얻음
      #     ddpm_out = x_hat, down, up, t
      #     decoder_out, fc_out = diffe(x, ddpm_out)

      #     loss_c = criterion_class(fc_out, y_cat)    
      #     loss_c.backward()
      #     optim2.step()
      
      ############################## Test ###########################################
      with torch.no_grad():
        if epoch > start_test:
          test_period = 1
        if epoch % test_period == 0:
          ddpm.eval()
          diffe.eval()
          
          # evaluate 함수에서 confusion matrix, t-SNE 시각화를 epoch 단위로 저장
          metrics = evaluate(diffe.encoder, fc_ema, test_loader, device)

          acc = metrics["accuracy"]
          f1 = metrics["f1"]
          recall = metrics["recall"]
          precision = metrics["precision"]
          auc = metrics["auc"]

          writer.add_scalar("Accuracy", acc * 100, epoch)
          writer.add_scalar("F1-score", f1 * 100, epoch)
          writer.add_scalar("Recall", recall * 100, epoch)
          writer.add_scalar("Precision", precision * 100, epoch)
          writer.add_scalar("AUC", auc * 100, epoch)

          with open(metrics_csv_path, mode="a", newline="") as f:
              csv.writer(f).writerow([epoch, acc * 100, f1 * 100, recall * 100, precision * 100, auc * 100])
          
          if acc > best_acc:
            best_acc = acc
            torch.save(diffe.state_dict(), f'{root_dir}/DiffE/{config.NUM_CLASSES}class/{model_dir}/diffe_{best_acc*100:.2f}.pt')
          best_f1 = max(best_f1, f1)
          best_recall = max(best_recall, recall)
          best_precision = max(best_precision, precision)
          best_auc = max(best_auc, auc)

          description = f"Best accuracy: {best_acc*100:.2f}%"
          pbar.set_description(
            f"Method ALL - Processing subject {subjects} - {description}"
          )
          print()
          print(f"Accuracy: {acc:.4f}, F1 Score: {f1:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, AUC: {auc:.4f}")
      
      pbar.update(1)
      

if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Specify the device here
  subjects = config.SUB
  print(f'Subjects : {subjects}')
  print(f'Session : {config.SESSION}')
  print(f'Model_type : {config.MODEL_TYPE}')
  print(f'Epochs : {config.EPOCHS}')
  
  if config.LOSO:
    for loso in range(len(subjects)):
      args = argparse.Namespace(device=device, subjects=subjects, test_sub_idx=loso)
      train(args)
  else:
    for sub in subjects:
      sub = [sub]
      args = argparse.Namespace(device=device, subjects=sub)
      train(args)
      print(f"\n\nSubject {sub} is done.")