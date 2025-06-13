IS_BASELINE = False
USE_Modality = True
FINETUNE = False # eegpt fine-tuning
FINETUNE_PERCENT = 0.7

LOSO = False

ROOT_DIR = "/exHDD/eyko/concept"
DATA_DIR = 'data/processed_data'
DATA_FOLDER = 'raw'

SESSION = 'is' # 'all', 'vi', 'word', 'is', 'separate'z
SPLIT_RANGE = 600 if SESSION == 'all' else 200
NUM_CLASSES = 16 if SESSION == 'separate' else 4
# MODEL_TYPE = "1_only_datacollection_2days"
# MODEL_TYPE = "1_baseline" if IS_BASELINE else "2_EEGPT_feature" if not USE_Modality else "3_EEGPT_latent_add_every_epoch"
MODEL_TYPE = "5_parameter_tuning"

# 환자 번호
# SUB = [6, 12, 18, 21]

# 일반인 번호
SUB = [i for i in range(1, 30) if i not in {1, 2, 4, 6, 10, 11, 12, 18, 21, 25}]
# SUB = [5, 7, 28, 29]
# SUB = [5]

EPOCHS = 200

if FINETUNE:
  BATCH_SIZE = 16
else:
  BATCH_SIZE = 32
BATCH_SIZE2 = 260
SEED = 42

MEMO = f"""
      사용 데이터: datacollection만 사용
      epoch: {EPOCHS}

      latent space에 center loss + projection head 개선

      semantic reconstruction을 강화 (text/image feature 복원)

      modality-aware contrastive loss 추가

      Classifier에 ArcFace 적용

      (필요시) Encoder LoRA fine-tuning 추가
      
      """
      
      
MEMO2 = f"""
      cross entropy loss 사용
      커널사이즈 1로 수정

      스케쥴러: cyclic learning rate 사용 (기존 모델)
      
      # 스케쥴러를 CosineAnnealingWarmRestarts로 수정
      # >> scheduler2 = optim.lr_scheduler.CosineAnnealingWarmRestarts(
      #                   optimizer=optim2,
      #                   T_0=30,       # 첫 번째 주기(epoch)의 길이
      #                   T_mult=2,     # 일정한 주기로 반복함
      #                   eta_min=1e-6  # 학습률의 최소값
      #                 )
      
      옵티마이저 AdamW로 수정
      >> optim.AdamW(ddpm.parameters(), lr=base_lr, weight_decay=1e-1)
      
      # Residual Conv Block에 dropout 추가
      # ConditionalUNet의 forward에 dropout 추가
      # Decoder의 업샘플링에 Dropout 추가
      
      **아침에 와서 해야할 일: BayesConv1d로 수정해보고, group normalization 추가해보기
      """
      
LABEL = {
  0: 'clock',
  1: 'toilet',
  2: 'water',
  3: 'rest',
}

# 내 데이터
channels_names = [
                                "Fp1", "Fp2",
                      "AF7", "AF3", "AFz", "AF4", "AF8",
              "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "FT9", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8", "FT10",
              "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
    "TP9", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8", "TP10",
            "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
                      "PO7", "PO3", "POz", "PO4", "PO8",
                              "O1", "Oz", "O2",      
]

# EEGPT channels
use_channels_names = [      'FP1', 'FP2', 
                        "AF7", 'AF3', 'AF4', "AF8", 
            'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 
        'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 
            'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 
        'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
             'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 
                      'PO7', 'PO3', 'POZ', 'PO4', 'PO8', 
                               'O1', 'OZ', 'O2', ]

# original_use_channels_names = [      'FP1', 'FPZ', 'FP2', 
#                         "AF7", 'AF3', 'AF4', "AF8", 
#             'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 
#         'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 
#             'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 
#         'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
#              'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 
#                       'PO7', "PO5", 'PO3', 'POZ', 'PO4', "PO6", 'PO8', 
#                                'O1', 'OZ', 'O2', ]

# Loss weights
LOSS_WEIGHTS = {
    'alpha': 0.05,        # classification loss weight (기존 0.1에서 감소)
    'beta': 0.05,         # contrastive/feature matching loss weight (기존 0.1에서 감소)
    'beta_center': 0.001, # center loss weight (기존 0.01에서 감소)
    'gamma': 0.01,        # semantic reconstruction loss weight (기존 0.05에서 감소)
}

# ArcFace parameters
ARCFACE_PARAMS = {
    'margin': 0.2,        # angular margin (기존 0.5에서 감소)
    'scale': 15,          # feature scale (기존 30에서 감소)
}

# Loss ablation switches
USE_CENTER_LOSS = True
USE_CONTRASTIVE_LOSS = True
USE_SEMANTIC_RECONSTRUCTION = True
USE_ARCFACE_CLASSIFIER = True
USE_FEATURE_MATCHING = False