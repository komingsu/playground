import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import config
import glob


# Define a function to perform z-score normalization on the data
def zscore_norm(data):
  # Calculate the mean and standard deviation for each channel in each batch
  mean = torch.mean(data, dim=(1, 2))
  std = torch.std(data, dim=(1, 2))

  # Subtract the mean from each channel in each batch and divide by the standard deviation
  norm_data = (data - mean[:, None, None]) / std[:, None, None]

  return norm_data

# Define a function to perform min-max normalization on the data
def minmax_norm(data):
  # Calculate the minimum and maximum values for each channel and sequence in the batch
  min_vals = torch.min(data, dim=-1)[0]
  max_vals = torch.max(data, dim=-1)[0]

  # Scale the data to the range [0, 1]
  norm_data = (data - min_vals.unsqueeze(-1)) / (
      max_vals.unsqueeze(-1) - min_vals.unsqueeze(-1)
  )

  return norm_data

class EEGDataset(Dataset):
  "Characterizes a dataset for PyTorch"

  def __init__(self, X, Y, M, transform=None):
    "Initialization"
    self.X = X
    self.Y = Y
    self.M = M
    self.transform = transform

  def __len__(self):
    "Denotes the total number of samples"
    return len(self.X)

  def __getitem__(self, index):
    "Generates one sample of data"
    # Load data and get label
    x = self.X[index]
    y = self.Y[index]
    m = self.M[index]
    if self.transform:
      x = self.transform(x)
    return x.float(), y, m

def update_labels_for_session(y, ranges):
  np_labels = np.full_like(y, fill_value=3)

  for range_idx, (start, end) in enumerate(ranges):
      mask = (y >= start) & (y <= end)
      relative_pos = y[mask] - start + 1

      if config.SESSION == 'all':
        nested_label = np.where(
            relative_pos == 1, 0,         # 첫 번째만 word
            np.where(relative_pos <= 3, 1, 2)  # 2,3 → is / 4,5 → vi
        )
      elif config.SESSION == 'word':
          nested_label = np.full_like(relative_pos, 0)
      elif config.SESSION == 'is':
          nested_label = np.full_like(relative_pos, 1)
      elif config.SESSION == 'vi':
          nested_label = np.full_like(relative_pos, 2)
      else:
          raise ValueError(f"Unknown SESSION: {config.SESSION}")
      
      np_labels[mask] = nested_label
  return np_labels

def update_labels(x, y):
    # 데이터 셔플
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    x, y = x[indices], y[indices]
    
    # y의 최대값 찾기
    max_value = np.max(y)
    
    # y == max_value인 데이터의 개수를 1/3로 줄이기
    if config.SESSION != 'all':
      max_indices = np.where(y == max_value)[0]
      num_keep = len(max_indices) // 3
      selected_max_indices = np.random.choice(max_indices, num_keep, replace=False)
      
      # 나머지 데이터 선택
      remaining_indices = np.setdiff1d(np.arange(len(y)), max_indices)
      selected_indices = np.concatenate([selected_max_indices, remaining_indices])
      
    
    # 나머지 데이터 처리 (기존 로직 유지)
    step = max_value // 3
    ranges = [(i, min(i + step - 1, max_value)) for i in range(1, max_value, step)]
    print("Ranges: ", ranges)
    mapped_labels = np.where(
        y == max_value, 3,
        np.where(
            (ranges[0][0] <= y) & (y <= ranges[0][1]), 0,
            np.where(
                (ranges[1][0] <= y) & (y <= ranges[1][1]), 1,
                np.where(
                    (ranges[2][0] <= y) & (y <= ranges[2][1]), 2, y
                )
            )
        )
    )
    if config.SESSION != 'all':  
      # 선택된 인덱스를 기반으로 x, y 업데이트
      x, mapped_labels = x[selected_indices], mapped_labels[selected_indices]
      modality_labels = update_labels_for_session(mapped_labels, ranges)
    else:
      modality_labels = update_labels_for_session(y, ranges)
    print("Mapped labels: ", np.unique(mapped_labels))
    print("Modality labels: ", np.unique(modality_labels))
    print("Mapped labels shape: ", mapped_labels.shape)
    print("Modality labels shape: ", modality_labels.shape)
    return x, mapped_labels, modality_labels

def load_data(root_dir, data_dir, data_folder, subjects, session, is_datacollection):
    all_X = None
    all_Y = None
    all_M = None

    print("<< Load DATA >>")
    print("session: ", session)
    for subject in subjects:
        subject = 'sub' + str(subject)
        print(f'{subject} >> \t', end='\n')
        if is_datacollection:
          files = glob.glob(os.path.join(root_dir, data_dir, f'{subject}', data_folder, f"*{subject}*datacollection*{session}*.npz"))
        else:
          files = glob.glob(os.path.join(root_dir, data_dir, f'{subject}', data_folder, f"*{subject}*online*{session}*.npz"))
        print(f'files: {files}')
        for file_path in files:
          print(f'file path: {file_path.split("/")[-1]}')
          file = np.load(file_path)
          X = np.float32(file['X'])
          X = np.transpose(X, (2, 1, 0))
          Y = np.int_(file["y"])
          print(f'Original >> X shape: {X.shape} , Y shape: {Y.shape}')
          print("Unique labels: ", np.unique(Y))
          X, Y, M = update_labels(X, Y)

          print(f'Mapped >> X shape: {X.shape} , Y shape: {Y.shape}')
          # print(Y)
          print("Unique labels: ", np.unique(Y))
          if all_X is None:
              all_X = X
          else:
              all_X = np.concatenate((all_X, X), axis=0)

          if all_Y is None:
              all_Y = Y
          else:
              all_Y = np.concatenate((all_Y, Y), axis=0)
          
          if all_M is None:
              all_M = M
          else:
              all_M = np.concatenate((all_M, M), axis=0)

          # Print label distribution for the current subject
          unique, counts = np.unique(Y, return_counts=True)
          print("Label counts: ", dict(zip(unique, counts)))
          print("=====================================")
          print()

    X, Y, M = all_X, all_Y, all_M
    print(f'Final >> X shape: {X.shape} , Y shape: {Y.shape}, M shape: {M.shape}')
    X, Y, M = zscore_norm(torch.from_numpy(X)), torch.from_numpy(Y), torch.from_numpy(M)
    # Print final label distribution
    unique, counts = np.unique(Y.numpy(), return_counts=True)
    print("\nFinal Label counts: ", dict(zip(unique, counts)))
    unique2, counts2 = np.unique(M.numpy(), return_counts=True)
    print("Final Modality counts: ", dict(zip(unique2, counts2)))
    print("\n=====\n")
    
    print(f'X shape: {X.shape}')
    print(f'Y shape: {Y.shape}')
    print(f'M shape: {M.shape}')
    print("\n=====\n")
    return X, Y, M

def split_subjects_data(all_X, all_Y, test_subject_index):
  train_X, train_Y = [], []
  test_X, test_Y = [], []

  test_subject_start = config.SPLIT_RANGE * test_subject_index
  test_subject_end = test_subject_start + config.SPLIT_RANGE
  print('start idx: ', test_subject_start, ', end idx: ', test_subject_end)
  for i in range(len(all_X)):
    if i >= test_subject_start and i < test_subject_end:
      # print("i >> ", i+1)
      test_X.append(all_X[i])
      test_Y.append(all_Y[i])
    else:
      train_X.append(all_X[i])
      train_Y.append(all_Y[i])
  print("train len: ", len(train_X), ", test len: ", len(test_X))
  
  # 리스트를 텐서로 변환
  train_X = torch.stack(train_X) if train_X else torch.tensor([])
  train_Y = torch.stack(train_Y) if train_Y else torch.tensor([])
  test_X = torch.stack(test_X) if test_X else torch.tensor([])
  test_Y = torch.stack(test_Y) if test_Y else torch.tensor([])
  
  return train_X, test_X, train_Y, test_Y

def get_dataloader(X, Y, M, batch_size, batch_size2, seed, shuffle=True, test_sub_idx=None):
    print("<< Get Data Loader >>")
    print('batch_size: ', batch_size)
    print('batch_size2: ', batch_size2)
    print('seed: ', seed)

    if config.LOSO:
        print("\n=====\n")
        print("<< Subject Split >>")
        sub_len = X.shape[0] // config.SPLIT_RANGE
        print("subject length: ", sub_len)
        print('test sub index: ', test_sub_idx)

        test_subject_start = config.SPLIT_RANGE * test_sub_idx
        test_subject_end = test_subject_start + config.SPLIT_RANGE

        X_train = torch.cat([X[:test_subject_start], X[test_subject_end:]], dim=0)
        Y_train = torch.cat([Y[:test_subject_start], Y[test_subject_end:]], dim=0)
        M_train = torch.cat([M[:test_subject_start], M[test_subject_end:]], dim=0)

        X_test = X[test_subject_start:test_subject_end]
        Y_test = Y[test_subject_start:test_subject_end]
        M_test = M[test_subject_start:test_subject_end]

        training_set = EEGDataset(X_train, Y_train, M_train)
        training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=shuffle)

        test_set = EEGDataset(X_test, Y_test, M_test)
        test_loader = DataLoader(test_set, batch_size=batch_size2, shuffle=False)

        return training_loader, test_loader

    else:
      X_train, X_test, Y_train, Y_test, M_train, M_test = train_test_split(
          X, Y, M, test_size=0.2, shuffle=shuffle, stratify=Y, random_state=seed
        )
    training_set = EEGDataset(X_train, Y_train, M_train)
    training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=shuffle)
    test_set = EEGDataset(X_test, Y_test, M_test)
    test_loader = DataLoader(test_set, batch_size=batch_size2, shuffle=False)

    return training_loader, test_loader

def match_channels(full_channels, target_channels):
    full_channels_upper = [ch.upper() for ch in full_channels]
    target_channels_upper = [ch.upper() for ch in target_channels]

    matched_indices = []
    not_found_channels = []

    for target in target_channels_upper:
        if target in full_channels_upper:
            idx = full_channels_upper.index(target)
            matched_indices.append(idx)
        else:
            not_found_channels.append(target)

    if not_found_channels:
        print(f"Warning: {len(not_found_channels)} channels not found in your data: {not_found_channels}")

    return matched_indices