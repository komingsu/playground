# Baseline: DiffE (DDPM + Encoder/Decoder + FC)
# Supports: word-level (20-class) or condition-level (4-class) classification
# Training: subject-wise and session-wise (day1/day2/day3)
# Handles missing files, logs to TensorBoard and CSV

from models.diffe import *
from utils import *

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


def evaluate(encoder, fc, generator, device):
    labels = np.arange(0, 20)
    Y = []
    Y_hat = []
    for x, y in generator:
        x, y = x.to(device), y.type(torch.LongTensor).to(device)
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
    subject = args.subject
    session = args.session
    device = torch.device(args.device)
    batch_size = 32
    batch_size2 = 260
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    print("Random Seed: ", seed)

    root_dir = "/exHDD/sunny/newGCN/6r__epo"
    try:
        X, Y = load_data(root_dir=root_dir, subject=subject, session=session)
        # X, Y = load_data(root_dir=root_dir, subject=subject, session=session, task=task)
    except FileNotFoundError:
        print(f"sub{subject} day{session} not found. Skipping.")
        return

    train_loader, test_loader = get_dataloader(X, Y, batch_size, batch_size2, seed, shuffle=True)

    num_classes = 20
    channels = X.shape[1]

    n_T = 1000
    ddpm_dim = 128
    encoder_dim = 256
    fc_dim = 512

    ddpm_model = ConditionalUNet(in_channels=channels, n_feat=ddpm_dim).to(device)
    ddpm = DDPM(nn_model=ddpm_model, betas=(1e-6, 1e-2), n_T=n_T, device=device).to(device)
    encoder = Encoder(in_channels=channels, dim=encoder_dim).to(device)
    decoder = Decoder(in_channels=channels, n_feat=ddpm_dim, encoder_dim=encoder_dim).to(device)
    fc = LinearClassifier(encoder_dim, fc_dim, emb_dim=num_classes).to(device)
    diffe = DiffE(encoder, decoder, fc).to(device)

    print("ddpm size: ", sum(p.numel() for p in ddpm.parameters()))
    print("encoder size: ", sum(p.numel() for p in encoder.parameters()))
    print("decoder size: ", sum(p.numel() for p in decoder.parameters()))
    print("fc size: ", sum(p.numel() for p in fc.parameters()))

    criterion = nn.L1Loss()
    criterion_class = nn.MSELoss()

    base_lr, lr = 9e-5, 1.5e-3
    optim1 = optim.RMSprop(ddpm.parameters(), lr=base_lr)
    optim2 = optim.RMSprop(diffe.parameters(), lr=base_lr)

    fc_ema = EMA(diffe.fc, beta=0.95, update_after_step=100, update_every=10)

    step_size = 150
    scheduler1 = optim.lr_scheduler.CyclicLR(optim1, base_lr, lr, step_size_up=step_size, mode="exp_range", cycle_momentum=False, gamma=0.9998)
    scheduler2 = optim.lr_scheduler.CyclicLR(optim2, base_lr, lr, step_size_up=step_size, mode="exp_range", cycle_momentum=False, gamma=0.9998)

    result_dir = f"./results/sub{subject}/day{session}"
    # result_dir = f"./results/sub{subject}/day{session}_{task}"
    os.makedirs(result_dir, exist_ok=True)
    metrics_csv_path = os.path.join(result_dir, "metrics.csv")
    writer = SummaryWriter(log_dir=f"./runs/sub{subject}/day{session}")
    if not os.path.exists(metrics_csv_path):
        with open(metrics_csv_path, mode="w", newline="") as f:
            csv.writer(f).writerow(["Epoch", "Accuracy", "F1", "Recall", "Precision", "AUC"])

    num_epochs = 500
    test_period = 1
    start_test = test_period
    alpha = 0.1

    best_acc = 0
    best_f1 = 0
    best_recall = 0
    best_precision = 0
    best_auc = 0

    with tqdm(total=num_epochs, desc=f"Method ALL - sub{subject} day{session}") as pbar:
        for epoch in range(num_epochs):
            ddpm.train()
            diffe.train()

            for x, y in train_loader:
                x, y = x.to(device), y.type(torch.LongTensor).to(device)
                y_cat = F.one_hot(y, num_classes=20).float().to(device)

                optim1.zero_grad()
                x_hat, down, up, noise, t = ddpm(x)
                loss_ddpm = F.l1_loss(x_hat, x, reduction="none")
                loss_ddpm.mean().backward()
                optim1.step()
                ddpm_out = x_hat, down, up, t

                optim2.zero_grad()
                decoder_out, fc_out = diffe(x, ddpm_out)
                loss_gap = criterion(decoder_out, loss_ddpm.detach())
                loss_c = criterion_class(fc_out, y_cat)
                loss = loss_gap + alpha * loss_c
                loss.backward()
                optim2.step()

                scheduler1.step()
                scheduler2.step()
                fc_ema.update()

            if epoch > start_test and epoch % test_period == 0:
                ddpm.eval()
                diffe.eval()

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
                    torch.save(diffe.state_dict(), os.path.join(result_dir, f"diffe_sub{subject}_day{session}.pt"))

                best_f1 = max(best_f1, f1)
                best_recall = max(best_recall, recall)
                best_precision = max(best_precision, precision)
                best_auc = max(best_auc, auc)

                pbar.set_description(f"sub{subject} day{session} - Best acc: {best_acc*100:.2f}%")
            pbar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_subjects", type=int, default=22)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    for session in [3]:  #############################################################
        for i in range(7, args.num_subjects + 1):
            args.subject = i
            args.session = session
            train(args)
