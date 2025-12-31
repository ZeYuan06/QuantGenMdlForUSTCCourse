#!/usr/bin/env python

import argparse
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MNIST01CNN(nn.Module):
    def __init__(self, feat_dim: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14->7
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 7->3
        )
        self.fc1 = nn.Linear(32 * 3 * 3, feat_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(feat_dim, 2)

    def forward(self, x):
        h = self.conv(x)
        h = h.reshape(h.shape[0], -1)
        feat = self.relu(self.fc1(h))
        logits = self.fc2(feat)
        return logits

    @torch.no_grad()
    def features(self, x):
        h = self.conv(x)
        h = h.reshape(h.shape[0], -1)
        feat = self.relu(self.fc1(h))
        return feat


def accuracy_from_logits(logits, y):
    pred = torch.argmax(logits, dim=1)
    return (pred == y).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/mnist01")
    ap.add_argument("--out", default="data/mnist01/classifier")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    x_tr = np.load(os.path.join(args.data, "x_train_14x14.npy")).astype(np.float32)
    y_tr = np.load(os.path.join(args.data, "y_train.npy")).astype(np.int64)
    x_te = np.load(os.path.join(args.data, "x_test_14x14.npy")).astype(np.float32)
    y_te = np.load(os.path.join(args.data, "y_test.npy")).astype(np.int64)

    # (N,1,14,14)
    x_tr_t = torch.from_numpy(x_tr)[:, None, :, :]
    y_tr_t = torch.from_numpy(y_tr)
    x_te_t = torch.from_numpy(x_te)[:, None, :, :]
    y_te_t = torch.from_numpy(y_te)

    train_loader = DataLoader(TensorDataset(x_tr_t, y_tr_t), batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_te_t, y_te_t), batch_size=args.batch, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNIST01CNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = -1.0
    best_path = os.path.join(args.out, "mnist01_cnn.pt")

    for ep in range(1, args.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

        model.eval()
        accs = []
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            accs.append(accuracy_from_logits(logits, yb))
        acc = float(np.mean(accs))

        if acc > best_acc:
            best_acc = acc
            torch.save({"state_dict": model.state_dict()}, best_path)

        print(f"epoch {ep:02d}  test_acc={acc:.4f}  best={best_acc:.4f}")

    print("saved:", best_path)


if __name__ == "__main__":
    main()
