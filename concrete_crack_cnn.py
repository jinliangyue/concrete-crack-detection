"""
混凝土裂缝检测 - 优化版 CNN
全量数据 + 128x128 + 数据增强
"""
import os, numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR

DATA_DIR = '/Users/xiayuhao/sdnet2018'
SAMPLE = 10000
IMG_SIZE = 96

# Collect files
files_crack, files_nocrack = [], []
for surface in os.listdir(DATA_DIR):
    spath = os.path.join(DATA_DIR, surface)
    if not os.path.isdir(spath): continue
    for sub in os.listdir(spath):
        folder = os.path.join(spath, sub)
        if not os.path.isdir(folder): continue
        is_crack = sub.startswith('C')
        for f in os.listdir(folder):
            if f.endswith(('.jpg','.png','.jpeg')):
                (files_crack if is_crack else files_nocrack).append(os.path.join(folder, f))

print(f"Total: crack={len(files_crack)} nocrack={len(files_nocrack)}")

np.random.seed(42)
if len(files_crack) > SAMPLE//2: files_crack = list(np.random.choice(files_crack, SAMPLE//2, replace=False))
if len(files_nocrack) > SAMPLE//2: files_nocrack = list(np.random.choice(files_nocrack, SAMPLE//2, replace=False))
print(f"Sampled: crack={len(files_crack)} nocrack={len(files_nocrack)}")

# Load
X_list, y_list = [], []
for i, fpath in enumerate(files_crack + files_nocrack):
    try:
        img = Image.open(fpath).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
        X_list.append(np.array(img).transpose(2,0,1) / 255.0)
        y_list.append(1 if os.path.basename(os.path.dirname(fpath))[0] == 'C' else 0)
    except: pass
    if (i+1) % 3000 == 0: print(f"  Loaded {i+1}/{len(files_crack)+len(files_nocrack)}")

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.int64)
print(f"Total loaded: {len(X)} images, shape={X.shape}")

# Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# Data augmentation inline
def augment(x):
    if np.random.random() > 0.5: x = np.flip(x, axis=2).copy()  # horizontal flip
    if np.random.random() > 0.5: x = np.flip(x, axis=1).copy()  # vertical flip
    return x

class AugDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, train=True):
        self.X, self.y, self.train = X, y, train
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        x = self.X[i].copy()
        if self.train: x = augment(x)
        return torch.tensor(x), torch.tensor(self.y[i])

train_ds = AugDataset(X_train, y_train, train=True)
test_ds  = AugDataset(X_test, y_test, train=False)
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
test_dl  = DataLoader(test_ds, batch_size=128)

# Model (deeper)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Device: {device}")

model = nn.Sequential(
    nn.Conv2d(3,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(),
    nn.Conv2d(32,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.1),

    nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
    nn.Conv2d(64,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.15),

    nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(),
    nn.Conv2d(128,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.2),

    nn.Conv2d(128,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.25),

    nn.Flatten(),
    nn.Linear(256*(IMG_SIZE//16)*(IMG_SIZE//16), 512), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(512, 2)
).to(device)

print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=15)

# Train
best_acc = 0
for epoch in range(15):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for bx, by in train_dl:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        loss = criterion(model(bx), by)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (model(bx).argmax(1) == by).sum().item()
        total += len(by)

    model.eval()
    v_correct, v_total = 0, 0
    with torch.no_grad():
        for bx, by in test_dl:
            bx, by = bx.to(device), by.to(device)
            v_correct += (model(bx).argmax(1) == by).sum().item()
            v_total += len(by)
    v_acc = v_correct / v_total
    if v_acc > best_acc: best_acc = v_acc
    scheduler.step()

    print(f"E{epoch+1:2d} | Loss {total_loss/len(train_dl):.4f} | "
          f"Train {correct/total:.4f} | Val {v_acc:.4f} | Best {best_acc:.4f}")

# Final Eval
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for bx, by in test_dl:
        all_preds.extend(model(bx.to(device)).argmax(1).cpu().tolist())
        all_labels.extend(by.tolist())

from sklearn.metrics import accuracy_score, classification_report
acc = accuracy_score(all_labels, all_preds)
print(f"\nFinal Accuracy: {acc:.4f} ({acc*100:.2f}%)")
print(classification_report(all_labels, all_preds, target_names=['NoCrack','Crack']))

# Save checkpoint (project-relative path; legacy script from pre-refactor era)
from pathlib import Path as _P
_LEGACY_CKPT_DIR = _P(__file__).resolve().parent / "models"
_LEGACY_CKPT_DIR.mkdir(exist_ok=True)
torch.save(model.state_dict(), str(_LEGACY_CKPT_DIR / "crack_cnn.pt"))
print(f"Saved: {_LEGACY_CKPT_DIR / 'crack_cnn.pt'}")
