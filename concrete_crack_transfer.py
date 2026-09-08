"""
混凝土裂缝检测 - ResNet18 迁移学习
"""
import os, numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms, models
from torch.optim.lr_scheduler import CosineAnnealingLR

DATA_DIR = '/Users/xiayuhao/sdnet2018'
SAMPLE = 15000
IMG_SIZE = 224  # ResNet needs 224

# Collect
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
print("Loading images...")
X_list, y_list = [], []
for i, fpath in enumerate(files_crack + files_nocrack):
    try:
        img = Image.open(fpath).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
        X_list.append(np.array(img).transpose(2,0,1) / 255.0)
        y_list.append(1 if os.path.basename(os.path.dirname(fpath))[0] == 'C' else 0)
    except: pass
    if (i+1) % 5000 == 0: print(f"  {i+1}/{len(files_crack)+len(files_nocrack)}")

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.int64)
print(f"Loaded: {len(X)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

# Normalize (ImageNet stats)
mean = np.array([0.485, 0.456, 0.406]).reshape(3,1,1)
std  = np.array([0.229, 0.224, 0.225]).reshape(3,1,1)
X_train_norm = (X_train - mean) / std
X_test_norm  = (X_test - mean) / std

# Augmentation
class AugDS(torch.utils.data.Dataset):
    def __init__(self, X, y, train=True):
        self.X, self.y, self.train = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        x = self.X[i].copy()
        if self.train:
            if np.random.random() > 0.5: x = np.flip(x, 2).copy()
            if np.random.random() > 0.5: x = np.flip(x, 1).copy()
        return torch.tensor(x).float(), torch.tensor(self.y[i])

train_dl = DataLoader(AugDS(X_train_norm, y_train, True), batch_size=32, shuffle=True)
test_dl  = DataLoader(AugDS(X_test_norm, y_test, False), batch_size=64)

# ResNet18
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Device: {device}")

model = models.resnet18(weights='IMAGENET1K_V1')
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(512, 2)
)
model = model.to(device)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=12)

# Train
print("\nTraining...")
best_acc = 0
for epoch in range(12):
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

# Final
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

torch.save(model.state_dict(), '/Users/xiayuhao/Desktop/Claude code/crack_resnet18.pt')
print("Saved: crack_resnet18.pt")
