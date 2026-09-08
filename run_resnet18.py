"""
混凝土裂缝检测 - ResNet18 迁移学习
一键运行：加载数据 → 训练 → 生成论文用图表
输出：resnet18_results.png（训练曲线 + 混淆矩阵）
"""
import os, numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
import matplotlib.pyplot as plt

# ========================
# 1. 加载数据
# ========================
DATA = os.path.expanduser('~/sdnet2018')
SAMPLE, SIZE = 8000, 160

files_crack, files_nocrack = [], []
for s in ['D', 'P', 'W']:
    for sub, lst in [('C', files_crack), ('U', files_nocrack)]:
        folder = os.path.join(DATA, s, sub + s)
        if os.path.exists(folder):
            for f in os.listdir(folder):
                if f.endswith(('.jpg', '.png', '.jpeg')):
                    lst.append(os.path.join(folder, f))

np.random.seed(42)
files_crack = list(np.random.choice(files_crack, SAMPLE // 2, replace=False))
files_nocrack = list(np.random.choice(files_nocrack, SAMPLE // 2, replace=False))

print(f"Crack images: {len(files_crack)}")
print(f"NoCrack images: {len(files_nocrack)}")
print("Loading images...")

X_list, y_list = [], []
for i, fp in enumerate(files_crack + files_nocrack):
    try:
        img = Image.open(fp).convert('RGB').resize((SIZE, SIZE))
        X_list.append(np.array(img).transpose(2, 0, 1) / 255.0)
        y_list.append(1 if os.path.basename(os.path.dirname(fp)).startswith('C') else 0)
    except:
        pass
    if (i + 1) % 2000 == 0:
        print(f"  Loaded {i + 1}/{len(files_crack) + len(files_nocrack)}")

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.int64)
print(f"Total loaded: {len(X)} images, shape={X.shape}")

# ========================
# 2. 预处理
# ========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
X_train_n = (X_train - mean) / std
X_test_n = (X_test - mean) / std

# ========================
# 3. DataLoader
# ========================
class CrackDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return torch.tensor(self.X[i]).float(), torch.tensor(self.y[i])

train_dl = DataLoader(CrackDataset(X_train_n, y_train), batch_size=16, shuffle=True)
test_dl = DataLoader(CrackDataset(X_test_n, y_test), batch_size=32)

# ========================
# 4. ResNet18
# ========================
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"\nDevice: {device}")

model = models.resnet18(weights='IMAGENET1K_V1')
model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, 2))
model = model.to(device)

opt = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

# ========================
# 5. 训练
# ========================
print("\nTraining...")
history = {'train_acc': [], 'val_acc': []}
best_acc = 0

for epoch in range(12):
    # Train
    model.train()
    tc, tt = 0, 0
    for bx, by in train_dl:
        bx, by = bx.to(device), by.to(device)
        opt.zero_grad()
        loss = criterion(model(bx), by)
        loss.backward()
        opt.step()
        tc += (model(bx).argmax(1) == by).sum().item()
        tt += len(by)

    # Validate
    model.eval()
    vc, vt = 0, 0
    with torch.no_grad():
        for bx, by in test_dl:
            vc += (model(bx.to(device)).argmax(1).cpu() == by).sum().item()
            vt += len(by)

    train_acc = tc / tt
    val_acc = vc / vt
    if val_acc > best_acc:
        best_acc = val_acc

    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    print(f"Epoch {epoch + 1:2d} | Train: {train_acc:.4f} | Val: {val_acc:.4f} | Best: {best_acc:.4f}")

# ========================
# 6. 评估 + 出图
# ========================
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for bx, by in test_dl:
        all_preds.extend(model(bx.to(device)).argmax(1).cpu().tolist())
        all_labels.extend(by.tolist())

acc = accuracy_score(all_labels, all_preds)
print(f"\n{'='*50}")
print(f"Final Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
print(f"{'='*50}")
print(classification_report(all_labels, all_preds, target_names=['NoCrack', 'Crack']))

# 画图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：训练曲线
axes[0].plot(history['train_acc'], 'b-o', label='Train', markersize=5, linewidth=1.5)
axes[0].plot(history['val_acc'], 'r-o', label='Validation', markersize=5, linewidth=1.5)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title('ResNet18 Training Curve (SDNET2018)', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim([0.5, 1.0])

# 右图：混淆矩阵
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['NoCrack', 'Crack'])
disp.plot(ax=axes[1], cmap='Blues', colorbar=False, text_kw={'fontsize': 14})
axes[1].set_title(f'Confusion Matrix (Acc = {acc:.4f})', fontsize=13, fontweight='bold')

plt.tight_layout()

# 保存到项目文件夹
output_path = os.path.join(os.path.dirname(__file__), 'resnet18_results.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nFigure saved: {output_path}")
plt.show()
