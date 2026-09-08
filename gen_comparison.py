"""
论文用：裂缝检测效果对比图
从测试集抽4组（2张裂缝+2张无裂缝），原图 vs 模型预测
"""
import os, numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
import matplotlib.pyplot as plt

DATA = os.path.expanduser('~/sdnet2018')
SAMPLE, SIZE = 2000, 160

# 加载数据
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

X_list, y_list, paths = [], [], []
for fp in files_crack + files_nocrack:
    try:
        img = Image.open(fp).convert('RGB').resize((SIZE, SIZE))
        X_list.append(np.array(img).transpose(2, 0, 1) / 255.0)
        y_list.append(1 if os.path.basename(os.path.dirname(fp)).startswith('C') else 0)
        paths.append(fp)
    except:
        pass

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.int64)

_, X_test, _, y_test, _, paths_test = train_test_split(
    X, y, paths, test_size=0.15, random_state=42, stratify=y
)

mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
X_test_n = (X_test - mean) / std

# 训练一个小模型（不用加载）
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Device: {device}")

model = models.resnet18(weights='IMAGENET1K_V1')
model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, 2))
model = model.to(device)

# 划分训练集
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_train_n = (X_train - mean) / std

class CD(torch.utils.data.Dataset):
    def __init__(self, X, y): self.X, self.y = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return torch.tensor(self.X[i]).float(), torch.tensor(self.y[i])

train_dl = DataLoader(CD(X_train_n, y_train), batch_size=16, shuffle=True)
opt = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

print("Training (6 epochs)...")
for epoch in range(6):
    model.train()
    for bx, by in train_dl:
        opt.zero_grad()
        loss = criterion(model(bx.to(device)), by.to(device))
        loss.backward()
        opt.step()
    print(f"  Epoch {epoch+1}/6 done")

model.eval()

# 预测
preds = []
with torch.no_grad():
    for i in range(0, len(X_test_n), 32):
        batch = torch.tensor(X_test_n[i:i + 32]).float().to(device)
        preds.extend(model(batch).argmax(1).cpu().tolist())

# 挑4组：2张裂缝（label=1）+ 2张无裂缝（label=0）
crack_idx = [i for i in range(len(y_test)) if y_test[i] == 1]
nocrack_idx = [i for i in range(len(y_test)) if y_test[i] == 0]
selected = [crack_idx[0], crack_idx[1], nocrack_idx[0], nocrack_idx[1]]

# 出图
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
labels_map = {0: 'NoCrack', 1: 'Crack'}

for i, idx in enumerate(selected):
    ax = axes[i // 2][i % 2]
    true_label = y_test[idx]
    pred_label = preds[idx]
    status = 'Correct' if true_label == pred_label else 'Wrong'
    color = 'green' if status == 'Correct' else 'red'

    img_orig = Image.open(paths_test[idx]).convert('RGB')
    ax.imshow(img_orig)
    ax.set_title(
        f'Ground Truth: {labels_map[true_label]}  |  '
        f'Prediction: {labels_map[pred_label]}  [{status}]',
        fontsize=11, color=color, fontweight='bold'
    )
    ax.axis('off')

plt.tight_layout()
output = os.path.join(os.path.dirname(__file__), 'crack_comparison.png')
plt.savefig(output, dpi=200, bbox_inches='tight')
print(f"Saved: {output}")
plt.show()
