"""
混凝土裂缝检测 - 随机森林版（全量）
"""
import os, numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score

DATA_DIR = '/Users/xiayuhao/sdnet2018'
SAMPLE = 5000  # 取5000张

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

print(f"Total crack files: {len(files_crack)}, nocrack: {len(files_nocrack)}")

# Sample
np.random.seed(42)
if len(files_crack) > SAMPLE//2:
    files_crack = list(np.random.choice(files_crack, SAMPLE//2, replace=False))
if len(files_nocrack) > SAMPLE//2:
    files_nocrack = list(np.random.choice(files_nocrack, SAMPLE//2, replace=False))
print(f"Sampled: crack={len(files_crack)}, nocrack={len(files_nocrack)}")

# Load
X, y = [], []
for fpath in files_crack + files_nocrack:
    try:
        img = Image.open(fpath).convert('L').resize((64,64))  # grayscale 64x64 = 4096 features
        X.append(np.array(img).flatten() / 255.0)
        y.append(1 if 'C' in os.path.basename(os.path.dirname(fpath))[0] else 0)
    except: pass

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)
print(f"Loaded: {len(X)} images, {X.shape[1]} features")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
acc = rf.score(X_test, y_test)
print(f"\nAccuracy: {acc:.4f} ({acc*100:.2f}%)")

cv = cross_val_score(rf, X, y, cv=3, scoring='accuracy', n_jobs=-1)
print(f"3-Fold CV: {cv.mean():.4f} (+/- {cv.std():.4f})")
print("Done!")
