import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import cv2

# ============================================================
# 설정
# ============================================================
IMG_SIZE = 64
BATCH_SIZE = 256
EPOCHS = 50
LR = 1e-3
THRESHOLD_PERCENTILE = 95
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Device: {DEVICE}")

# ============================================================
# 데이터 로드 및 전처리
# ============================================================
def clean_label(x):
    if isinstance(x, np.ndarray):
        return str(x.flatten()[0]).strip()
    while isinstance(x, list):
        x = x[0]
    return str(x).strip()

def load_and_preprocess(pkl_path):
    print("데이터 로드 중...")
    with open(pkl_path, 'rb') as f:
        df = pickle.load(f)
    df['label'] = df['failureType'].apply(clean_label)
    images = []
    labels = []
    for _, row in df.iterrows():
        wafer_map = np.array(row['waferMap'], dtype=np.float32)

        # 0,1 → 0 (정상/빈공간), 2 → 1 (불량) 로 이진화
        wafer_binary = (wafer_map == 2).astype(np.float32)

        resized = cv2.resize(wafer_binary, (IMG_SIZE, IMG_SIZE),
                           interpolation=cv2.INTER_NEAREST)
        images.append(resized)
        labels.append(row['label'])
    images = np.array(images)
    labels = np.array(labels)
    print(f"전체 데이터: {images.shape}")
    return images, labels

# ============================================================
# Dataset
# ============================================================
class WaferDataset(Dataset):
    def __init__(self, images):
        self.images = torch.FloatTensor(images).unsqueeze(1)
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        return self.images[idx]

# ============================================================
# AutoEncoder (BatchNorm + Dropout + BCE)
# ============================================================
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# ============================================================
# 학습
# ============================================================
def train(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for images in dataloader:
        images = images.to(DEVICE)
        output = model(images)
        loss = criterion(output, images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def get_reconstruction_errors(model, dataloader):
    model.eval()
    errors = []
    with torch.no_grad():
        for images in dataloader:
            images = images.to(DEVICE)
            output = model(images)
            mse = ((output - images) ** 2).mean(dim=[1, 2, 3])
            errors.extend(mse.cpu().numpy())
    return np.array(errors)

# ============================================================
# 메인
# ============================================================
if __name__ == '__main__':
    images, labels = load_and_preprocess('data/LSWMD.pkl')

    normal_idx = labels == 'none'
    normal_images = images[normal_idx]
    abnormal_images = images[~normal_idx]

    print(f"\n정상 데이터: {len(normal_images)}")
    print(f"비정상 데이터: {len(abnormal_images)}")

    X_train, X_temp = train_test_split(normal_images, test_size=0.4, random_state=42)
    X_val, X_test_normal = train_test_split(X_temp, test_size=0.5, random_state=42)

    print(f"\nnone Train: {X_train.shape}")
    print(f"none Val:   {X_val.shape}")
    print(f"none Test:  {X_test_normal.shape}")

    X_test = np.concatenate([X_test_normal, abnormal_images], axis=0)
    y_test = np.array([0] * len(X_test_normal) + [1] * len(abnormal_images))

    print(f"\nTest 데이터: {X_test.shape}")
    print(f"Test 정상: {len(X_test_normal)}, 비정상: {len(abnormal_images)}")

    train_loader = DataLoader(WaferDataset(X_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(WaferDataset(X_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(WaferDataset(X_test), batch_size=BATCH_SIZE, shuffle=False)

    model = AutoEncoder().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()  # BCE로 변경

    print(f"\n=== 학습 시작 (BCE Loss) ===")
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        train_loss = train(model, train_loader, optimizer, criterion)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images_batch in val_loader:
                images_batch = images_batch.to(DEVICE)
                output = model(images_batch)
                val_loss += criterion(output, images_batch).item()
        val_loss /= len(val_loader)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {train_loss:.6f} Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'checkpoints/autoencoder_bce_best.pth')

    print("\n=== 평가 ===")
    model.load_state_dict(torch.load('checkpoints/autoencoder_bce_best.pth'))

    val_errors = get_reconstruction_errors(model, val_loader)
    threshold = np.percentile(val_errors, THRESHOLD_PERCENTILE)
    print(f"Threshold: {threshold:.6f}")

    test_errors = get_reconstruction_errors(model, test_loader)
    y_pred = (test_errors > threshold).astype(int)

    print(f"\n=== 성능 결과 (BCE Loss) ===")
    print(classification_report(y_test, y_pred, target_names=['정상', '비정상']))
    print(f"AUC-ROC: {roc_auc_score(y_test, test_errors):.4f}")
