import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch
import cv2

CLASSES = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full', 'Random', 'Scratch', 'none']
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}

def load_data(pkl_path):
    print("데이터 로드 중...")
    with open(pkl_path, 'rb') as f:
        df = pickle.load(f)
    print(f"로드 완료: {df.shape}")
    return df

def clean_failure_type(x):
    """numpy.ndarray [['none']] → none"""
    if isinstance(x, np.ndarray):
        return str(x.flatten()[0]).strip()
    while isinstance(x, list):
        x = x[0]
    return str(x).strip()

def preprocess(df, img_size=64):
    print("전처리 중...")
    df['label'] = df['failureType'].apply(clean_failure_type)
    print("\n=== 정리된 라벨 종류 ===")
    print(df['label'].value_counts())
    unmapped = df[~df['label'].isin(CLASS_TO_IDX.keys())]['label'].unique()
    print(f"\n매핑 안 된 라벨: {unmapped}")
    df = df[df['label'].isin(CLASS_TO_IDX.keys())]
    print(f"매핑 후 데이터 수: {len(df)}")
    df['label_idx'] = df['label'].map(CLASS_TO_IDX)
    images = []
    labels = []
    for idx, row in df.iterrows():
        wafer_map = np.array(row['waferMap'], dtype=np.float32)
        resized = cv2.resize(wafer_map, (img_size, img_size),
                           interpolation=cv2.INTER_NEAREST)
        resized = resized / 2.0
        images.append(resized)
        labels.append(row['label_idx'])
    images = np.array(images)
    labels = np.array(labels)
    print(f"전처리 완료: {images.shape}")
    return images, labels

def split_data(images, labels):
    X_train, X_temp, y_train, y_temp = train_test_split(
        images, labels, test_size=0.4, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test

class WaferDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = torch.FloatTensor(images).unsqueeze(1)
        self.labels = torch.LongTensor(labels)
        self.transform = transform
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

if __name__ == '__main__':
    df = load_data('data/LSWMD.pkl')
    images, labels = preprocess(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(images, labels)
    print("\n=== 데이터셋 확인 ===")
    train_dataset = WaferDataset(X_train, y_train)
    print(f"Train 샘플 shape: {train_dataset[0][0].shape}")
    print(f"Train 라벨 예시: {train_dataset[0][1]}")
    print("\n전처리 완료!")
