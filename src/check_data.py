import pickle
import pandas as pd
import numpy as np

with open('data/LSWMD.pkl', 'rb') as f:
    df = pickle.load(f)

print(f"pandas 버전: {pd.__version__}")
print(f"shape: {df.shape}")
print(f"컬럼: {df.columns.tolist()}")

print("\n=== 클래스 분포 ===")
print(df['failureType'].value_counts())

print("\n=== 웨이퍼 맵 크기 샘플 ===")
print(f"첫 번째 웨이퍼 맵 크기: {df['waferMap'].iloc[0].shape}")
