import pickle
import numpy as np
import cv2

with open('data/LSWMD.pkl', 'rb') as f:
    df = pickle.load(f)

# 웨이퍼 맵 픽셀값 분포 확인
total_0, total_1, total_2 = 0, 0, 0

for i, row in df.iloc[:1000].iterrows():
    wafer_map = np.array(row['waferMap'])
    total_0 += (wafer_map == 0).sum()
    total_1 += (wafer_map == 1).sum()
    total_2 += (wafer_map == 2).sum()

total = total_0 + total_1 + total_2
print(f"빈공간(0): {total_0} ({total_0/total*100:.1f}%)")
print(f"정상(1):   {total_1} ({total_1/total*100:.1f}%)")
print(f"불량(2):   {total_2} ({total_2/total*100:.1f}%)")
