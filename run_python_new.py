# FILTER 3: LOC RA GIAI DOAN TEST 2018 - 2020
import numpy as np
import pandas as pd

df = pd.read_csv('/content/drive/My Drive/Colab Notebooks/DIEMLVCHUANHOA1.csv')

df = df.loc[(df['F_NHCANDUOI'] >= 2018) & (df['F_NHCANDUOI'] <= 2020)]
print(df.shape)

# Lọc ra các cột cần thiết để lưu file mới
df = df[['F_MASV', 'F_MAMH', 'NHHK', 'F_DIEM4']]

# Convert F_NHHK về string:
df['NHHK'] = df['NHHK'].astype(str)

# Hàm chuyển đổi F_NHHK về kí số:


def yearConvert(yNeed):
    yVar = int(yNeed[0:4])
    yTemp = int(yNeed[4])
    rs = (yVar - 2007 - 1) * 3 + yTemp
    return rs


df['NHHK'] = df['NHHK'].apply(yearConvert)

#df[['F_MASV', 'F_MAMH', 'NHHK', 'IDKH', 'F_NHHK']]

# WRITE FILE TO CSV PANDAS
df.to_csv('/content/drive/My Drive/Colab Notebooks/TEST20182020.csv',
          sep=',', index=False)
