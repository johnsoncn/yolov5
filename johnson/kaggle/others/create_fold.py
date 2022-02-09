# -*- coding: utf-8 -*-
"""
@Time    : 2022/1/6 17:51
@Author  : Johnson
@FileName: create_fold.py
"""

from sklearn.model_selection import GroupKFold

kf = GroupKFold(n_splits = 5)
df = df.reset_index(drop=True)
df['fold'] = -1
for fold, (train_idx, val_idx) in enumerate(kf.split(df, y = df.video_id.tolist(), groups=df.sequence)):
    df.loc[val_idx, 'fold'] = fold
print(df.fold.value_counts())