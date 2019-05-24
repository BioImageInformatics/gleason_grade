from __future__ import print_function
import os
import re
import cv2
import sys
import glob
import argparse
import numpy as np
import pandas as pd

pd.set_option('expand_frame_repr', False)

np_patt = 'unet/10x/inference/s*.npy'
np_list = sorted(glob.glob(np_patt))
opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
print(len(np_list))

class_dict = {0: 'G3', 1: 'G4', 2: 'G5', 3: 'BN', 4: 'ST'}
case_patt = r'(?P<cc>s\d{1,2}[-_]\d+)-(?P<ss>\d+)'
df = pd.DataFrame()
for np_path in np_list:
    np_base = os.path.basename(np_path).replace('_prob.npy', '')
    case_name, slide_num = re.findall(string=np_base, pattern=case_patt)[0]

    prob = np.load(np_path)
    pred = np.argmax(prob, axis=-1)
    pred[prob.sum(axis=-1) < 1. - 1e-3] = 5
    total_area = np.prod(pred.shape)
    case_comp = {}
    for key, val in class_dict.items():
        n_key = (pred == key).sum()
        case_comp[val] = n_key


    hgpc = ((pred == 1) + (pred == 2)).astype(np.uint8)
    hgpc = cv2.morphologyEx(hgpc, cv2.MORPH_OPEN, opening_kernel)
    hgpc = cv2.dilate(hgpc, dilate_kernel, iterations=2)
    hgpc_name = np_path.replace('_prob.npy', '_hgpc.jpg')
    cv2.imwrite(hgpc_name, hgpc * 255)

    data = pd.DataFrame({
        'File': np_path,
        'CaseID': case_name,
        'ScannedSlideNum': int(slide_num),
        'G3': case_comp['G3'],
        'G4': case_comp['G4'],
        'G5': case_comp['G5'],
        'RawHighGrade': case_comp['G4']+case_comp['G5'],
        'ProcessedHighGrade': hgpc.sum(),
        'IsMax': 0,
        'Rank': 0}, index=[np_base])
    df = df.append(data)
    print(np_base, case_name, prob.shape)


case_ids = np.unique(df.CaseID.values)
print(case_ids)

for case_id in case_ids:
    df_index = df.CaseID == case_id
    case_df = df.iloc[[x == case_id for x in df.CaseID], :]
    case_df = case_df.ProcessedHighGrade
    case_max = case_df.idxmax(axis=0)
    print(case_max)

    case_sorted = pd.DataFrame(case_df).sort_values('ProcessedHighGrade', ascending=False)
    df.loc[case_max, 'IsMax'] = 1

    for position, idx in enumerate(case_sorted.index):
        df.loc[idx, 'Rank'] = position+1

print(df)
df.to_csv('Durham_HighGrade_Content_CC.csv')
