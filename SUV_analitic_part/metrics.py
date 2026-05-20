import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
import os

def get_calculated_data():
    script_dir = os.path.dirname(__file__)
    excel_path = os.path.abspath(os.path.join(script_dir, '..', 'Data_exam_st.xlsx'))

    df = pd.read_excel(excel_path, header=1)
    df.columns = df.columns.str.strip().str.replace(r'\s+', ' ', regex=True)
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]

    target_col = next((col for col in df.columns if 'Онк' in col and 'Неонк' in col), None)
    y = pd.to_numeric(df[target_col], errors='coerce')

    suv_18f = pd.to_numeric(df['SUVочаг_18F'], errors='coerce')
    suv_11c = pd.to_numeric(df['SUVочаг_11С'], errors='coerce')

    #18F
    mask_18f = suv_18f.notna() & y.notna()
    scores_18f = suv_18f.loc[mask_18f].values
    labels_18f = y.loc[mask_18f].values
    fpr_18f, tpr_18f, thresholds_18f = roc_curve(labels_18f, scores_18f, pos_label=1)
    thresh_18f = thresholds_18f[np.argmax(tpr_18f - fpr_18f)]
    onco_18f = scores_18f[labels_18f == 1]
    nononco_18f = scores_18f[labels_18f == 0]

    #11C
    mask_11c = suv_11c.notna() & y.notna()
    scores_11c = suv_11c.loc[mask_11c].values
    labels_11c = y.loc[mask_11c].values
    fpr_11c, tpr_11c, thresholds_11c = roc_curve(labels_11c, scores_11c, pos_label=1)
    thresh_11c = thresholds_11c[np.argmax(tpr_11c - fpr_11c)]
    onco_11c = scores_11c[labels_11c == 1]
    nononco_11c = scores_11c[labels_11c == 0]

    return {
        'auc_18f': auc(fpr_18f, tpr_18f), 'thresh_18f': thresh_18f, 
        'auc_11c': auc(fpr_11c, tpr_11c), 'thresh_11c': thresh_11c,
        'fpr_18f': fpr_18f, 'tpr_18f': tpr_18f, 'fpr_11c': fpr_11c, 'tpr_11c': tpr_11c,
        'onco_18f': onco_18f, 'nononco_18f': nononco_18f,
        'onco_11c': onco_11c, 'nononco_11c': nononco_11c
    }

if __name__ == '__main__':
    get_calculated_data()