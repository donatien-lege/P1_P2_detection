import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def eval(pred_path, curve_path, annot_path, out_path):
    
    #chargements
    curve = np.load(curve_path, allow_pickle=True)
    preds = np.load(pred_path, allow_pickle=True).astype(int)
    peaks = np.load(annot_path, allow_pickle=True).astype(int)
    df_error = pd.DataFrame()
    
    # Supprime les courbes sans P1/P2
    idx = np.arange(len(curve))
    p1, p2 = peaks[:, 0], peaks[:, 1]
    div = curve[idx, p1] / curve[idx, p2]
    mask = (div > 0.1) & (div < 10)
    peaks = peaks[mask]
    preds = preds[mask]
    curve = curve[mask]
    
    missed = preds[:, 0] == preds[:, 1]

    # Ratio
    idx = range(len(curve))
    preds_P1, preds_P2 = curve[idx, preds[:, 0]], curve[idx, preds[:, 1]]
    peaks_P1, peaks_P2 = curve[idx, peaks[:, 0]], curve[idx, peaks[:, 1]]
    missed = np.where(preds_P1 == preds_P2)
    preds_P1[missed] = np.nan
    preds_P2[missed] = np.nan
    
    ratio_pred = preds_P1 / preds_P2
    ratio_annot = peaks_P1 / peaks_P2
    df_error["Ratio"] = ratio_pred - ratio_annot
    df_error["RP"] = ratio_pred > 1
    df_error["RA"] = ratio_annot > 1
    df_error["NAN"] = np.isnan(ratio_pred)

    # Erreurs horizontales
    df_error["hz_P1"] = preds[:, 0] - peaks[:, 0]
    df_error["hz_P2"] = preds[:, 1] - peaks[:, 1]
            
    # Erreurs verticales
    df_error["vt_P1"] = preds_P1 - peaks_P2
    df_error["vt_P2"] = preds_P2 - peaks_P2
    
    # Supprime les lignes avec valeurs manquantes
    df_error = df_error.replace([-np.inf, np.inf], np.nan)
    df_error = df_error.dropna()
    
    df_error.to_csv(out_path, index=False)
    
eval(snakemake.input['pred'], 
     snakemake.input['pulse'], 
     snakemake.input['annot'], 
     snakemake.output[0])