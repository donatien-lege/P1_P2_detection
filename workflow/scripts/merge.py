import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
from scipy.stats import t

def get_path(file):
    steps = file.split("_")
    path = ' '.join((steps[-2], steps[-1]))
    return path.split('.csv')[0]

def col(x, dico):
    return plt.get_cmap("nipy_spectral")(x/len(dico))

def format_ax(ax, title, xlabel=''):
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probabilité")
    
def error(arr):
    mean = np.mean(arr)
    inf, sup = t.interval(0.95, len(arr), mean, np.std(arr))
    return mean, sup-mean
    
def plot_error(dico, key_1, key_2, save, title):
    errors = {}
    for df in dico:
        P1, P2 = np.abs(dico[df][key_1]), np.abs(dico[df][key_2])
        errors[df] = (*error(P1), *error(P2))
    errors = pd.DataFrame(errors).T
    errors.columns = ("MAE_P1", "err_P1", "MAE_P2", "err_P2")
    sns.heatmap(errors, fmt=".3g", annot=True,  cmap='Blues')
    plt.title(title)
    plt.savefig(save, bbox_inches='tight')
    plt.clf()
    return errors
    
def plot_ratio(dico, key, save, title):
    errors = {}
    for df in dico:
        r_12 = np.abs(dico[df][key])
        errors[df] = error(r_12)
    
    errors = pd.DataFrame(errors).T
    errors.columns = ("MAE_ratio", "err_ratio")
    sns.heatmap(errors, fmt=".3g", annot=True, cmap='Blues')
    plt.title(title)
    plt.savefig(save, bbox_inches='tight')
    plt.clf()
    return errors

def stats(df):
    TP = 100*(sum(df['RP'] & df['RA']))/sum(df['RA'])
    TN = 100*(sum(~df['RP'] & ~df["NAN"] & ~df['RA']))/sum(~df['RA'])
    FP, FN = 100-TP, 100-TN
    precision = 100*(sum(df['RP'] == df['RA']) - sum(df["NAN"]))/len(df)
    metrics = {'FP': FP, 'FN': FN, 'acc': precision}
    return metrics


def merge(files, hz, vt, ratio, clasf, metrics):
    dico = defaultdict(list)
    
    for file in files:
        dico[get_path(file)].append(pd.read_csv(file))
    
    for key in dico:
        dico[key] = pd.concat(dico[key]).reset_index(drop=True)
        
    #MAE on time detection
    he = plot_error(dico, 
               key_1='hz_P1',
               key_2='hz_P2',
               save=hz,
               title="MAE peaks")
               
    #MAE on amplitude
    ve = plot_error(dico, 
               key_1='vt_P1',
               key_2='vt_P2',
               save=vt,
               title="MAE peaks")

    #MAE on P1/P2 ratio
    re = plot_ratio(dico, 
               key='Ratio',
               save=ratio,
               title="MAE ratio")
    
    df_classif = pd.DataFrame({k: stats(dico[k]) for k in dico}).T
    plt.figure(figsize=(10, 10))
    df_classif.plot.bar(rot=0)
    plt.xticks(rotation = 45)
    plt.savefig(clasf, bbox_inches='tight')
    df_tot = pd.concat((he, ve, re, df_classif), axis=1)
    print(df_tot)
    df_tot.to_csv(metrics)


merge(snakemake.input["error"],
      snakemake.output["hz"],
      snakemake.output["vt"],
      snakemake.output["ratio"],
      snakemake.output["clasf"],
      snakemake.output["metrics"])

