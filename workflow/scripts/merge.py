import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns

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
    
def plot_error(dico, key_1, key_2, save, title):
    errors = {}
    for df in dico:
        P1, P2 = np.abs(dico[df][key_1]), np.abs(dico[df][key_2])
        errors[df] = (np.mean(P1), np.std(P1), np.mean(P2), np.std(P2))
    
    errors = pd.DataFrame(errors).T
    errors.columns = ("MAE_P1", "std_P1", "MAE_P2", "std_P2")
    sns.heatmap(errors, fmt=".3g", annot=True,  cmap='Blues')
    plt.title(title)
    plt.savefig(save, bbox_inches='tight')
    plt.clf()
    
def plot_ratio(dico, key, save, title):
    errors = {}
    for df in dico:
        r_12 = np.abs(dico[df][key])
        errors[df] = (np.mean(r_12), np.std(r_12))
    
    errors = pd.DataFrame(errors).T
    errors.columns = ("MAE_ratio", "std_ratio")
    sns.heatmap(errors, fmt=".3g", annot=True, cmap='Blues')
    plt.title(title)
    plt.savefig(save, bbox_inches='tight')
    plt.clf()

    
def stats(df):
    TP = (sum(df['RP'] & df['RA']))/sum(df['RA'])
    TN = (sum(~df['RP'] & ~df["NAN"] & ~df['RA']))/sum(~df['RA'])
    FP, FN = 1-TP, 1-TN
    precision = (sum(df['RP'] == df['RA']) - sum(df["NAN"]))/len(df)
    metrics = {'FP': FP, 'FN': FN, 'prec': precision}
    return metrics


def merge(files, vt, ratio, clasf, metrics):
    
    dico = defaultdict(list)
    
    for file in files:
        dico[get_path(file)].append(pd.read_csv(file))
    
    for key in dico:
        dico[key] = pd.concat(dico[key]).reset_index(drop=True)
        
    #Erreurs verticales
    plot_error(dico, 
               key_1='vt_P1',
               key_2='vt_P2',
               save=vt,
               title="MAE peaks")

    #Erreurs ratio
    plot_ratio(dico, 
               key='Ratio',
               save=ratio,
               title="MAE ratio")
    
    df_classif = pd.DataFrame({k: stats(dico[k]) for k in dico}).T
    plt.figure(figsize=(10, 10))
    df_classif.plot.bar(rot=0)
    plt.xticks(rotation = 45)
    plt.savefig(clasf, bbox_inches='tight')
    print(df_classif)
    df_classif.to_csv(metrics)


merge(snakemake.input["error"],
      snakemake.output["vt"],
      snakemake.output["ratio"],
      snakemake.output["clasf"],
      snakemake.output["metrics"])

