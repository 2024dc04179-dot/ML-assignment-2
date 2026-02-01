"""Visualization utilities"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    
    classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
    labels = [f'Grade {int(c)}' for c in classes]
    
    cmap = sns.light_palette("#6366f1", as_cmap=True)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'}, linewidths=2, linecolor='white',
                annot_kws={'size': 14, 'weight': 'bold', 'color': '#1e293b'})
    
    ax.set_xlabel('Predicted', fontsize=13, fontweight='bold', color='#1e293b', labelpad=10)
    ax.set_ylabel('Actual', fontsize=13, fontweight='bold', color='#1e293b', labelpad=10)
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15, color='#1e293b')
    
    ax.tick_params(colors='#475569', labelsize=11)
    plt.setp(ax.get_xticklabels(), color='#475569', fontweight='500')
    plt.setp(ax.get_yticklabels(), color='#475569', fontweight='500')
    
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_tick_params(color='#475569')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#475569')
    cbar.set_label('Count', color='#475569', fontweight='600')
    
    plt.tight_layout()
    return fig


def plot_metrics_comparison(results_dict):
    """Compare model metrics"""
    import pandas as pd
    
    df = pd.DataFrame(results_dict).T
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.patch.set_facecolor('#ffffff')
    
    metrics = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
    colors = ['#6366f1', '#8b5cf6', '#06b6d4', '#10b981', '#f59e0b', '#ef4444']
    
    for idx, (metric, ax) in enumerate(zip(metrics, axes.flatten())):
        ax.set_facecolor('#f8fafc')
        
        values = df[metric].values
        models = df.index.tolist()
        
        bars = ax.barh(models, values, color=colors[idx], edgecolor='white', linewidth=2,
                      alpha=0.85, height=0.6)
        
        ax.set_xlabel(metric, fontsize=11, fontweight='bold', color='#1e293b')
        ax.set_xlim(0, 1.1)
        ax.axvline(x=values.max(), color='#22c55e', linestyle='--', alpha=0.8, linewidth=2)
        
        for bar, val in zip(bars, values):
            ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
                   f'{val:.3f}', va='center', fontsize=10, color='#1e293b',
                   fontweight='600')
        
        ax.tick_params(colors='#475569', labelsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#e2e8f0')
        ax.spines['left'].set_color('#e2e8f0')
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, linestyle='--', alpha=0.3, color='#cbd5e1')
        plt.setp(ax.get_yticklabels(), color='#475569', fontsize=10, fontweight='500')
        plt.setp(ax.get_xticklabels(), color='#64748b')
    
    plt.suptitle('Model Performance Comparison', fontsize=18, fontweight='bold', 
                 y=0.98, color='#1e293b')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


