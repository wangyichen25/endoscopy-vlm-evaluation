#!/usr/bin/env python3
"""Regenerate all quantitative figures from the released aggregate data."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

COLORS={"Fine-tuned MedGemma-27B":"#2166AC","Fine-tuned MedGemma-4B":"#D89B25","ResNet-50 (seed 42)":"#D97706"}

def benchmark(data,out):
    frame=pd.read_csv(data/'benchmark_metrics.csv').sort_values(['accuracy','model'])
    labels=frame.model.tolist(); y=np.arange(len(frame)); colors=[COLORS.get(x,'#B8B8B8') for x in labels]
    fig,axes=plt.subplots(1,2,figsize=(14.8,9.9),sharey=True,facecolor='white')
    for ax,values,xlabel,title in ((axes[0],frame.accuracy.to_numpy()*100,'Accuracy (%)','A. Accuracy'),(axes[1],frame.macro_f1.to_numpy()*100,'Macro-F1 (%)','B. Macro-F1')):
        bars=ax.barh(y,values,color=colors,edgecolor='#5D5D5D',linewidth=.45); ax.set_xlim(0,100); ax.set_xlabel(xlabel,fontsize=10.5,labelpad=8); ax.set_title(title,fontsize=11.5,loc='left',pad=10); ax.xaxis.grid(True,color='#D8D8D8',linewidth=.65,linestyle='--'); ax.set_axisbelow(True); ax.spines[['top','right']].set_visible(False)
        for bar,value in zip(bars,values):
            if np.isfinite(value): ax.text(min(value+.8,98.5),bar.get_y()+bar.get_height()/2,f'{value:.1f}%',va='center',ha='left' if value<97 else 'right',fontsize=8.7,color='#222222')
    axes[0].set_yticks(y,labels,fontsize=9.3); axes[1].tick_params(axis='y',left=False,labelleft=False)
    fig.suptitle('Accuracy and macro-F1 on the internal held-out prompt benchmark',fontsize=13,y=.985); fig.text(.18,.948,'320 randomly sampled prompt-answer records from 303 held-out images',fontsize=9.5,color='#444444'); fig.text(.18,.026,'Macro-F1 is the unweighted mean across prespecified labels, retaining classes with F1 = 0.\nBlue/gold: fine-tuned MedGemma; orange: training-matched ResNet-50; gray: zero-shot generalist VLMs.',fontsize=8.1,color='#444444'); fig.subplots_adjust(left=.18,right=.975,top=.90,bottom=.105,wspace=.16); fig.savefig(out/'benchmark_accuracy_macro_f1.png',dpi=300,bbox_inches='tight',pad_inches=.12,facecolor='white'); plt.close(fig)

def distribution(data,out):
    frame=pd.read_csv(data/'development_class_counts.csv').sort_values('development_count'); fig,ax=plt.subplots(figsize=(8.8,9.2)); bars=ax.barh(frame.source_class,frame.development_count,color='#4C78A8',edgecolor='#2B4C6F',linewidth=.45); ax.set_xlim(0,980); ax.set_xlabel('Development images (n)'); fig.suptitle('Development-set source-label distribution',fontsize=13,y=.985); ax.xaxis.grid(True,color='#DADADA',linestyle='--',linewidth=.6); ax.set_axisbelow(True); ax.spines[['top','right']].set_visible(False)
    for bar,value in zip(bars,frame.development_count): ax.text(value+8,bar.get_y()+bar.get_height()/2,f'{value:,}',va='center',fontsize=8.5)
    fig.subplots_adjust(left=.36,right=.96,top=.92,bottom=.08); fig.savefig(out/'development_class_distribution.png',dpi=300,bbox_inches='tight',pad_inches=.12,facecolor='white'); plt.close(fig)

def finetuned_confusion(data,out,stratum):
    frame=pd.read_csv(data/f'finetuned_medgemma_{stratum}_sample_confusion.csv'); models=['Fine-tuned MedGemma-27B','Fine-tuned MedGemma-4B']; labels=frame.true_label.drop_duplicates().tolist(); columns=labels+['Other']; fig,axes=plt.subplots(1,2,figsize=(17,8.6 if stratum=='high' else 10.2),sharey=True)
    for ax,model in zip(axes,models):
        matrix=frame[frame.model.eq(model)].pivot(index='true_label',columns='predicted_label',values='row_proportion').reindex(index=labels,columns=columns); sns.heatmap(matrix,ax=ax,cmap='Blues',vmin=0,vmax=1,annot=True,fmt='.2f',cbar=False); ax.set_title(model); ax.set_xlabel('Predicted source label'); ax.set_ylabel('True source label'); ax.tick_params(axis='x',labelrotation=52,labelsize=8)
    fig.tight_layout(); fig.savefig(out/f'finetuned_medgemma_{stratum}_sample_confusion.png',dpi=300,bbox_inches='tight',facecolor='white'); plt.close(fig)

def resnet_confusion(data,out):
    counts=pd.read_csv(data/'resnet50_seed42_confusion_counts.csv',index_col=0); norm=pd.read_csv(data/'resnet50_seed42_confusion_row_normalized.csv',index_col=0); fig,axes=plt.subplots(1,2,figsize=(18,8.8),constrained_layout=True); sns.heatmap(counts,ax=axes[0],cmap='Blues',cbar_kws={'label':'Images'},square=True); sns.heatmap(norm,ax=axes[1],cmap='Blues',vmin=0,vmax=1,cbar_kws={'label':'Within-class proportion'},square=True); axes[0].set_title('A. Counts'); axes[1].set_title('B. Row-normalized proportions')
    for ax in axes: ax.set_xlabel('Predicted source class'); ax.set_ylabel('True source class'); ax.tick_params(axis='x',labelrotation=55,labelsize=7); ax.tick_params(axis='y',labelrotation=0,labelsize=7)
    fig.suptitle('ResNet-50 direct 23-class classification on 2,122 held-out images (seed 42)',fontsize=14); fig.savefig(out/'resnet50_seed42_confusion.png',dpi=300,bbox_inches='tight',facecolor='white'); plt.close(fig)

def error_profile(data,out):
    frame=pd.read_csv(data/'resnet50_seed42_per_class_metrics.csv'); frame['predicted_count']=frame.tp+frame.fp; total=frame.support.sum(); frame['true_share']=100*frame.support/total; frame['predicted_share']=100*frame.predicted_count/total; frame['difference']=frame.predicted_share-frame.true_share; rho=frame.support.rank().corr(frame.sensitivity.rank()); selected=frame.assign(abs_difference=frame.difference.abs()).nlargest(8,'abs_difference').sort_values('difference'); fig,axes=plt.subplots(1,2,figsize=(16,8.6)); axes[0].scatter(frame.support,100*frame.sensitivity,color='#2B6CB0'); axes[0].set_xscale('log'); axes[0].set_xlabel('Held-out images in source class (log scale)'); axes[0].set_ylabel('Sensitivity (%)'); axes[0].set_title(f'A. Sensitivity versus support (Spearman rho = {rho:.2f})'); y=np.arange(len(selected)); axes[1].barh(y-.18,selected.true_share,height=.35,label='True',color='#2B6CB0'); axes[1].barh(y+.18,selected.predicted_share,height=.35,label='Predicted',color='#D97706'); axes[1].set_yticks(y,labels=selected.display_label); axes[1].set_xlabel('Share of held-out images (%)'); axes[1].set_title('B. Largest class-share discrepancies'); axes[1].legend(frameon=False)
    for ax in axes: ax.grid(axis='y',color='#D9E0E7',linewidth=.8); ax.spines[['top','right']].set_visible(False)
    fig.suptitle('ResNet-50 held-out error profile',fontsize=16); fig.tight_layout(); fig.savefig(out/'resnet50_seed42_error_profile.png',dpi=300,bbox_inches='tight',facecolor='white'); plt.close(fig)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--data-dir',type=Path,default=Path('data/aggregate')); parser.add_argument('--output-dir',type=Path,default=Path('figures')); args=parser.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True); benchmark(args.data_dir,args.output_dir); distribution(args.data_dir,args.output_dir); finetuned_confusion(args.data_dir,args.output_dir,'high'); finetuned_confusion(args.data_dir,args.output_dir,'low'); resnet_confusion(args.data_dir,args.output_dir); error_profile(args.data_dir,args.output_dir)
if __name__=='__main__': main()
