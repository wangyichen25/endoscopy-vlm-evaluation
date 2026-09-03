#!/usr/bin/env python3
"""Validate released aggregate data against values reported in the final manuscript."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd

def close(actual,expected,label,tol=5e-4):
    if not np.isclose(actual,expected,rtol=0,atol=tol): raise AssertionError(f'{label}: expected {expected}, found {actual}')
def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--data-dir',type=Path,default=Path('data/aggregate')); args=parser.parse_args(); data=args.data_dir
    benchmark=pd.read_csv(data/'benchmark_metrics.csv'); assert len(benchmark)==25 and benchmark.model.nunique()==25; assert benchmark[['accuracy','macro_f1']].notna().all().all(); assert benchmark[['accuracy','macro_f1']].apply(lambda column: column.between(0,1).all()).all()
    expected={'Fine-tuned MedGemma-27B':(.928125,.805),'Fine-tuned MedGemma-4B':(.925,.839),'ResNet-50 (seed 42)':(.840625,.6969699842167872)}
    for model,(accuracy,macro_f1) in expected.items():
        row=benchmark[benchmark.model.eq(model)].iloc[0]; close(row.accuracy,accuracy,f'{model} accuracy'); close(row.macro_f1,macro_f1,f'{model} macro-F1'); close(row.evaluable_count,320,f'{model} evaluable count',0)
    counts=pd.read_csv(data/'development_class_counts.csv'); assert len(counts)==23 and counts.development_count.sum()==8540
    per_class=pd.read_csv(data/'resnet50_seed42_per_class_metrics.csv'); assert len(per_class)==23 and per_class.support.sum()==2122 and (per_class.sensitivity==0).sum()==7
    seeds=pd.read_csv(data/'resnet50_three_seed_metrics.csv'); assert sorted(seeds.seed.astype(int).tolist())==[17,42,123]; primary=seeds[seeds.seed.eq(42)].iloc[0]; close(primary['prompt_benchmark__accuracy'],.840625,'seed-42 accuracy'); close(primary['prompt_benchmark__macro_f1'],.6969699842167872,'seed-42 macro-F1')
    audit=json.loads((data/'split_audit_summary.json').read_text()); assert (audit['development_images'],audit['held_out_images'],audit['total_images'],audit['exact_cross_split_duplicates'],audit['near_pair_threshold_counts']['both_phash_le_4'])==(8540,2122,10662,0,1)
    statistics=json.loads((data/'statistical_results.json').read_text())
    benchmark_ci=statistics['confidence_intervals']['prompt_benchmark']['accuracy']; close(benchmark_ci['lower_95'],.8005981595092025,'seed-42 benchmark accuracy CI lower'); close(benchmark_ci['upper_95'],.8802616296384537,'seed-42 benchmark accuracy CI upper')
    image_ci=statistics['confidence_intervals']['image_full']['accuracy']; close(image_ci['lower_95'],.644674835061263,'seed-42 image accuracy CI lower'); close(image_ci['upper_95'],.6842601319509897,'seed-42 image accuracy CI upper')
    paired=statistics['paired_comparisons']['benchmark_prompts_320']; close(paired['accuracy_difference_resnet50_minus_finetuned_medgemma_27b'],-.0875,'paired benchmark accuracy difference'); close(paired['difference_lower_95'],-.12576735347455337,'paired difference CI lower'); close(paired['difference_upper_95'],-.050473186119873815,'paired difference CI upper'); assert paired['paired_cluster_permutation_p'] < .001
    direct=statistics['primary_resnet50_metrics']['image_full']; close(direct['accuracy'],.6649387370405277,'direct image accuracy'); close(direct['macro_f1'],.4312999712301691,'direct image macro-F1'); close(direct['probability_macro_auroc_ovr'],.9760173857345962,'direct image macro-AUROC')
    for stratum,rows in [('high',180),('low',420)]:
        frame=pd.read_csv(data/f'finetuned_medgemma_{stratum}_sample_confusion.csv'); assert len(frame)==rows; assert np.allclose(frame.groupby(['model','true_label']).row_proportion.sum(),1)
    print('PASS: aggregate release data match final manuscript values')
if __name__=='__main__': main()
