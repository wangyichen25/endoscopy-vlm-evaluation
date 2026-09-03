import importlib.util,subprocess,sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('record_analysis',ROOT/'scripts/analyze_record_predictions.py')
record_analysis=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(record_analysis)
def test_reported_results(): subprocess.run([sys.executable,str(ROOT/'scripts/validate_reported_results.py'),'--data-dir',str(ROOT/'data/aggregate')],check=True)
def test_benchmark_metrics_are_complete(): assert pd.read_csv(ROOT/'data/aggregate/benchmark_metrics.csv')[['accuracy','macro_f1']].notna().all().all()
def test_summary_tables(tmp_path):
    subprocess.run([sys.executable,str(ROOT/'scripts/generate_summary_tables.py'),'--data-dir',str(ROOT/'data/aggregate'),'--output-dir',str(tmp_path)],check=True)
    assert {path.name for path in tmp_path.glob('*.csv')} == {'benchmark_model_metrics.csv','paired_accuracy_comparisons.csv','resnet50_confidence_intervals.csv','resnet50_three_seed_summary.csv'}
def test_reproduction_entrypoint_exists(): assert (ROOT/'scripts/reproduce_release.py').is_file()
def test_record_release_completeness():
    benchmark=pd.read_csv(ROOT/'data/records/benchmark_predictions.csv')
    attempts=pd.read_csv(ROOT/'data/records/benchmark_response_attempts.csv')
    medgemma=pd.read_csv(ROOT/'data/records/medgemma_full_predictions.csv')
    resnet_prompt=pd.read_csv(ROOT/'data/records/resnet50_prompt_predictions.csv')
    resnet_image=pd.read_csv(ROOT/'data/records/resnet50_image_predictions.csv')
    assert (len(benchmark),benchmark.model.nunique(),benchmark.sample_id.nunique())==(8000,25,320)
    assert (len(attempts),attempts.model.nunique(),attempts.sample_id.nunique())==(8186,24,320)
    assert (len(medgemma),medgemma.model.nunique(),medgemma.sample_id.nunique())==(25552,4,6388)
    assert (len(resnet_prompt),resnet_prompt.seed.nunique(),resnet_prompt.sample_id.nunique())==(19164,3,6388)
    assert (len(resnet_image),resnet_image.seed.nunique(),resnet_image.image_id.nunique())==(6366,3,2122)
    forbidden={'path','relative_path','absolute_path','patient_id','mrn'}
    for frame in (benchmark,attempts,medgemma,resnet_prompt,resnet_image): assert forbidden.isdisjoint(frame.columns)
def test_reconstructed_benchmark_matches_release(tmp_path):
    subprocess.run([sys.executable,str(ROOT/'scripts/reconstruct_record_level_results.py'),'--data-dir',str(ROOT/'data/records'),'--output-dir',str(tmp_path),'--bootstrap-replicates','20','--permutation-replicates','20'],check=True)
    reconstructed=pd.read_csv(tmp_path/'benchmark_metrics.csv').set_index('model').sort_index()
    reported=pd.read_csv(ROOT/'data/aggregate/benchmark_metrics.csv').set_index('model').sort_index()
    columns=['evaluable_count','accuracy','macro_sensitivity','macro_specificity','macro_ppv','macro_npv','macro_auroc','average_latency_seconds','average_cost_usd']
    pd.testing.assert_frame_equal(reconstructed[columns],reported[columns],check_exact=False,check_dtype=False,atol=1e-12,rtol=0)
    assert (reconstructed.macro_f1-reported.macro_f1).abs().max()<5e-4
def test_reconstructed_statistics_match_release(tmp_path):
    import json
    subprocess.run([sys.executable,str(ROOT/'scripts/reconstruct_record_level_results.py'),'--data-dir',str(ROOT/'data/records'),'--output-dir',str(tmp_path)],check=True)
    reconstructed=json.loads((tmp_path/'statistical_tests.json').read_text())
    reported=json.loads((ROOT/'data/aggregate/statistical_results.json').read_text())
    assert reconstructed['resnet50_confidence_intervals']==reported['confidence_intervals']
    for scope,key in [('image_only','image_only_2122'),('all_prompts','all_prompts_6388'),('benchmark','benchmark_prompts_320')]:
        actual=reconstructed['paired_resnet50_minus_finetuned_medgemma_27b'][scope]
        expected=reported['paired_comparisons'][key]
        assert actual['accuracy_difference_first_minus_second']==expected['accuracy_difference_resnet50_minus_finetuned_medgemma_27b']
        assert actual['first_accuracy']==expected['resnet50_accuracy']
        assert actual['second_accuracy']==expected['finetuned_medgemma_27b_accuracy']
        for field in ['difference_lower_95','difference_upper_95','paired_cluster_permutation_p','n_records','n_image_clusters']:
            assert actual[field]==expected[field]
def test_aggregate_files_have_no_record_identifiers():
    forbidden={'image_id','sample_index','path','filename','patient_id','mrn'}
    for path in (ROOT/'data/aggregate').glob('*.csv'): assert forbidden.isdisjoint(pd.read_csv(path,nrows=1).columns)
def test_record_analysis():
    frame=pd.DataFrame({'sample_id':[1,2,3,4],'image_id':['a','b','c','d'],'model':['first']*4,'ground_truth':['x','x','y','y'],'prediction':['x','y','y','y']})
    second=frame.assign(model='second',prediction=['x','x','y','x'])
    assert record_analysis.hard_prediction_metrics(frame)['accuracy']==.75
    comparison=record_analysis.paired_cluster_comparison(frame,second,100,100,7)
    assert comparison['n_records']==4 and comparison['n_image_clusters']==4
