import importlib.util,subprocess,sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('record_analysis',ROOT/'scripts/analyze_record_predictions.py')
record_analysis=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(record_analysis)
def test_reported_results(): subprocess.run([sys.executable,str(ROOT/'scripts/validate_reported_results.py'),'--data-dir',str(ROOT/'data/aggregate')],check=True)
def test_summary_tables(tmp_path):
    subprocess.run([sys.executable,str(ROOT/'scripts/generate_summary_tables.py'),'--data-dir',str(ROOT/'data/aggregate'),'--output-dir',str(tmp_path)],check=True)
    assert {path.name for path in tmp_path.glob('*.csv')} == {'benchmark_model_metrics.csv','paired_accuracy_comparisons.csv','resnet50_confidence_intervals.csv','resnet50_three_seed_summary.csv'}
def test_aggregate_files_have_no_record_identifiers():
    forbidden={'image_id','sample_index','path','filename','patient_id','mrn'}
    for path in (ROOT/'data/aggregate').glob('*.csv'): assert forbidden.isdisjoint(pd.read_csv(path,nrows=1).columns)
def test_record_analysis():
    frame=pd.DataFrame({'sample_id':[1,2,3,4],'image_id':['a','b','c','d'],'model':['first']*4,'ground_truth':['x','x','y','y'],'prediction':['x','y','y','y']})
    second=frame.assign(model='second',prediction=['x','x','y','x'])
    assert record_analysis.hard_prediction_metrics(frame)['accuracy']==.75
    comparison=record_analysis.paired_cluster_comparison(frame,second,100,100,7)
    assert comparison['n_records']==4 and comparison['n_image_clusters']==4
