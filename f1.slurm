#! /bin/sh
#SBATCH --job-name=f1_mask
#SBATCH --output=f1.out
#SBATCH --error=f1.err
#SBATCH --partition=studentbatch
#SBATCH --time=150
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
export HF_DATASETS_CACHE="/home/joberant/NLP_2122/dorcoh4/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="/home/joberant/NLP_2122/dorcoh4/.cache/huggingface/"
PYTHONPATH=./:$PYTHONPATH python3 BERT_rationale_benchmark/metrics.py --data_dir data/movies/ --split test --results bert_models/movies/distilbert/identifier_results_5.json
PYTHONPATH=./:$PYTHONPATH python3 BERT_rationale_benchmark/metrics.py --data_dir data/movies/ --split test --results bert_models/movies/distilbert/identifier_results_10.json
PYTHONPATH=./:$PYTHONPATH python3 BERT_rationale_benchmark/metrics.py --data_dir data/movies/ --split test --results bert_models/movies/distilbert/identifier_results_30.json
PYTHONPATH=./:$PYTHONPATH python3 BERT_rationale_benchmark/metrics.py --data_dir data/movies/ --split test --results bert_models/movies/distilbert/identifier_results_65.json
PYTHONPATH=./:$PYTHONPATH python3 BERT_rationale_benchmark/metrics.py --data_dir data/movies/ --split test --results bert_models/movies/distilbert/identifier_results_80.json

