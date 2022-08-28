#! /bin/sh
#SBATCH --job-name=movies-mask_6
#SBATCH --output=movies.out
#SBATCH --error=movies.err
#SBATCH --partition=studentbatch
#SBATCH --time=1500
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
export HF_DATASETS_CACHE="/home/joberant/NLP_2122/dorcoh4/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="/home/joberant/NLP_2122/dorcoh4/.cache/huggingface/"
PYTHONPATH=./:$PYTHONPATH python3 BERT_rationale_benchmark/metrics.py --data_dir data/movies/ --split test --results bert_models/movies/distilbert/identifier_results_k.json

