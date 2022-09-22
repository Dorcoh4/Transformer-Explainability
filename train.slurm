#! /bin/sh
#SBATCH --job-name=movies-mask__bert_d2_lr
#SBATCH --output=movies__bert_d2_lr.out
#SBATCH --error=movies__bert_d2_lr.err
#SBATCH --partition=studentbatch
#SBATCH --time=1500
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
export HF_DATASETS_CACHE="/home/joberant/NLP_2122/dorcoh4/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="/home/joberant/NLP_2122/dorcoh4/.cache/huggingface/"
PYTHONPATH=./:$PYTHONPATH python BERT_rationale_benchmark/models/pipeline/weight_map.py --model_params BERT_params/movies_bert.json --output_dir bert_models/movies/ --data_dir data/movies/