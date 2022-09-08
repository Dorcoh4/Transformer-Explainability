#! /bin/sh
#SBATCH --job-name=bert-mask
#SBATCH --output=mask.out
#SBATCH --error=mask.err
#SBATCH --partition=studentbatch
#SBATCH --time=2500
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
export HF_DATASETS_CACHE="/home/joberant/NLP_2122/dorcoh4/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="/home/joberant/NLP_2122/dorcoh4/.cache/huggingface/"
python weight_map.py --model_params BERT_params/movies_bert.json


