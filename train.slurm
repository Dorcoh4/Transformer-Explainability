#! /bin/sh
#SBATCH --job-name=d__bert_dyn_l
#SBATCH --output=movies__bert_dyn_l.out
#SBATCH --error=movies__bert_dyn_l.err
#SBATCH --partition=studentbatch
#SBATCH --time=1500
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
export HF_DATASETS_CACHE="/home/joberant/NLP_2122/dorcoh4/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="/home/joberant/NLP_2122/dorcoh4/.cache/huggingface/"
PYTHONPATH=./:$PYTHONPATH python BERT_rationale_benchmark/models/pipeline/weight_map.py --model_params BERT_params/movies_bert.json --output_dir bert_models/movies/ --data_dir data/movies/