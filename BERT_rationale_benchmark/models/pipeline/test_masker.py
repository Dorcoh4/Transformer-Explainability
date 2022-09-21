import argparse
import os
import sys
from itertools import chain

import torch
from sklearn.model_selection import train_test_split
from transformers import AutoModelForTokenClassification

import weight_map
from BERT_rationale_benchmark.utils import load_datasets, load_documents

from BERT_rationale_benchmark.utils import annotations_from_jsonl


def main():
    parser = argparse.ArgumentParser(description=""" FORDOR

            """, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--data_dir', dest='data_dir', required=True,
                        help='Which directory contains a {train,val,test}.jsonl file?')
    parser.add_argument('--output_dir', dest='output_dir', required=True,
                        help='Where shall we write intermediate models + final data to?')
    parser.add_argument('--model_params', dest='model_params', required=True,
                        help='JSoN file for loading arbitrary model parameters (e.g. optimizers, pre-saved files, etc.')

    args = parser.parse_args()

    train, val, test = load_datasets(weight_map.data_dir)
    docids = set(e.docid for e in
                 chain.from_iterable(chain.from_iterable(map(lambda ann: ann.evidences, chain(train, val, test)))))
    documents = load_documents(weight_map.data_dir, docids)
    # imdb_data = load_dataset("imdb")
    # train_dataset = imdb_data['train']
    # print(f"IMDB dataset - train:{len(imdb_data['train'])}, test:{imdb_data['test']}")
    train_dataset = weight_map.convert_dataset(train, documents, "train")
    val_dataset = weight_map.convert_dataset(val, documents, "validation")
    test_dataset = weight_map.convert_dataset(test, documents, "test")

    evidence_classifier, word_interner, de_interner, evidence_classes, tokenizer = weight_map.load_classifier(args.model_params)

    # train_dataset = weight_map.tokenize_dataset(train_dataset, tokenizer)
    # val_dataset = tokenize_dataset(val_dataset)
    # test_dataset = tokenize_dataset(test_dataset)

    cache = os.path.join(args.output_dir, 'preprocessed.pkl')
    if os.path.exists(cache):
        print(f'Loading interned documents from {cache}')
        (interned_documents) = torch.load(cache)
    annotations = annotations_from_jsonl(os.path.join(args.data_dir, 'test' + '.jsonl'))
    masker = weight_map.load_masker("7_bert_0_adamw")
    weight_map.epoch_validation(0, masker, evidence_classifier, tokenizer, test, word_interner, de_interner, evidence_classes,
                         interned_documents, documents, annotations, range(5, 85, 5))

if __name__ == '__main__':
    sys.exit(main())
