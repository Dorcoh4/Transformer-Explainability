import os
import sys
import torch
from itertools import chain
import weight_map
import argparse
from BERT_rationale_benchmark.utils import load_documents, load_datasets, annotations_from_jsonl

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
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
    data_dir = weight_map.data_dir
    train, val, test = load_datasets(data_dir)
    docids = set(e.docid for e in
                 chain.from_iterable(chain.from_iterable(map(lambda ann: ann.evidences, chain(train, val, test)))))
    documents = load_documents(data_dir, docids)
    evidence_classifier, word_interner, de_interner, evidence_classes, tokenizer = weight_map.load_classifier(args.model_params)
    cache = os.path.join(args.output_dir, 'preprocessed.pkl')
    if os.path.exists(cache):
        print(f'Loading interned documents from {cache}')
        (interned_documents) = torch.load(cache)
    annotations = None
    weight_map.args_data_dir = args.data_dir

    dir = weight_map.directory
    onlyfiles = [f for f in os.listdir(dir) if f.endswith(".pt") and "99" in f and os.path.isfile(os.path.join(dir, f))]
    def test_masker(masker):
        return weight_map.epoch_validation(-3, masker, evidence_classifier, tokenizer, test, word_interner, de_interner, evidence_classes,
                         interned_documents, documents, annotations)
    scores = [test_masker(torch.load(dir + f, map_location=device)) for f in onlyfiles]
    print(sorted(zip(scores, onlyfiles), key=lambda pair: pair[0]))



if __name__ == '__main__':
    sys.exit(main())


