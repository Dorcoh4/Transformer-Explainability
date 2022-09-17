import argparse
import json
import os
import sys
from itertools import chain

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from datasets import load_metric
import torch
from transformers import AutoModelForSequenceClassification
from transformers import Trainer
from transformers import TrainingArguments
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification
from torch.optim import AdamW
from torch.optim import SGD
# import pyarrow as pa
# import pyarrow.parquet as pq
# import pyarrow.dataset as ds
from datasets import Dataset
from transformers import get_scheduler
from tqdm.auto import tqdm
from captum.attr import (
    visualization
)
# import torchviz
import distilbert_pipeline
from BERT_rationale_benchmark.utils import load_documents, load_datasets, annotations_from_jsonl
from BERT_rationale_benchmark.metrics import score_hard_rationale_predictions, Rationale
from BERT_explainability.modules.BERT.ExplanationGenerator import Generator

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# directory = "/home/joberant/NLP_2122/dorcoh4/weight_map/"
# data_dir = "/home/joberant/NLP_2122/dorcoh4/weight_map/movies"
directory = "C:/Users/Dor_local/Downloads/" if 'win' in sys.platform else "/home/joberant/NLP_2122/dorcoh4/weight_map/"
data_dir = "C:/Users/Dor_local/Downloads/movies.tar/movies/" if 'win' in sys.platform else "/home/joberant/NLP_2122/dorcoh4/weight_map/movies/"

suffix = "_p3_mv"

best_validation_score = 0
best_validation_epoch = 0

distilbert_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def convert_dataset(raw_dataset, documents, name, imdb_data=None):
    texts = []
    labels = []
    for line in raw_dataset:
        if line.annotation_id in documents:  # FORDOR this if
            # sentence_list = [" ".join(sent) for sent in documents[line.annotation_id]]
            # texts.append("\n".join(sentence_list))
            texts.append(documents[line.annotation_id])
            labels.append(0 if line.classification.upper() == 'NEG' else 1)
    if imdb_data is not None:
        texts += imdb_data['text']
        labels += imdb_data['label']
    # file_name = f"eraser_movies_{name}.parquet"
    # table = pa.table({'text': texts,
    #                   'label': labels,})
    # pq.write_table(table, file_name)
    # dataset = ds.dataset(file_name)
    dataset = Dataset.from_dict({'text': texts, 'label': labels})
    return dataset


def tokenize_dataset(raw_dataset, tokenizer=None):
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    return raw_dataset.shuffle(seed=42).map(tokenize_function, batched=True)


def train_classifier(train_dataset, eval_dataset):
    metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        logits = eval_pred.predictions
        labels = eval_pred.label_ids
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    model.train()

    training_args = TrainingArguments("DAN",
                                      # YOUR CODE HERE
                                      num_train_epochs=4,  # must be at least 10.
                                      per_device_train_batch_size=8,
                                      per_device_eval_batch_size=8,
                                      learning_rate=0.00005,
                                      # END YOUR END

                                      save_total_limit=2,
                                      log_level="error",
                                      evaluation_strategy="epoch")

    trainer = Trainer(
        model=model,
        # data_collator=co,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    torch.save(model, directory + 'imdb_classifier.pt')
    print("classifier trained")

    return model


def load_classifier(model_params):
    model = torch.load(directory + 'imdb_classifier.pt', map_location=device)
    with open(model_params, 'r') as fp:
        print(f'Loading model parameters from {model_params}')
        model_params = json.load(fp)
        print(f'Params: {json.dumps(model_params, indent=2, sort_keys=True)}')
    evidence_classifier, word_interner, de_interner, evidence_classes, tokenizer = \
        distilbert_pipeline.initialize_models(model_params, batch_first=True)
    evidence_classifier.eval()
    return model, word_interner, de_interner, evidence_classes, tokenizer


def train_masker(classifier, classify_tokenizer, train_dataset, val, word_interner, de_interner, evidence_classes, interned_documents, documents, annotations):
    train_dataset = train_dataset.remove_columns(["text"])

    # train_dataset = train_dataset.remove_columns(["label"])

    train_dataset.set_format("torch")
    batch_size = 4
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

    mask_model = AutoModelForTokenClassification.from_pretrained("distilbert-base-uncased", num_labels=1)

    optimizer = SGD(mask_model.parameters(), lr=5e-5, momentum=0.9)

    num_epochs = 100
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    classifier.to(device)
    mask_model.to(device)

    classifier.eval()
    for param in classifier.parameters():
        param.requires_grad = False

    for param in mask_model.distilbert.parameters():
        param.requires_grad = False
    for param in mask_model.distilbert.transformer.layer[5].parameters():
        param.requires_grad = True

    mask_model.train()
    crossEntropyLoss = torch.nn.CrossEntropyLoss()
    softmax = torch.nn.Softmax(dim=1)
    # nllLoss = torch.nn.NLLLoss()
    sigmoid = torch.nn.Sigmoid()
    tanh = torch.nn.Tanh()

    progress_bar = tqdm(range(num_training_steps))
    lambda1 = 0.001
    for epoch in range(num_epochs):
        running_loss = 0
        running_loss_ce = 0
        running_loss_mask = 0
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop('label', None)
            g_out = mask_model(**batch)
            mask = sigmoid(g_out.logits)
            attention_mask = batch['attention_mask']
            unrelated_tokens = attention_mask.detach().clone()
            # sep_locs = [attention_mask[r].tolist().index(0) - 1 if attention_mask[r][-1] == 0 else len(attention_mask[r]) - 1 for r in range(len(attention_mask))]
            unrelated_tokens = unrelated_tokens.roll(-1, 1)
            unrelated_tokens[:, 0] = 0
            unrelated_tokens[:, -1] = 0

            # unrelated_tokens[:,sep_locs] = 0
            unrelated_tokens = unrelated_tokens.unsqueeze(2)
            relevant_mask = mask * unrelated_tokens
            mask = relevant_mask + ~(unrelated_tokens.bool())
            masked_in = classifier.distilbert.embeddings(batch['input_ids']) * mask

            # out1 = classifier(**batch)
            batch.pop('input_ids', None)
            batch['inputs_embeds'] = masked_in
            out2 = classifier(**batch)
            # original_probs = softmax(out1.logits)
            masked_probs = softmax(out2.logits)

            ce_loss = crossEntropyLoss(out2.logits, masked_probs)
            mask_loss = torch.norm(relevant_mask, 1, dim=1).sum() / batch_size
            # mask_losses = - torch.var(mask, dim=1, unbiased=False)
            # mask_loss = mask_losses.sum() / batch_size

            loss = ce_loss + lambda1 * mask_loss

            loss.backward()

            # torchviz.make_dot(loss, params=dict(mask_model.named_parameters())).render("mask_torchviz", format="png")
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            running_loss += loss.item()
            running_loss_ce += ce_loss.item()
            running_loss_mask += mask_loss.item()
        print(f"epoch {epoch}", flush=True)
        print(f"total loss : {running_loss}", flush=True)
        print(f"Primary loss : {running_loss_ce}", flush=True)
        print(f"Regularization loss : {running_loss_mask}", flush=True)
        torch.save(mask_model, f'{directory}imdb_masker-{epoch}{suffix}.pt')
        epoch_validation(epoch, mask_model, classifier, classify_tokenizer,  val, word_interner, de_interner, evidence_classes, interned_documents, documents, annotations)

    print('Finished Training')
    return mask_model

def epoch_validation(epoch, mask_model, classifier, tokenizer,  val, word_interner, de_interner, evidence_classes, interned_documents, documents, annotations):
    global best_validation_score
    global best_validation_epoch
    test_batch_size = 1
    results = []
    mask_model.eval()
    explanations = Generator(classifier, mask_model, tokenizer)

    j = 0
    for batch_start in range(0, len(val), test_batch_size):
        batch_elements = val[0:min(0 + test_batch_size, len(val))]
        val = val[test_batch_size:]
        targets = [evidence_classes[s.classification] for s in batch_elements]
        targets = torch.tensor(targets, dtype=torch.long, device=device)
        samples_encoding = [interned_documents[distilbert_pipeline.extract_docid_from_dataset_element(s)] for s in batch_elements]
        input_ids = torch.stack(
            [samples_encoding[i]['input_ids'] for i in range(len(samples_encoding))]).squeeze(1).to(device)
        attention_masks = torch.stack(
            [samples_encoding[i]['attention_mask'] for i in range(len(samples_encoding))]).squeeze(1).to(
            device)
        preds = classifier(input_ids=input_ids, attention_mask=attention_masks)[0]
        for s in batch_elements:
            doc_name = distilbert_pipeline.extract_docid_from_dataset_element(s)
            inp = documents[doc_name].split()
            classification = "neg" if targets.item() == 0 else "pos"
            is_classification_correct = 1 if preds.argmax(dim=1) == targets else 0
            text = tokenizer.convert_ids_to_tokens(input_ids[0])
            classification = "neg" if targets.item() == 0 else "pos"
            is_classification_correct = 1 if preds.argmax(dim=1) == targets else 0
            target_idx = targets.item()
            print("FORDOR")
            print(batch_start)
            cam_target = explanations.generate_distilbert_explanation(input_ids=input_ids, attention_mask=attention_masks, index=target_idx)[
                0]
            if len(cam_target) == 3:
                input_ids = cam_target[1]
                attention_masks = cam_target[2]
                cam_target = cam_target[0]
                tokenizer = distilbert_tokenizer
            cam_target = cam_target.clamp(min=0)
            cam = cam_target
            cam = my_scores_per_word_from_scores_per_token(tokenizer, cam.unsqueeze(0), input_ids)[0][0]
            j = j + 1
            doc_name = distilbert_pipeline.extract_docid_from_dataset_element(s)
            for i in [80]:
                hard_rationales = []
                print("calculating top ", i)
                _, indices = cam.topk(k=i)
                for index in indices.tolist():
                    hard_rationales.append({
                        "start_token": index,
                        "end_token": index + 1
                    })
                result_dict = {
                    "annotation_id": doc_name,
                    "rationales": [{
                        "docid": doc_name,
                        "hard_rationale_predictions": hard_rationales
                    }],
                }
                results.append(result_dict)
                # result_files[res].write(json.dumps(result_dict) + "\n")
    truth = list(chain.from_iterable(Rationale.from_annotation(ann) for ann in annotations))
    pred = list(chain.from_iterable(Rationale.from_instance(inst) for inst in results))
    token_level_truth = list(chain.from_iterable(rat.to_token_level() for rat in truth))
    token_level_pred = list(chain.from_iterable(rat.to_token_level() for rat in pred))
    token_level_prf = score_hard_rationale_predictions(token_level_truth, token_level_pred)
    print(token_level_prf)
    value = token_level_prf['instance_macro']['f1']
    if value > best_validation_score:
        best_validation_score = value
        best_validation_epoch = epoch
    print(f"epoch {epoch} validation score: {value}")
    print(f"BEST epoch {best_validation_epoch} BEST score: {best_validation_score}")
    mask_model.train()

def eval_eye(mask_model, classifier, eval_dataset, index):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    eval_dataset = eval_dataset.remove_columns(["text"])
    # eval_dataset = eval_dataset.remove_columns(["label"])
    eval_dataset.set_format("torch")
    eval_dataloader = DataLoader(eval_dataset, batch_size=4)
    with open(f'vizs-{index}.html', 'w') as outfile:
        k = 0
        for temp_batch in eval_dataloader:
            labels = temp_batch.pop('label', None)
            batch = {k: v.to(device) for k, v in temp_batch.items()}
            masker_output = torch.sigmoid(mask_model(**batch).logits)
            classifier_output = classifier(**batch)

            for j in range(len(masker_output)):
                tokens = tokenizer.batch_decode(batch['input_ids'][j].unsqueeze(1))
                x = masker_output[j]
                expl = [x[i].item() for i in range(len(x))]
                # expl = [0 if x < max(expl) / 10 else x / max(expl) for x in expl]
                # print(classifier_output.logits.shape)
                classification = classifier_output.logits[j].argmax(dim=-1).item()
                print(j)
                true_class = labels[j]
                # print(list(zip(orig, x2)))
                vis_data_records = [visualization.VisualizationDataRecord(
                    expl,
                    classifier_output.logits[j][classification],
                    classification,
                    true_class,
                    true_class,
                    1,
                    tokens,
                    1)]
                html_obj = visualization.visualize_text(vis_data_records)
                # print(html_obj)
                # print(html_obj.data)
                outfile.write(html_obj.data + "\n")
                # print(tokenizer.batch_decode(batch['input_ids']))
            # tokens = tokenizer.convert_ids_to_tokens(input_ids.flatten())
            # print([(tokens[i], expl[i]) for i in range(len(tokens))])
            k = k + 1
            if k > 10:
                break

# def eval_batch(mask_model, input_ids, attention_mask, index=None):
#     if index == None:
#         #this is th lable
#         index = np.argmax(output.cpu().data.numpy(), axis=-1)
#     tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
#     for temp_batch in eval_dataloader:
#         labels = temp_batch.pop('label', None)
#         batch = {k: v.to(device) for k, v in temp_batch.items()}
#         masker_output = torch.sigmoid(mask_model(input_ids=input_ids, attention_mask=attention_mask).logits)
#         classifier_output = classifier(**batch)
#
#         for j in range(len(masker_output)):
#             tokens = tokenizer.batch_decode(batch['input_ids'][j].unsqueeze(1))
#             x = masker_output[j]
#             expl = [x[i].item() for i in range(len(x))]
#             # expl = [0 if x < max(expl) / 10 else x / max(expl) for x in expl]
#             # print(classifier_output.logits.shape)
#             classification = classifier_output.logits[j].argmax(dim=-1).item()
#             print(j)
#             true_class = labels[j]
#             # print(list(zip(orig, x2)))
#             vis_data_records = [visualization.VisualizationDataRecord(
#                 expl,
#                 classifier_output.logits[j][classification],
#                 classification,
#                 true_class,
#                 true_class,
#                 1,
#                 tokens,
#                 1)]
#             html_obj = visualization.visualize_text(vis_data_records)
#             # print(html_obj)
#             # print(html_obj.data)
#             outfile.write(html_obj.data + "\n")
#             # print(tokenizer.batch_decode(batch['input_ids']))
#         # tokens = tokenizer.convert_ids_to_tokens(input_ids.flatten())
#         # print([(tokens[i], expl[i]) for i in range(len(tokens))])
#         k = k + 1
#         if k > 10:
#             break

# FORDOR remember you copy pasted the validation code!!
def my_scores_per_word_from_scores_per_token(tokenizer, scores, input_ids):
    res = []
    sentences = tokenizer.batch_decode(input_ids)
    for i in range(len(input_ids)):
        curr_res = []
        res.append(curr_res)
        curr_scores = scores[i]
        curr_ids = input_ids[i]
        curr_sentence = sentences[i].split(" ")
        tokens = tokenizer.batch_decode(curr_ids)
        # alnum_trouble = False
        curr_word_len = 0
        j = -1
        first_token= True
        for t in range(len(tokens)):
            token = tokens[t]
            token_len = len(token)
            part_token = token.startswith("##")
            if not first_token and (token.startswith("##") or curr_word_len < len(curr_sentence[j])):
                # alnum_trouble = not (tokens[t].startswith("##") or tokens[t].isalnum())
                curr_res[j] = max(curr_res[j], curr_scores[t].item())
                curr_word_len += token_len -2 if part_token else token_len
            else:
                first_token = False
                j += 1
                curr_word_len = token_len
                curr_res.append(curr_scores[t].item())
    return torch.tensor(res), sentences


def scores_per_token_from_scores_per_word(tokenizer, scores, sentences):

    tokens = tokenizer(sentences)
    res = torch.zeros(tokens)
    for i in range(len(tokens)):
        curr_res = res[i]
        # res.append(curr_res)
        curr_scores = scores[i]
        # curr_ids = input_ids[i]
        curr_sentence = sentences[i].split(" ")
        curr_tokens = tokenizer.batch_decode(tokens[i])
        # alnum_trouble = False
        curr_word_len = 0
        j = -1
        first_token= True
        for t in range(len(curr_tokens)):
            token = curr_tokens[t]
            token_len = len(token)
            part_token = token.startswith("##")
            if not first_token and (token.startswith("##") or curr_word_len < len(curr_sentence[j])):
                # alnum_trouble = not (tokens[t].startswith("##") or tokens[t].isalnum())
                curr_res[t] = (curr_scores[j])
                curr_word_len += token_len -2 if part_token else token_len
            else:
                first_token = False
                j += 1
                curr_word_len = token_len
                curr_res[t] = (curr_scores[j])
    return res


def load_masker(epoch):
    mask_model = torch.load(f'{directory}imdb_masker-{epoch}.pt', map_location=device)
    mask_model.eval()
    return mask_model


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

    train, val, test = load_datasets(data_dir)
    docids = set(e.docid for e in
                 chain.from_iterable(chain.from_iterable(map(lambda ann: ann.evidences, chain(train, val, test)))))
    documents = load_documents(data_dir, docids)
    # imdb_data = load_dataset("imdb")
    # train_dataset = imdb_data['train']
    # print(f"IMDB dataset - train:{len(imdb_data['train'])}, test:{imdb_data['test']}")
    train_dataset = convert_dataset(train, documents, "train")
    val_dataset = convert_dataset(val, documents, "validation")
    test_dataset = convert_dataset(test, documents, "test")

    train_dataset = tokenize_dataset(train_dataset)
    # val_dataset = tokenize_dataset(val_dataset)
    # test_dataset = tokenize_dataset(test_dataset)

    evidence_classifier, word_interner, de_interner, evidence_classes, tokenizer = load_classifier(args.model_params)
    cache = os.path.join(args.output_dir, 'preprocessed.pkl')
    if os.path.exists(cache):
        print(f'Loading interned documents from {cache}')
        (interned_documents) = torch.load(cache)
    annotations = annotations_from_jsonl(os.path.join(args.data_dir, 'val' + '.jsonl'))
    masker = train_masker(evidence_classifier, tokenizer, train_dataset, val, word_interner, de_interner, evidence_classes, interned_documents, documents, annotations)
    # eval_eye(masker, evidence_classifier, tokenizer, val_dataset, "20_gt")


def recursive_set_dropout(model, p=0):
    child_list = [c for c in model.children()]
    if len(child_list) == 0:
        if "pout" in str(type(model)):
            model.p = p
    else:
        for child in child_list:
            recursive_set_dropout(child, p)


if __name__ == '__main__':
    sys.exit(main())





# class EraserMoviesDataset(torch.utils.data.Dataset):
#     def __init__(self, texts, labels):
#         self.texts = texts
#         self.labels = labels
#
#     def __getitem__(self, idx):
#         item = {key: torch.tensor(val[idx]) for key, val in self.texts.items()}
#         item['labels'] = torch.tensor(self.labels[idx])
#         return item
#
#     def __len__(self):
#         return len(self.labels)
#
#     def shuffle(self, seed):
#         c = list(zip(self.texts, self.labels))
#
#         random.shuffle(c)
#
#         a, b = zip(*c)