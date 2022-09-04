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

from BERT_rationale_benchmark.utils import load_documents, load_datasets

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

directory = "/home/joberant/NLP_2122/dorcoh4/weight_map/"
data_dir = "/home/joberant/NLP_2122/dorcoh4/weight_map/movies"
# directory = "C:/Users/Dor_local/Downloads/"
# data_dir = "C:/Users/Dor_local/Downloads/movies.tar/movies"


def convert_dataset(raw_dataset, documents, name):
    texts = []
    labels = []
    for line in raw_dataset:
        if line.annotation_id in documents:  # FORDOR this if
            sentence_list = [" ".join(sent) for sent in documents[line.annotation_id]]
            texts.append("\n".join(sentence_list))
            labels.append(0 if line.classification.upper() == 'NEG' else 1)

    # file_name = f"eraser_movies_{name}.parquet"
    # table = pa.table({'text': texts,
    #                   'label': labels,})
    # pq.write_table(table, file_name)
    # dataset = ds.dataset(file_name)
    dataset = Dataset.from_dict({'text': texts, 'label': labels})
    return dataset


def tokenize_dataset(raw_dataset):
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


def load_classifier():
    # model = torch.load(directory + 'imdb_classifier.pt', map_location=device)
    model.eval()
    return model


def train_masker(classifier, train_dataset):
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
            masked_in = classifier.distilbert.embeddings(batch['input_ids']) * mask

            out1 = classifier(**batch)
            batch.pop('input_ids', None)
            batch['inputs_embeds'] = masked_in
            out2 = classifier(**batch)
            original_probs = softmax(out1.logits)
            masked_probs = softmax(out2.logits)

            ce_loss = crossEntropyLoss(out2.logits, masked_probs)
            mask_loss = torch.norm(mask, 1, dim=1).sum() / batch_size
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
        torch.save(mask_model, f'{directory}imdb_masker-{epoch}_001_lay.pt')

    print('Finished Training')
    return mask_model


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

def eval_batch(mask_model, input_ids, attention_mask, index=None):
    if index == None:
        #this is th lable
        index = np.argmax(output.cpu().data.numpy(), axis=-1)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    for temp_batch in eval_dataloader:
        labels = temp_batch.pop('label', None)
        batch = {k: v.to(device) for k, v in temp_batch.items()}
        masker_output = torch.sigmoid(mask_model(input_ids=input_ids, attention_mask=attention_mask).logits)
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


def load_masker(epoch):
    mask_model = torch.load(f'{directory}imdb_masker-{epoch}_001_gt_lay.pt', map_location=device)
    mask_model.eval()
    return mask_model


def main():
    train, val, test = load_datasets(data_dir)
    docids = set(e.docid for e in
                 chain.from_iterable(chain.from_iterable(map(lambda ann: ann.evidences, chain(train, val, test)))))
    documents = load_documents(data_dir, docids)
    train_dataset = convert_dataset(train, documents, "train")
    val_dataset = convert_dataset(val, documents, "validation")
    test_dataset = convert_dataset(test, documents, "test")

    train_dataset = tokenize_dataset(train_dataset)
    val_dataset = tokenize_dataset(val_dataset)
    test_dataset = tokenize_dataset(test_dataset)
    classifier = load_classifier()
    masker = train_masker(classifier, train_dataset)
    eval_eye(masker, classifier, val_dataset, "20_gt")


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