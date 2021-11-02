# IMPORTS

from typing import Any, Dict, List, cast
import torch
import json
from random import sample
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.nn import CrossEntropyLoss
# from sklearn.model_selection import train_test_split
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from copy import deepcopy
import time
from src.utils import format_time, flat_accuracy, confidence_accuracy

import argparse


my_parser = argparse.ArgumentParser(description='Train Model')

my_parser.add_argument('--data_dir',
                       type=str,
                       help='path to training data')

# my_parser.add_argument('--val_ratio',
#                        type=float,
#                        default=0.1,
#                        help='ratio of validation dataset')

my_parser.add_argument('--model_arch',
                       type=str,
                       default='roberta-large',
                       help='model used')

my_parser.add_argument('--max_length',
                       type=int,
                       default=512,
                       help='max length for tokenization')

my_parser.add_argument('--batch_size',
                       type=int,
                       default=16,
                       help='batch size')

my_parser.add_argument('--learning_rate',
                       type=float,
                       default=1e-6,
                       help='Learning Rate')

my_parser.add_argument('--epsilon',
                       type=float,
                       default=1e-6,
                       help='Epsilon Value')

my_parser.add_argument('--weight_decay',
                       type=float,
                       default=0.1,
                       help='Weight Decay')

my_parser.add_argument('--epochs',
                       type=int,
                       default=3,
                       help='Number of Epochs')

my_parser.add_argument('--warmup_ratio',
                       type=float,
                       default=0.06,
                       help='Warmup Ratio')

my_parser.add_argument('--verbose',
                       action='store_true',
                       default=False,
                       help='Verbose Output')

my_parser.add_argument('--time_step_size',
                       type=int,
                       default=100,
                       help='Step Size for time')

args = my_parser.parse_args()

data_dir = args.data_dir  # 'train_data/'
# val_ratio = args.val_ratio  # 0.1
model_arch = args.model_arch  # 'roberta-large'
max_length = args.max_length  # 512
batch_size = args.batch_size  # 16
lr = args.learning_rate  # 1e-6
eps = args.epsilon  # 1e-6
weight_decay = args.weight_decay  # 0.1
epochs = args.epochs  # 3
warmup_ratio = args.warmup_ratio  # 0.06
verbose = args.verbose  # True
time_step_size = args.time_step_size  # 100


# true_file = data_dir + 'true.json'
# false_file = data_dir + 'false.json'

# # LOAD DATA
# true_theories = json.load(open(true_file, 'r'))
# false_theories = json.load(open(false_file, 'r'))

# # Split train and val
# train_true_theories, val_true_theories = train_test_split(true_theories, test_size=val_ratio / 2)
# train_false_theories, val_false_theories = train_test_split(false_theories, test_size=val_ratio / 2)

# train_theories_1 = train_true_theories + train_false_theories
# val_theories = val_true_theories + val_false_theories

train_file = data_dir + 'train.jsonl'
val_file = data_dir + 'val.jsonl'

train_theories_1 = [json.loads(jline) for jline in open(train_file, "r").read().splitlines()]
val_theories = [json.loads(jline) for jline in open(val_file, "r").read().splitlines()]

# UPDATE DATA FOR wBCE
for x in tqdm(train_theories_1):
    if(not x['output']):
        x['hyp_weight'] = 1 - x['hyp_weight']
train_theories_1 = sample(train_theories_1, len(train_theories_1))
train_theories_2 = deepcopy(train_theories_1)
for x in tqdm(train_theories_2):
    x['output'] = False if x['output'] else True
    x['hyp_weight'] = 1 - x['hyp_weight']

train_theories = cast(List[Dict[Any, Any]], [item for sublist
                      in list(map(list, zip(train_theories_1, train_theories_2))) for item in sublist])

# prepare training data
train_context = [t['context'] for t in train_theories]
train_hypotheses = [t['hypothesis_sentence'] for t in train_theories]
train_labels_ = [1 if t['output'] else 0 for t in train_theories]
train_data_weights_ = [t['hyp_weight'] for t in train_theories]

# prepare val data
val_context = [t['context'] for t in val_theories]
val_hypotheses = [t['hypothesis_sentence'] for t in val_theories]
val_labels_ = [1 if t['output'] else 0 for t in val_theories]
val_data_weights_ = [t['hyp_weight'] for t in val_theories]

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_arch)


# tokenize training data
train_input_ids_ = []
train_attention_masks_ = []

for c, h in tqdm(zip(train_context, train_hypotheses)):
    encoded = tokenizer.encode_plus(c, h,
                                    max_length=max_length,
                                    truncation=True,
                                    return_tensors='pt',
                                    padding='max_length')
    train_input_ids_.append(encoded['input_ids'])
    train_attention_masks_.append(encoded['attention_mask'])

train_input_ids = torch.cat(train_input_ids_, dim=0)
train_attention_masks = torch.cat(train_attention_masks_, dim=0)

train_labels = torch.tensor(train_labels_)
train_data_weights = torch.tensor(train_data_weights_)

train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels, train_data_weights)


# tokenize val data
val_input_ids_ = []
val_attention_masks_ = []

for c, h in tqdm(zip(val_context, val_hypotheses)):
    encoded = tokenizer.encode_plus(c, h,
                                    max_length=max_length,
                                    truncation=True,
                                    return_tensors='pt',
                                    padding='max_length')
    val_input_ids_.append(encoded['input_ids'])
    val_attention_masks_.append(encoded['attention_mask'])

val_input_ids = torch.cat(val_input_ids_, dim=0)
val_attention_masks = torch.cat(val_attention_masks_, dim=0)

val_labels = torch.tensor(val_labels_)
val_data_weights = torch.tensor(val_data_weights_)

val_dataset = TensorDataset(val_input_ids, val_attention_masks, val_labels, val_data_weights)

train_dataloader = DataLoader(dataset=train_dataset,
                              sampler=SequentialSampler(train_dataset),
                              batch_size=batch_size,
                              )

val_dataloader = DataLoader(dataset=val_dataset,
                            sampler=RandomSampler(val_dataset),
                            batch_size=batch_size,
                            )


# Load model
device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

model = AutoModelForSequenceClassification.from_pretrained(model_arch, num_labels=2)
model = model.to(device)

optimizer = AdamW(model.parameters(),
                  lr=lr,
                  eps=eps,
                  weight_decay=weight_decay)

total_steps = len(train_dataloader) * epochs

scheduler = get_linear_schedule_with_warmup(optimizer,  # Default value in run_glue.py
                                            num_warmup_steps=int(warmup_ratio * total_steps),
                                            num_training_steps=int((1 - warmup_ratio) * total_steps))

loss_fct = CrossEntropyLoss(reduction='none')

training_stats = []

total_t0 = time.time()

for epoch_i in range(epochs):
    # ========================================
    #               Training
    # ========================================
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    t0 = time.time()
    total_train_loss = 0.0

    model.train()

    for step, batch in enumerate(train_dataloader):
        if step % time_step_size == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            if verbose:
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        b_weights = batch[3].to(device)

        model.zero_grad()

        o = model(b_input_ids,
                  attention_mask=b_input_mask)

        logits = o[0]
        loss = torch.mean(loss_fct(logits.view(-1, 2), b_labels.view(-1)) * b_weights)

        total_train_loss += loss.item()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        scheduler.step()

    avg_train_loss = total_train_loss / len(train_dataloader)

    training_time = format_time(time.time() - t0)
    if verbose:
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))

    # ========================================
    #               Validation
    # ========================================

    print("")
    print("Running Validation...")

    t0 = time.time()

    model.eval()

    total_eval_accuracy = 0
    total_eval_loss = 0.0
    total_conf_acc = 0
    nb_eval_steps = 0

    for batch in val_dataloader:

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        b_weights = batch[3].to(device)

        with torch.no_grad():
            o = model(b_input_ids, attention_mask=b_input_mask)

        logits = o[0]
        loss = torch.mean(loss_fct(logits.view(-1, 2), b_labels.view(-1)) * b_weights)

        total_eval_loss += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        total_eval_accuracy += flat_accuracy(logits, label_ids)
        total_conf_acc += confidence_accuracy(logits, b_labels, b_weights)
    avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
    avg_val_conf_acc = total_conf_acc / len(val_dataloader)

    print("  Accuracy: {}".format(avg_val_accuracy))

    avg_val_loss = total_eval_loss / len(val_dataloader)

    validation_time = format_time(time.time() - t0)

    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time,
            'Val_Conf_Acc': avg_val_conf_acc
        }
    )

total_train_time = format_time(time.time() - total_t0)
training_stats.append({'total_train_time': total_train_time})
print("")
print("Training complete!")

print("Total training took {:} (h:mm:ss)".format(total_train_time))

model_arch = 'roberta-large'

training_stats.append({'hyperparameters': {'max_length': max_length,
                                           'batch_size': batch_size,
                                           'learning_rate': lr,
                                           'epsilon': eps,
                                           'weight_decay': weight_decay,
                                           'n_epochs': epochs,
                                           'warmup_ratio': warmup_ratio}})
training_stats.append({'model': model_arch,
                       'dataset': data_dir})

# output model and dict of results
model_path = f'models/{time.strftime("%Y%m%dT%H%M%S")}/'
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
json.dump(training_stats, open(f"{model_path}train_stats.json", "w"), indent=4)
