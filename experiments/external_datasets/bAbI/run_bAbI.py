import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm
import json
import argparse

# TODO F1 score

my_parser = argparse.ArgumentParser(description='Train Model')

my_parser.add_argument('--babi_dir',
                       default='data/external_datasets/bAbI/',
                       type=str,
                       help='path to bAbI data')

my_parser.add_argument('--epochs',
                       default=3,
                       type=int,
                       help='number of epochs')

my_parser.add_argument('--lr',
                       default=1.5e-5,
                       type=float,
                       help='learning rate')

my_parser.add_argument('--batch_size',
                       default=16,
                       type=int,
                       help='batch size')
my_parser.add_argument('--warmup_ratio',
                       default=0.06,
                       type=float,
                       help='warmup ratio')

args = my_parser.parse_args()

babi_dir = args.babi_dirs
epochs = args.epochs
lr = args.lr
batch_size = args.batch_size
warmup_ratio = args.warmip_ratio


train_file = babi_dir + 'qa15_train.txt'
val_file = babi_dir + 'qa15_valid.txt'
test_file = babi_dir + 'qa15_test.txt'

def get_start_end_indices(context, answer, proof):
    c = " ".join(context)
    rule_index = int(proof.split(" ")[-1])
    start_index = c.index(context[rule_index - 1]) + context[rule_index - 1].index(answer)
    end_index = start_index + len(answer)
    if c[start_index:end_index] != answer:
        AssertionError('Should be equal')
    return start_index, end_index

def parse_file(file):
    with open(file, 'r') as f:
        lines = f.readlines()

    all_questions = []
    all_contexts = []
    all_answers = []

    questions = []
    context = []
    answers = []

    for idx, line in enumerate(lines[:]):

        ind = int(line.split(" ")[0])
        if ind == 1 and idx > 2:
            if questions:
                all_questions.append(questions)
                all_contexts.append(context)
                all_answers.append(answers)

            questions = []
            context = []
            answers = []

        if '?' not in line:
            sent = " ".join(line.strip().split(" ")[1:])
            context.append(sent)

        else:
            question, answer, proof = " ".join(line.strip().split(" ")[1:]).split('\t')
            questions.append(question)
            answer = context[int(proof.split(" ")[-1]) - 1].split(" ")[-1][:-1]
            s, e = get_start_end_indices(context, answer, proof)
            d = {'text': answer, 'answer_start': s, 'answer_end': e}
            answers.append(d)
    return all_questions, all_contexts, all_answers

def prepare_data(all_contexts, all_questions, all_answers):
    qs = []
    az = []
    cntxs = []

    for i in range(len(all_contexts)):
        for j in range(len(all_questions[i])):
            qs.append(all_questions[i][j])
            az.append(all_answers[i][j])
            cntxs.append(" ".join(all_contexts[i]))
    return qs, cntxs, az

def add_token_positions(encodings, answers):
    # initialize lists to contain the token indices of answer start/end
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        # append start/end token position using char_to_token method
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
        end_positions.append(encodings.char_to_token(i, answers[i]['answer_end']))

        # if start position is None, the answer passage has been truncated
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        # end position cannot be found, char_to_token found space, so shift position until found
        shift = 1
        while end_positions[-1] is None:
            end_positions[-1] = encodings.char_to_token(i, answers[i]['answer_end'] - shift)
            shift += 1
    # update our encodings object with the new token-based start/end positions
    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})


class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


train_questions, train_context, train_answers = parse_file(train_file)
val_questions, val_context, val_answers = parse_file(val_file)
test_questions, test_context, test_answers = parse_file(test_file)

# load RuleBERT model
rulebert_model = 'models/ruleBERT/rulebert500/'
rulebert_trained = AutoModelForSequenceClassification.from_pretrained(rulebert_model)

train_questions, train_context, train_answers = prepare_data(train_context, train_questions, train_answers)
val_questions, val_context, val_answers = prepare_data(val_context, val_questions, val_answers)
test_questions, test_context, test_answers = prepare_data(test_context, test_questions, test_answers)

final_d = []
for e in range(1, epochs + 1):
    d = {}
    for q in range(10):

        print(f'Epoch {q+1}/10')

        model_type = 'roberta-large'
        model = AutoModelForQuestionAnswering.from_pretrained(model_type)
        tokenizer = AutoTokenizer.from_pretrained(model_type)
        rulebert_trained = AutoModelForSequenceClassification.from_pretrained(rulebert_model)
        # move rulebert encoder to roberta
        model.roberta = rulebert_trained.roberta

        # tokenize
        train_encodings = tokenizer(train_context, train_questions, truncation=True, padding=True)
        val_encodings = tokenizer(val_context, val_questions, truncation=True, padding=True)
        test_encodings = tokenizer(test_context, test_questions, truncation=True, padding=True)

        add_token_positions(train_encodings, train_answers)
        add_token_positions(val_encodings, val_answers)
        add_token_positions(test_encodings, test_answers)

        # build datasets for both our training and validation sets
        train_dataset = SquadDataset(train_encodings)
        val_dataset = SquadDataset(val_encodings)
        test_dataset = SquadDataset(test_encodings)

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device)
        model.train()
        optim = AdamW(model.parameters(), lr=lr)
        # initialize data loader for training data
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        total_steps = len(train_loader) * epochs

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(optim,
                                                    num_warmup_steps=int(total_steps * warmup_ratio),
                                                    num_training_steps=total_steps)

        for epoch in range(e):
            model.train()
            loop = tqdm(train_loader, leave=True)

            for batch in loop:
                optim.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                start_positions = batch['start_positions'].to(device)
                end_positions = batch['end_positions'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask,
                                start_positions=start_positions,
                                end_positions=end_positions)
                loss = outputs[0]
                loss.backward()
                optim.step()
                scheduler.step()
                loop.set_description(f'Epoch {epoch}')
                loop.set_postfix(loss=loss.item())

            # Run Validation
            total_eval_loss = 0
            total_eval_accuracy = 0
            model.eval()
            pred_answers = []
            true_answers = []
            acc = []
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                start_positions = batch['start_positions'].to(device)
                end_positions = batch['end_positions'].to(device)

                with torch.no_grad():
                    outputs = model(input_ids, attention_mask=attention_mask,
                                    start_positions=start_positions,
                                    end_positions=end_positions)

                loss = outputs[0]
                total_eval_loss += loss.item()

                start_true = batch['start_positions'].to(device)
                end_true = batch['end_positions'].to(device)

                start_pred = torch.argmax(outputs['start_logits'], dim=1)
                end_pred = torch.argmax(outputs['end_logits'], dim=1)

                acc.append(((start_pred == start_true).sum() / len(start_pred)).item())
                acc.append(((end_pred == end_true).sum() / len(end_pred)).item())
                for i in range(len(input_ids)):
                    pred_answers.append(tokenizer.convert_ids_to_tokens(input_ids[i]
                                        [int(start_pred[i]):int(end_pred[i])]))
                    true_answers.append(tokenizer.convert_ids_to_tokens(input_ids[i][start_true[i]:end_true[i]]))
            acc_ = sum(acc) / len(acc)
            print(f'\n Accuracy Val Set:{acc_}\n')

        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Run Test
        total_test_loss = 0
        model.eval()

        pred_answers = []
        true_answers = []

        acc = []
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)

            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask,
                                start_positions=start_positions,
                                end_positions=end_positions)

            loss = outputs[0]
            total_test_loss += loss.item()

            start_true = batch['start_positions'].to(device)
            end_true = batch['end_positions'].to(device)

            start_pred = torch.argmax(outputs['start_logits'], dim=1)
            end_pred = torch.argmax(outputs['end_logits'], dim=1)

            acc.append(((start_pred == start_true).sum() / len(start_pred)).item())
            acc.append(((end_pred == end_true).sum() / len(end_pred)).item())

            for i in range(len(input_ids)):
                pred_answers.append(tokenizer.convert_ids_to_tokens(input_ids[i]
                                    [int(start_pred[i]):int(end_pred[i])]))
                true_answers.append(tokenizer.convert_ids_to_tokens(input_ids[i][start_true[i]:end_true[i]]))

        acc_ = sum(acc) / len(acc)
        d[q] = acc_
    final_d.append(d)
# TODO Average values
json.dump(final_d, open(f"{babi_dir}test_stats.json", "w"))
