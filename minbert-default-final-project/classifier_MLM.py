import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace
import csv

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score, recall_score, accuracy_score

# change it with respect to the original model
from tokenizer import BertTokenizer, SpecialTokensMixin, PreTrainedTokenizer

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

import torch.nn as nn


TQDM_DISABLE=False
# fix the random seed
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class BertSentimentClassifier(torch.nn.Module):
    '''
    This module performs sentiment classification using BERT embeddings on the SST dataset.

    In the SST dataset, there are 5 sentiment categories (from 0 - "negative" to 4 - "positive").
    Thus, your forward() should return one logit for each of the 5 classes.
    '''
    def __init__(self, config):
        super(BertSentimentClassifier, self).__init__()
        self.num_labels = config.num_labels
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Pretrain mode does not require updating bert paramters.
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True

        ### TODO
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear = nn.Linear(config.hidden_size, self.num_labels)



    def forward(self, input_ids, attention_mask):
        '''Takes a batch of sentences and returns logits for sentiment classes'''
        # The final BERT contextualized embedding is the hidden state of [CLS] token (the first token).
        # HINT: you should consider what is the appropriate output to return given that
        # the training loop currently uses F.cross_entropy as the loss function.
        ### TODO
        encoded_sentences = self.bert.forward(input_ids, attention_mask)['pooler_output']
        encoded_sentences_dropout = self.dropout(encoded_sentences)
        encoded_sentences_linear = self.linear(encoded_sentences_dropout)
        
        return encoded_sentences_linear


class MLM(torch.nn.Module):
    '''
    This module performs MLM.

    In the SST dataset, there are 5 sentiment categories (from 0 - "negative" to 4 - "positive").
    Thus, your forward() should return one logit for each of the 5 classes.
    '''

    def __init__(self, config):
        super(MLM, self).__init__()
        self.num_labels = config.num_labels
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Pretrain mode does not require updating bert paramters.
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True

        ### TODO
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear = nn.Linear(config.hidden_size, self.bert.config.vocab_size)

    def forward(self, input_ids, attention_mask):
        encoded_sentences = self.bert.forward(input_ids, attention_mask)['pooler_output']
        # encoded_sentences_dropout = self.dropout(encoded_sentences)
        encoded_sentences_linear = self.linear(encoded_sentences)

        return encoded_sentences_linear


class SentimentDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        
        sents = [x[0] for x in data]
        labels = [x[1] for x in data]
        sent_ids = [x[2] for x in data]

        encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])
        labels = torch.LongTensor(labels)

        return token_ids, attention_mask, labels, sents, sent_ids

    def collate_fn(self, all_data):
        token_ids, attention_mask, labels, sents, sent_ids= self.pad_data(all_data)

        batched_data = {
                'token_ids': token_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'sents': sents,
                'sent_ids': sent_ids
            }

        return batched_data


class MLMDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def clean_up_tokenization(self, out_string):
      """
      Clean up a list of simple English tokenization artifacts like spaces before punctuations and abbreviated forms.
      Args:
        out_string (:obj:`str`): The text to clean up.
      Returns:
        :obj:`str`: The cleaned-up string.
      """
      out_string_2 = (
        out_string.replace(" .", ".")
          .replace(" ?", "?")
          .replace(" !", "!")
          .replace(" ,", ",")
          .replace(" ' ", "'")
          .replace(" n't", "n't")
          .replace(" 'm", "'m")
          .replace(" 's", "'s")
          .replace(" 've", "'ve")
          .replace(" 're", "'re")
      )
      return out_string_2

    def mask(self, sents):
        sents_masked = []
        indices_all = []
        for sent in sents:
            original = sent
            print('sent before: ', sent)
            sent = self.tokenizer._tokenize(sent)
            print("sent after tokenization: ", sent)
            #print("sent size: ", len(sent))
            mask = np.random.binomial(1, 0.15, (len(sent),))
            #print('mask: ', mask)
            #print('np.where command: ', np.where(mask)[0])
            for word_id in np.where(mask)[0]:
                if(word_id != (len(mask) - 1)):
                    if (sent[word_id + 1])[:2] != "##":
                        sent[word_id] = "[MASK]"
                else:
                    sent[word_id] = "[MASK]"

            indices_all.append(np.where(mask)[0])
            print("tokenized sent after mask: ", sent)
            # ids = [self.tokenizer._convert_token_to_id(token) for token in sent]
            # print(ids)
            sent_str = self.tokenizer.convert_tokens_to_string(sent)
            print("raw masked sentence: ", sent_str)
            #sent_str = self.clean_up_tokenization(sent_str)
            #print("cleaned up masked sentence: ", sent_str)
            #assert(original == sent_str)
            print("re-tokenized masked sentence:", self.tokenizer._tokenize(sent_str))
            sents_masked.append(sent_str)
        #print("sents_masked in mask fxn: ", sents_masked)
        return sents_masked, indices_all

    def pad_data(self, data):
        sents = [x[0] for x in data]
        labels = [x[1] for x in data]
        sent_ids = [x[2] for x in data]
        sents_masked, mask_indices = self.mask(sents)
        #print('sents: ', sents)
        #print('sents_masked: ', sents_masked)
        encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
        print('encoding: ', encoding['input_ids'])
        encoding_masked = self.tokenizer(sents_masked, return_tensors='pt', padding=True, truncation=True)
        print('encoding_masked: ', encoding_masked['input_ids'])
        token_ids = torch.LongTensor(encoding['input_ids'])
        token_ids_masked = torch.LongTensor(encoding_masked['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])
        attention_mask_masked = torch.LongTensor(encoding_masked['attention_mask'])
        labels = torch.LongTensor(labels)

        return token_ids, attention_mask, labels, sents, sent_ids, token_ids_masked, attention_mask_masked, mask_indices

    def collate_fn(self, all_data):
        token_ids, attention_mask, labels, sents, sent_ids, token_ids_masked, attention_mask_masked, mask_indices = self.pad_data(all_data)

        batched_data = {
            'token_ids': token_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'sents': sents,
            'sent_ids': sent_ids,
            'token_ids_masked': token_ids_masked,
            'attention_mask_masked': attention_mask_masked,
            'mask_indices': mask_indices
        }

        return batched_data


class SentimentTestDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        
        sents = [x[0] for x in data]
        sent_ids = [x[1] for x in data]

        encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])

        return token_ids, attention_mask, sents, sent_ids

    def collate_fn(self, all_data):
        token_ids, attention_mask, sents, sent_ids= self.pad_data(all_data)

        batched_data = {
                'token_ids': token_ids,
                'attention_mask': attention_mask,
                'sents': sents,
                'sent_ids': sent_ids
            }

        return batched_data


# Load the data: a list of (sentence, label)
def load_data(filename, flag='train'):
    num_labels = {}
    data = []
    if flag == 'test':
        with open(filename, 'r') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                sent = record['sentence'].lower().strip()
                sent_id = record['id'].lower().strip()
                data.append((sent,sent_id))
    else:
        with open(filename, 'r') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                sent = record['sentence'].lower().strip()
                sent_id = record['id'].lower().strip()
                label = int(record['sentiment'].strip())
                if label not in num_labels:
                    num_labels[label] = len(num_labels)
                data.append((sent, label,sent_id))
        print(f"load {len(data)} data from {filename}")

    if flag == 'train':
        return data, len(num_labels)
    else:
        return data

# Evaluate the model for accuracy.
def model_eval(dataloader, model, device):
    model.eval() # switch to eval model, will turn off randomness like dropout
    y_true = []
    y_pred = []
    sents = []
    sent_ids = []
    for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
        b_ids, b_mask, b_labels, b_sents, b_sent_ids = batch['token_ids'],batch['attention_mask'],  \
                                                        batch['labels'], batch['sents'], batch['sent_ids']
                                                      

        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)

        logits = model(b_ids, b_mask)
        logits = logits.detach().cpu().numpy()
        preds = np.argmax(logits, axis=1).flatten()

        b_labels = b_labels.flatten()
        y_true.extend(b_labels)
        y_pred.extend(preds)
        sents.extend(b_sents)
        sent_ids.extend(b_sent_ids)

    f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)

    return acc, f1, y_pred, y_true, sents, sent_ids


def model_test_eval(dataloader, model, device):
    model.eval() # switch to eval model, will turn off randomness like dropout
    y_pred = []
    sents = []
    sent_ids = []
    for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
        b_ids, b_mask, b_sents, b_sent_ids = batch['token_ids'],batch['attention_mask'],  \
                                                         batch['sents'], batch['sent_ids']
                                                      

        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)
        logits = model(b_ids, b_mask)
        logits = logits.detach().cpu().numpy()
        preds = np.argmax(logits, axis=1).flatten()

        y_pred.extend(preds)
        sents.extend(b_sents)
        sent_ids.extend(b_sent_ids)

    return y_pred, sents, sent_ids


def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")

def find_first_zero(tensor):
    """Finds the index of the first occurrence of 0 in a tensor."""
    idx = torch.where(tensor == 0)[0]
    if len(idx) > 0:
        return idx[0].item()
    else:
        return tensor.shape[0]


def train(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Load data
    # Create the data and its corresponding datasets and dataloader
    train_data, num_labels = load_data(args.train, 'train')
    dev_data = load_data(args.dev, 'valid')

    train_dataset = MLMDataset(train_data, args)
    dev_dataset = SentimentDataset(dev_data, args)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                                  collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
                                collate_fn=dev_dataset.collate_fn)

    # Init model
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option}

    config = SimpleNamespace(**config)

    model = BertSentimentClassifier(config)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc = 0

    # Run for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for batch in tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            b_ids, b_mask, b_labels, b_ids_masked, b_mask_masked, mask_indices = (batch['token_ids'],
                                       batch['attention_mask'], batch['labels'],
                                       batch['token_ids_masked'], batch['attention_mask_masked'], batch['mask_indices'])

            b_ids_masked = b_ids_masked.to(device)
            b_mask_masked = b_mask_masked.to(device)
            b_labels = b_labels.to(device)

            # b_indices = []
            # b_targets = b_ids #copy original labels to use as targets
            # for sentence in b_ids:
            #     len_sen = find_first_zero(sentence)
            #     num_mask = int(0.15 * len_sen)
            #     indices = np.random.choice(range(0, len_sen), size=num_mask, replace=False)
            #     b_indices.append(indices)
            #     for i in indices:
            #         sentence[i] = -9999 #mask ID
            # print(b_ids)

            # for x in range(0, len(b_targets)): #loop through batch
            #     for y in range(0, len(b_targets[x])): #loop through each sentence
            #         if y not in b_indices[x]: #look at indices which aren't masked
            #             b_targets[x] = float('-inf')  # we don't want to use non-masked targets

            optimizer.zero_grad()
            logits = model(b_ids_masked, b_mask_masked)

            targets = torch.zeros_like(logits)
            for i in range(0, len(targets)): #loops thru sentences
                for j in range(0, len(targets[i])): #loops thru words in sentence
                    if j in mask_indices[i]:
                        targets[i][j][b_ids[i][j]] = 1
                    else:
                        targets[i][j] = torch.full_like(targets[i][j], -float('inf'))


            loss = F.cross_entropy(logits, targets, reduction='sum') / args.batch_size
            #loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / (num_batches)

        train_acc, train_f1, *_  = model_eval(train_dataloader, model, device)
        dev_acc, dev_f1, *_ = model_eval(dev_dataloader, model, device)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)

        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")


def test(args):
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']
        model = BertSentimentClassifier(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"load model from {args.filepath}")
        
        dev_data = load_data(args.dev, 'valid')
        dev_dataset = SentimentDataset(dev_data, args)
        dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=dev_dataset.collate_fn)

        test_data = load_data(args.test, 'test')
        test_dataset = SentimentTestDataset(test_data, args)
        test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=test_dataset.collate_fn)
        
        dev_acc, dev_f1, dev_pred, dev_true, dev_sents, dev_sent_ids = model_eval(dev_dataloader, model, device)
        print('DONE DEV')
        test_pred, test_sents, test_sent_ids = model_test_eval(test_dataloader, model, device)
        print('DONE Test')
        with open(args.dev_out, "w+") as f:
            print(f"dev acc :: {dev_acc :.3f}")
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(dev_sent_ids,dev_pred ):
                f.write(f"{p} , {s} \n")

        with open(args.test_out, "w+") as f:
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s  in zip(test_sent_ids,test_pred ):
                f.write(f"{p} , {s} \n")
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--dev_out", type=str, default="cfimdb-dev-output.txt")
    parser.add_argument("--test_out", type=str, default="cfimdb-test-output.txt")
                                    

    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-5)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    #args.filepath = f'{args.option}-{args.epochs}-{args.lr}.pt'

    print('Training Sentiment Classifier on SST...')
    config = SimpleNamespace(
        filepath='sst-classifier.pt',
        lr=args.lr,
        use_gpu=args.use_gpu,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dropout_prob=args.hidden_dropout_prob,
        train='data/ids-sst-train.csv',
        dev='data/ids-sst-dev.csv',
        test='data/ids-sst-test-student.csv',
        option=args.option,
        dev_out = 'predictions/'+args.option+'-sst-dev-out.csv',
        test_out = 'predictions/'+args.option+'-sst-test-out.csv'
    )

    train(config)

    print('Evaluating on SST...')
    test(config)

    print('Training Sentiment Classifier on cfimdb...')
    config = SimpleNamespace(
        filepath='cfimdb-classifier.pt',
        lr=args.lr,
        use_gpu=args.use_gpu,
        epochs=args.epochs,
        batch_size=8,
        hidden_dropout_prob=args.hidden_dropout_prob,
        train='data/ids-cfimdb-train.csv',
        dev='data/ids-cfimdb-dev.csv',
        test='data/ids-cfimdb-test-student.csv',
        option=args.option,
        dev_out = 'predictions/'+args.option+'-cfimdb-dev-out.csv',
        test_out = 'predictions/'+args.option+'-cfimdb-test-out.csv'
    )

    train(config)

    print('Evaluating on cfimdb...')
    test(config)
