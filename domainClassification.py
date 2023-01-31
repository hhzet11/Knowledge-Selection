# %%
from google.colab import drive
drive.mount('/content/drive/')

# %%
%cd /content/drive/MyDrive/MultiWoz/alexa-with-dstc9-track1-dataset

# %%
!pip install transformers
#!pip install datasets

# %%
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, BertForTokenClassification, DistilBertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset, TensorDataset
import os
from tqdm import tqdm
from tqdm.notebook import tqdm
import json

# %%
SPECIAL_TOKENS = {
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "pad_token": "<pad>",
    "additional_special_tokens": ["<speaker1>", "<speaker2>", "<knowledge_sep>", "<knowledge_tag>"],
}
SPECIAL_TOKENS_VALUES = ["<bos>", "<eos>", "<pad>", "<speaker1>", "<speaker2>", "<knowledge_sep>", "<knowledge_tag>"]


# %%
class DatasetWalker(object):
    def __init__(self, dataset, dataroot, labels=False, labels_file=None):
        path = os.path.join(os.path.abspath(dataroot))
            
        if dataset not in ['train', 'val', 'test']:
            raise ValueError('Wrong dataset name: %s' % (dataset))

        logs_file = os.path.join(path, dataset, 'logs.json')
        with open(logs_file, 'r') as f:
            self.logs = json.load(f)

        self.labels = None

        if labels is True:
            if labels_file is None:
                labels_file = os.path.join(path, dataset, 'labels.json')

            with open(labels_file, 'r') as f:
                self.labels = json.load(f)

    def __iter__(self):
        if self.labels is not None:
            for log, label in zip(self.logs, self.labels):
                yield(log, label)
        else:
            for log in self.logs:
                yield(log, None)

    def __len__(self, ):
        return len(self.logs)

# %%
class KnowledgeReader(object):
    def __init__(self, dataroot, knowledge_file):
        path = os.path.join(os.path.abspath(dataroot))

        with open(os.path.join(path, knowledge_file), 'r') as f:
            self.knowledge = json.load(f)

    def get_domain_list(self):
        return list(self.knowledge.keys())

    def get_entity_list(self, domain):
        if domain not in self.get_domain_list():
            raise ValueError("invalid domain name")

        entity_ids = []
        for entity_id in self.knowledge[domain].keys():
            try:
                entity_id = int(entity_id)
                entity_ids.append(int(entity_id))
            except:
                pass

        result = []
        for entity_id in sorted(entity_ids):
            entity_name = self.knowledge[domain][str(entity_id)]['name']
            result.append({'id': entity_id, 'name': entity_name})

        return result

    def get_entity_name(self, domain, entity_id):
        if domain not in self.get_domain_list():
            raise ValueError("invalid domain name: %s" % domain)

        if str(entity_id) not in self.knowledge[domain]:
            raise ValueError("invalid entity id: %s" % str(entity_id))

        result = self.knowledge[domain][str(entity_id)]['name'] or None

        return result


    def get_doc_list(self, domain=None, entity_id=None):
        if domain is None:
            domain_list = self.get_domain_list()
        else:
            if domain not in self.get_domain_list():
                raise ValueError("invalid domain name: %s" % domain)
            domain_list = [domain]

        result = []
        for domain in domain_list:
            if entity_id is None:
                for item_id, item_obj in self.knowledge[domain].items():
                    item_name = self.get_entity_name(domain, item_id)
                    
                    if item_id != '*':
                        item_id = int(item_id)

                    for doc_id, doc_obj in item_obj['docs'].items():
                        result.append({'domain': domain, 'entity_id': item_id, 'entity_name': item_name, 'doc_id': doc_id, 'doc': {'title': doc_obj['title'], 'body': doc_obj['body']}})
            else:
                if str(entity_id) not in self.knowledge[domain]:
                    raise ValueError("invalid entity id: %s" % str(entity_id))

                entity_name = self.get_entity_name(domain, entity_id)
                
                entity_obj = self.knowledge[domain][str(entity_id)]
                for doc_id, doc_obj in entity_obj['docs'].items():
                  
                    result.append({'domain': domain, 'entity_id': entity_id, 'entity_name': entity_name, 'doc_id': doc_id, 
                                   'doc': {'title': doc_obj['title'], 'body': doc_obj['body']}})
        return result

    def get_doc(self, domain, entity_id, doc_id):
        if domain not in self.get_domain_list():
            raise ValueError("invalid domain name: %s" % domain)

        if str(entity_id) not in self.knowledge[domain]:
            raise ValueError("invalid entity id: %s" % str(entity_id))

        entity_name = self.get_entity_name(domain, entity_id)

        if str(doc_id) not in self.knowledge[domain][str(entity_id)]['docs']:
            raise ValueError("invalid doc id: %s" % str(doc_id))

        doc_obj = self.knowledge[domain][str(entity_id)]['docs'][str(doc_id)]
        result = {'domain': domain, 'entity_id': entity_id, 'entity_name': entity_name, 'doc_id': doc_id, 'doc': {'title': doc_obj['title'], 'body': doc_obj['body']}}

        return result

# %%
def truncate_sequences(sequences, max_length):
    words_to_cut = sum(list(map(len, sequences))) - max_length
    if words_to_cut <= 0:
        return sequences

    while words_to_cut > len(sequences[0]):
        words_to_cut -= len(sequences[0])
        sequences = sequences[1:]
    
    sequences[0] = sequences[0][words_to_cut:]
    return sequences


# %%
class BaseDataset(torch.utils.data.Dataset): 
  def __init__(self, tokenizer, split_type, labels=True, labels_file = None) :
    
    self.dataroot = dataroot
    self.tokenizer = tokenizer
    self.split_type = split_type
    
    self.SPECIAL_TOKENS = SPECIAL_TOKENS
    self.SPECIAL_TOKENS_VALUES = SPECIAL_TOKENS_VALUES
    self.bos = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["bos_token"])
    self.eos = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["eos_token"])
    self.pad = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["pad_token"])
    self.speaker1, self.speaker2, self.knowledge_sep, self.knowledge_tag = self.tokenizer.convert_tokens_to_ids(
      self.SPECIAL_TOKENS["additional_special_tokens"]
    )
    self.knowledge_sep_token = self.SPECIAL_TOKENS["additional_special_tokens"][2]
    self.knowledge_tag_token = self.SPECIAL_TOKENS["additional_special_tokens"][3]

    self.dataset_walker = DatasetWalker(split_type, labels=labels, dataroot = self.dataroot, labels_file=labels_file)
    self.dialogs = self._prepare_conversations()
  
    self.knowledge_reader = KnowledgeReader(self.dataroot, "knowledge.json")
    self.knowledge_key, self.knowledge, self.snippets = self._prepare_knowledge()
    self._create_domain_examples()

  def _prepare_conversations(self):
    tokenized_dialogs = []
    for i, (log, label) in enumerate(tqdm(self.dataset_walker)) :
      dialog = {}
      dialog["id"] = i
      dialog["log"] = log
      if label is not None :
        if "response" in label:
          label["response_tokenized"] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(label["response"]))
      dialog["label"] = label
      tokenized_dialogs.append(dialog)

    return tokenized_dialogs

  def _knowledge_to_string(self, doc, name = ""):
    return doc["body"]

  def _prepare_knowledge(self): 
    tokenized_snippets = dict()
    knowledge_key = []
    self.knowledge_docs = self.knowledge_reader.get_doc_list()
    for snippet in self.knowledge_docs :
      key = "{}__{}__{}".format(snippet["domain"], str(snippet["entity_id"]) or "", snippet["doc_id"])
      knowledge_key.append(key)
      knowledge = self._knowledge_to_string(snippet["doc"], name = snippet["entity_name"] or "")
      
      ##토큰화하는 과정!
      tokenized_knowledge = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(knowledge))
      tokenized_snippets[key] = tokenized_knowledge[:128]

    return knowledge_key, knowledge, tokenized_snippets

  def _create_domain_examples(self):
    self.d_examples = []

    for dialog in tqdm(self.dialogs) :
      dialog_id = dialog["id"]
      label = dialog["label"]
      dialog = dialog["log"]

      if label is None :
        label = {"target" : False}
      
      target = label["target"]

      if not target :
        continue

      history = [turn["text"] for turn in dialog[-3:]]
      
      if target :

        if "knowledge" not in label :
          label["knowledge"] = [self.knowledge_docs[0]]

        knowledge = label["knowledge"][0]
        domain = knowledge["domain"]

      else : 
        domain = None
      
      self.d_examples.append({
          "history" : ' '.join(str(s) for s in history),
          "domain" : domain,
      })

    return self.d_examples
                
  def __getitem__(self, index):
    raise NotImplementedError
    
  def __len__(self):
    return len(self.examples)

# %%
# 먼저 token화 해야함
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
tokenizer.add_special_tokens(SPECIAL_TOKENS)

dataroot = 'data'
train_base = BaseDataset(tokenizer = tokenizer, split_type = "train")
train_examples = train_base._create_domain_examples()
print(train_examples[:5])

# %%
val_base = BaseDataset(tokenizer = tokenizer, split_type = "val")
val_examples = val_base._create_domain_examples()

dataroot = 'data_eval'
test_base = BaseDataset(tokenizer = tokenizer, split_type = "test")
test_examples = test_base._create_domain_examples()
print(test_examples[:5])

# %%
train_df = pd.DataFrame(train_examples)
val_df = pd.DataFrame(val_examples)
test_df = pd.DataFrame(test_examples)
train_df.head(10)

# %%
train_df['domain'].value_counts()

# %%
val_df['domain'].value_counts()

# %%
test_df['domain'].value_counts()

# %%
"""
## Encoding the Labels
"""

# %%
possible_labels = test_df.domain.unique()

label_dict = {}
for index, possible_label in enumerate(possible_labels):
  label_dict[possible_label] = index

label_dict

# %%
train_df['label'] = train_df.domain.replace(label_dict)
val_df['label'] = val_df.domain.replace(label_dict)
test_df['label'] = test_df.domain.replace(label_dict)
train_df.head(10)

# %%
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case = True)

encoded_data_train = tokenizer.batch_encode_plus(
    train_df.history.values,
    add_special_tokens = True,
    return_attention_mask = True,
    pad_to_max_length = True,
    max_length = 512,
    return_tensors = 'pt',
    truncation = True,
)

encoded_data_val = tokenizer.batch_encode_plus(
    val_df.history.values,
    add_special_tokens = True,
    return_attention_mask = True,
    pad_to_max_length = True,
    max_length = 512,
    return_tensors = 'pt',
    truncation = True,
)

input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train = torch.tensor(train_df.label.values)

input_ids_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
labels_val = torch.tensor(val_df.label.values)

dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

# %%
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import DistilBertForSequenceClassification

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels = len(label_dict),
                                                            output_attentions = False, output_hidden_states = False)

batch_size = 4

dataloader_train = DataLoader(dataset_train, sampler = RandomSampler(dataset_train), batch_size = batch_size)
dataloader_validation = DataLoader(dataset_val, sampler = RandomSampler(dataset_val), batch_size = batch_size)

# %%
from transformers import AdamW, get_linear_schedule_with_warmup

optimizer = AdamW(model.parameters(),
                  lr=1e-5, 
                  eps=1e-8)
                  
epochs = 10

scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=0,
                                            num_training_steps=len(dataloader_train)*epochs)

# %%
from sklearn.metrics import f1_score, classification_report, precision_score, recall_score

def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')

def precision_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return precision_score(labels_flat, preds_flat, average='weighted')

def recall_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return recall_score(labels_flat, preds_flat, average='weighted')

def accuracy_per_class(preds, labels):
    label_dict_inverse = {v: k for k, v in label_dict.items()}
    
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')

def classification_report_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return classification_report(labels_flat, preds_flat)

# %%
%cd /content/drive/MyDrive/MultiWoz

# %%
import random

seed_val = 17
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print(device)

def evaluate(dataloader_val):

    model.eval()
    
    loss_val_total = 0
    predictions, true_vals = [], []
    
    for batch in dataloader_val:
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():        
            outputs = model(**inputs)
            
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    
    loss_val_avg = loss_val_total/len(dataloader_val) 
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
            
    return loss_val_avg, predictions, true_vals
    
for epoch in tqdm(range(1, epochs+1)):
    
    model.train()
    
    loss_train_total = 0

    progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
    for batch in progress_bar:

        model.zero_grad()
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }       

        outputs = model(**inputs)
        
        loss = outputs[0]
        loss_train_total += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        
        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
         
        
    torch.save(model.state_dict(), f'data_volume/finetuned_BERT_epoch_{epoch}.model')
        
    tqdm.write(f'\nEpoch {epoch}')
    
    loss_train_avg = loss_train_total/len(dataloader_train)            
    tqdm.write(f'Training loss: {loss_train_avg}')
    
    val_loss, predictions, true_vals = evaluate(dataloader_validation)
    val_f1 = f1_score_func(predictions, true_vals)
    tqdm.write(f'Validation loss: {val_loss}')
    tqdm.write(f'F1 Score (Weighted): {val_f1}')

# %%
encoded_data_test = tokenizer.batch_encode_plus(
    test_df.history.values,
    add_special_tokens = True,
    return_attention_mask = True,
    pad_to_max_length = True,
    max_length = 512,
    return_tensors = 'pt'
)

input_ids_test = encoded_data_test['input_ids']
attention_masks_test = encoded_data_test['attention_mask']
labels_test = torch.tensor(test_df.label.values)

dataset_test = TensorDataset(input_ids_test, attention_masks_test, labels_test)
dataloader_test = DataLoader(dataset_test, sampler = RandomSampler(dataset_test), batch_size = 1)

# %%
#model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(label_dict),
#                                                            output_attentions=False, output_hidden_states=False)

#model.to(device)
#model.load_state_dict(torch.load('data_volume/finetuned_BERT_epoch_10.model', map_location = torch.device('cuda')))
_, predictions, true_test = evaluate(dataloader_test)
print(classification_report_func(predictions, true_test))
print("F1 score : ", f1_score_func(predictions, true_test))
print("precision score : ", precision_score_func(predictions, true_test))
print("recall score : ", recall_score_func(predictions, true_test))
accuracy_per_class(predictions, true_test)

# %%
