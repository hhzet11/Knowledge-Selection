# %%
from google.colab import drive
drive.mount('/content/drive/')

# %%
%cd /content/drive/MyDrive/MultiWoz/alexa-with-dstc9-track1-dataset

# %%
!pip install transformers
!pip install datasets

# %%
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import os 
import json

import transformers
from transformers import AutoTokenizer, DistilBertForTokenClassification, DebertaForTokenClassification, ElectraTokenizer, ElectraForTokenClassification

from torch.optim import AdamW
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

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
  def __init__(self, split_type, labels=True, labels_file = None) :
    
    self.dataroot = dataroot
    self.split_type = split_type

    self.dataset_walker = DatasetWalker(split_type, labels=labels, dataroot = self.dataroot, labels_file=labels_file)
    self.dialogs = self._prepare_conversations()
  
    self.knowledge_reader = KnowledgeReader(self.dataroot, "knowledge.json")
    self.knowledge_key, self.knowledge = self._prepare_knowledge()
    self._create_entity_examples()

  def _prepare_conversations(self):
    tokenized_dialogs = []
    for i, (log, label) in enumerate(tqdm(self.dataset_walker)) :
      dialog = {}
      dialog["id"] = i
      dialog["log"] = log
      dialog["label"] = label
      tokenized_dialogs.append(dialog)
    return tokenized_dialogs

  def _knowledge_to_string(self, doc, name = ""):
    return doc["body"]

  def _prepare_knowledge(self): 
    self.knowledge_key = []
    self.knowledge_docs = self.knowledge_reader.get_doc_list()
    for snippet in self.knowledge_docs :
      key = [snippet["domain"], str(snippet["entity_id"]) or "", snippet["entity_name"]]
      self.knowledge_key.append(key)
      knowledge = self._knowledge_to_string(snippet["doc"], name = snippet["entity_name"] or "")
   
    return self.knowledge_key, knowledge

  def _create_entity_examples(self):
    self.e_examples = []
    for dialog in tqdm(self.dialogs) :
      dialog_id = dialog["id"]
      label = dialog["label"]
      dialog = dialog["log"]
      if label is None :
        label = {"target" : False}

      target = label["target"]
      if not target:
        continue

      history = [turn["text"] for turn in dialog[-1:]]
      
      if target :
        if "knowledge" not in label :
          label["knowledge"] = [self.knowledge[0]]
        
        knowledge = label["knowledge"][0]
        if knowledge["domain"] in ['restaurant', 'hotel', 'attraction']:
          for snippet in self.knowledge_key :
            if knowledge['domain'] == snippet[0] and knowledge['entity_id'] == int(snippet[1]):
              entity = snippet[2].lower()
        else :
          continue
      
      #else : 
      #  entity = None

      self.e_examples.append({
          "history" : ' '.join(str(s) for s in history),
          "entity" : entity
      })

    return self.e_examples
                
  def __getitem__(self, index):
    raise NotImplementedError
    
  def __len__(self):
    return len(self.examples)

# %%
dataroot = 'data'
train_base = BaseDataset(split_type = "train")
train_examples = train_base._create_entity_examples()

val_base = BaseDataset(split_type = "val")
val_examples = val_base._create_entity_examples()

print(train_examples[:5])

# %%
dataroot = 'data_eval'
test_base = BaseDataset(split_type = "test")
test_examples = test_base._create_entity_examples()
print(test_examples[:5])

# %%
tokenizer = ElectraTokenizer.from_pretrained("bhadresh-savani/electra-base-discriminator-finetuned-conll03-english")
special_tokens = [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]

# %%
train_df = pd.DataFrame(train_examples)
val_df = pd.DataFrame(val_examples)
test_df = pd.DataFrame(test_examples)
train_df.head(10)

# %%
import nltk
import re
from nltk.tokenize import TreebankWordTokenizer

twt = TreebankWordTokenizer()

train_df['history'] = train_df['history'].str.lower()
train_df['entity'] = train_df['entity'].str.strip()
train_df['span_list'] = train_df['history'].apply(twt.span_tokenize)
train_df[['history', 'span_list']].head()

val_df['history'] = val_df['history'].str.lower()
val_df['entity'] = val_df['entity'].str.strip()
val_df['span_list'] = val_df['history'].apply(twt.span_tokenize)

test_df['history'] = test_df['history'].str.lower()
test_df['entity'] = test_df['entity'].str.strip()
test_df['span_list'] = test_df['history'].apply(twt.span_tokenize)

# %%
def get_iob(text, selected_text, twt=twt):
    """
    :param text: text
    :param selected_text: selected_text
    :param twt: Tokenizer that has `span_tokenize()` function
    :param sentiment: add sentiment info to IB tag e.g. `B-positive`, `I-neutral`
    :returns: iob string
    """
    iob_list = []

    if selected_text != None and re.search(re.escape(selected_text), text) != None:
      # 존재하지 않는 경우 해결필요
      start, end = re.search(re.escape(selected_text), text).span()
      # list of (start_idx, stop_idx)
      span_list = twt.span_tokenize(text)
      
      for start_sp, end_sp in span_list:
          iob_tag = 'O'
          if start_sp == start:
              iob_tag = 'B-ENTITY'
          elif start < start_sp and end_sp <= end:
              iob_tag = 'I-ENTITY'
            
          iob_list.append(iob_tag)

    else :
      span_list = twt.span_tokenize(text)
      for start_sp, end_sp in span_list:
        iob_list.append('O')

    return ' '.join(iob_list)
    

def get_iob_format_from_row(row, twt=twt):
    return get_iob(row.history, row.entity, twt=twt)

# %%
train_df['label'] = train_df.apply(get_iob_format_from_row, axis = 1)
val_df['label'] = val_df.apply(get_iob_format_from_row, axis = 1)
test_df['label'] = test_df.apply(get_iob_format_from_row, axis = 1)

# %%
num = 0
for i in val_df['label'] :
  if 'B-ENTITY' in i:
    num += 1
print(num/len(val_df))

# %%
train_df.drop(['entity', 'span_list'], axis = 1, inplace = True)
train_df.rename(columns = {'history' : 'sentence', 'label' : 'tags'}, inplace = True)
train_df

# %%
val_df.drop(['entity', 'span_list'], axis = 1, inplace = True)
val_df.rename(columns = {'history' : 'sentence', 'label' : 'tags'}, inplace = True)
test_df.drop(['entity', 'span_list'], axis = 1, inplace = True)
test_df.rename(columns = {'history' : 'sentence', 'label' : 'tags'}, inplace = True)

# %%
class DistilbertNER(nn.Module):
  """
  Implement NN class based on distilbert pretrained from Hugging face.
  Inputs : 
    tokens_dim : int specifyng the dimension of the classifier
  """
  
  def __init__(self, tokens_dim):
    super(DistilbertNER,self).__init__()
    
    if type(tokens_dim) != int:
            raise TypeError('Please tokens_dim should be an integer')

    if tokens_dim <= 0:
          raise ValueError('Classification layer dimension should be at least 1')

    self.pretrained = ElectraForTokenClassification.from_pretrained("bhadresh-savani/electra-base-discriminator-finetuned-conll03-english", ignore_mismatched_sizes=True, num_labels = tokens_dim) #set the output of each token classifier = unique_lables


  def forward(self, input_ids, attention_mask, labels = None): #labels are needed in order to compute the loss
    """
  Forwad computation of the network
  Input:
    - inputs_ids : from model tokenizer
    - attention :  mask from model tokenizer
    - labels : if given the model is able to return the loss value
  """

    #inference time no labels
    if labels == None:
      out = self.pretrained(input_ids = input_ids, attention_mask = attention_mask )
      return out

    out = self.pretrained(input_ids = input_ids, attention_mask = attention_mask , labels = labels)
    return out

# %%
class NerDataset(torch.utils.data.Dataset):
  """
  Custom dataset implementation to get (text,labels) tuples
  Inputs:
   - df : dataframe with columns [tags, sentence]
  """
  
  def __init__(self, df):
    if not isinstance(df, pd.DataFrame):
      raise TypeError('Input should be a dataframe')
    
    if "tags" not in df.columns or "sentence" not in df.columns:
      raise ValueError("Dataframe should contain 'tags' and 'sentence' columns")

    tags_list = [i.split() for i in df["tags"].values.tolist()]
    texts = df["sentence"].values.tolist()

    self.texts = [tokenizer(text, padding = "max_length", truncation = True, return_tensors = "pt") for text in texts]
    self.labels = [match_tokens_labels(text, tags) for text,tags in zip(self.texts, tags_list)]

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    batch_text = self.texts[idx]
    batch_labels = self.labels[idx]

    return batch_text, torch.LongTensor(batch_labels)

# %%
class MetricsTracking():
  """
  In order make the train loop lighter I define this class to track all the metrics that we are going to measure for our model.
  """
  def __init__(self):

    self.total_acc = 0
    self.total_f1 = 0
    self.total_precision = 0
    self.total_recall = 0

  def update(self, predictions, labels , ignore_token = -100):
    '''
    Call this function every time you need to update your metrics.
    Where in the train there was a -100, were additional token that we dont want to label, so remove them.
    If we flatten the batch its easier to access the indexed = -100
    '''  
    predictions = predictions.flatten()
    labels = labels.flatten()
    
    predictions = predictions[labels != ignore_token]
    labels = labels[labels != ignore_token]

    predictions = predictions.to("cpu")
    labels = labels.to("cpu")

    acc = accuracy_score(labels,predictions)
    f1 = f1_score(labels, predictions, average = "macro")
    precision = precision_score(labels, predictions, average = "macro")
    recall = recall_score(labels, predictions, average = "macro")

    self.total_acc  += acc
    self.total_f1 += f1
    self.total_precision += precision
    self.total_recall  += recall

  def return_avg_metrics(self,data_loader_size):
    n = data_loader_size
    metrics = {
        "acc": round(self.total_acc / n ,3), 
        "f1": round(self.total_f1 / n, 3), 
        "precision" : round(self.total_precision / n, 3), 
        "recall": round(self.total_recall / n, 3)
          }
    return metrics   

# %%
def tags_2_labels(tags : str, tag2idx : dict):
  '''
  Method that takes a list of tags and a dictionary mapping and returns a list of labels (associated).
  Used to create the "label" column in df from the "tags" column.
  '''
  return [tag2idx[tag] if tag in tag2idx else unseen_label for tag in tags.split()] 

# %%
def tags_mapping(tags_series : pd.Series):
  """
  tag_series = df column with tags for each sentence.
  Returns:
    - dictionary mapping tags to indexes (label)
    - dictionary mappign inedexes to tags
    - The label corresponding to tag 'O'
    - A set of unique tags ecountered in the trainind df, this will define the classifier dimension
  """

  if not isinstance(tags_series, pd.Series):
      raise TypeError('Input should be a padas Series')

  unique_tags = set()
  
  for tag_list in train_df["tags"]:
    for tag in tag_list.split():
      unique_tags.add(tag)


  tag2idx = {k:v for v,k in enumerate(sorted(unique_tags))}
  idx2tag = {k:v for v,k in tag2idx.items()}

  unseen_label = tag2idx["O"]

  return tag2idx, idx2tag, unseen_label, unique_tags

# %%
def match_tokens_labels(tokenized_input, tags, ignore_token = -100):
        '''
        Used in the custom dataset.
        -100 will be tha label used to match additional tokens like [CLS] [PAD] that we dont care about. 
        Inputs : 
          - tokenized_input : tokenizer over the imput text -> {input_ids, attention_mask}
          - tags : is a single label array -> [O O O O O O O O O O O O O O B-tim O]
        
        Returns a list of labels that match the tokenized text -> [-100, 3,5,6,-100,...]
        '''

        #gives an array [ None , 0 , 1 ,2 ,... None]. Each index tells the word of reference of the token
        word_ids = tokenized_input.word_ids()

        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:

            if word_idx is None:
                label_ids.append(ignore_token)

            #if its equal to the previous word we can add the same label id of the provious or -100 
            else :
                try:
                  reference_tag = tags[word_idx]
                  label_ids.append(tag2idx[reference_tag])
                except:
                  label_ids.append(ignore_token)
              
            
            previous_word_idx = word_idx

        return label_ids

# %%
def freeze_model(model,num_layers = 1):
  """
  Freeze last num_layers of a model to prevent catastrophic forgetting.
  Doesn't seem to work weel, its better to fine tune the entire netwok
  """
  for id , params in enumerate(model.parameters()):
    if id == len(list(model.parameters())) - num_layers: 
      print("last layer unfreezed")
      params.requires_grad = True
    else:
      params.requires_grad = False
  return model

# %%
def train_loop(model, train_dataset, dev_dataset, optimizer,  batch_size, epochs):
  
  train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
  dev_dataloader = DataLoader(dev_dataset, batch_size = batch_size, shuffle = True)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = model.to(device)

  for epoch in range(epochs) : 
    
    train_metrics = MetricsTracking()
    total_loss_train = 0

    model.train() #train mode

    for train_data, train_label in tqdm(train_dataloader):

      train_label = train_label.to(device)
      '''
      squeeze in order to match the sizes. From [batch,1,seq_len] --> [batch,seq_len] 
      '''
      mask = train_data['attention_mask'].squeeze(1).to(device)
      input_id = train_data['input_ids'].squeeze(1).to(device)

      optimizer.zero_grad()
      
      output = model(input_id, mask, train_label)
      loss, logits = output.loss, output.logits
      predictions = logits.argmax(dim= -1) 

      #compute metrics
      train_metrics.update(predictions, train_label)
      total_loss_train += loss.item()

      #grad step
      loss.backward()
      optimizer.step()
    
    '''
    EVALUATION MODE
    '''            
    model.eval()

    dev_metrics = MetricsTracking()
    total_loss_dev = 0
    
    with torch.no_grad():
      for dev_data, dev_label in dev_dataloader:

        dev_label = dev_label.to(device)

        mask = dev_data['attention_mask'].squeeze(1).to(device)
        input_id = dev_data['input_ids'].squeeze(1).to(device)

        output = model(input_id, mask, dev_label)
        loss, logits = output.loss, output.logits

        predictions = logits.argmax(dim= -1)     

        dev_metrics.update(predictions, dev_label)
        total_loss_dev += loss.item()
    
    train_results = train_metrics.return_avg_metrics(len(train_dataloader))
    dev_results = dev_metrics.return_avg_metrics(len(dev_dataloader))

    print(f"TRAIN \nLoss: {total_loss_train / len(train_dataset)} \nMetrics {train_results}\n" ) 
    print(f"VALIDATION \nLoss {total_loss_dev / len(dev_dataset)} \nMetrics{dev_results}\n" ) 

# %%
#create tag-label mapping
tag2idx, idx2tag , unseen_label, unique_tags = tags_mapping(train_df["tags"])

#create the label column from tag. Unseen labels will be tagged as "O"
for df in [train_df, val_df, test_df]:
  df["labels"] = df["tags"].apply(lambda tags : tags_2_labels(tags, tag2idx))

# %%
#original text
text = train_df["sentence"].values.tolist()

#toeknized text
#distilbert_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bhadresh-savani/electra-base-discriminator-finetuned-conll03-english")
text_tokenized = tokenizer(text , max_length = 512, padding = "max_length" , truncation = True, return_tensors = "pt" )

#mapping token to original word
word_ids = text_tokenized.word_ids()

# %%
model = DistilbertNER(len(unique_tags))
#Prevent Catastrofic Forgetting
#model = freeze_model(model, num_layers = 2)

#datasets
train_dataset = NerDataset(train_df)
dev_dataset = NerDataset(val_df)

lr = 1e-5
optimizer = AdamW(model.parameters(), lr=lr)  


#MAIN
parameters = {
    "model": model,
    "train_dataset": train_dataset,
    "dev_dataset" : dev_dataset,
    "optimizer" : optimizer,
    "batch_size" : 16,
    "epochs" : 10
}

train_loop(**parameters)

# %%
test_dataset = NerDataset(test_df)
test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle = True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = model.to(device)
model.eval()

test_metrics = MetricsTracking()
with torch.no_grad():
  for test_data, test_label in test_dataloader:
    test_label = test_label.to(device)
    mask = test_data['attention_mask'].squeeze(1).to(device)
    input_id = test_data['input_ids'].squeeze(1).to(device)

    output = model(input_id, mask, test_label)
    loss, logits = output.loss, output.logits

    predictions = logits.argmax(dim= -1)     

    test_metrics.update(predictions, test_label)

test_results = test_metrics.return_avg_metrics(len(test_dataloader))
print(test_results)

# %%
"""
#Deberta
"""

# %%
class DeBERTaNER(nn.Module):
  """
  Implement NN class based on distilbert pretrained from Hugging face.
  Inputs : 
    tokens_dim : int specifyng the dimension of the classifier
  """
  
  def __init__(self, tokens_dim):
    super(DeBERTaNER,self).__init__()
    
    if type(tokens_dim) != int:
            raise TypeError('Please tokens_dim should be an integer')

    if tokens_dim <= 0:
          raise ValueError('Classification layer dimension should be at least 1')

    self.pretrained = DebertaForTokenClassification.from_pretrained("dbsamu/deberta-base-finetuned-ner", num_labels = tokens_dim) #set the output of each token classifier = unique_lables


  def forward(self, input_ids, attention_mask, labels = None): #labels are needed in order to compute the loss
    """
  Forwad computation of the network
  Input:
    - inputs_ids : from model tokenizer
    - attention :  mask from model tokenizer
    - labels : if given the model is able to return the loss value
  """

    #inference time no labels
    if labels == None:
      out = self.pretrained(input_ids = input_ids, attention_mask = attention_mask )
      return out

    out = self.pretrained(input_ids = input_ids, attention_mask = attention_mask , labels = labels)
    return out

# %%
#original text
text = train_df["sentence"].values.tolist()

#toeknized text
deberta_tokenizer = AutoTokenizer.from_pretrained("dbsamu/deberta-base-finetuned-ner")
text_tokenized = deberta_tokenizer(text , max_length = 512, padding = "max_length" , truncation = True, return_tensors = "pt" )

#mapping token to original word
word_ids = text_tokenized.word_ids()

deberta_model = DeBERTaNER(len(unique_tags))
#Prevent Catastrofic Forgetting
#model = freeze_model(model, num_layers = 2)

#datasets
train_dataset = NerDataset(train_df)
dev_dataset = NerDataset(val_df)

lr = 1e-5
optimizer = AdamW(distilbert_model.parameters(), lr=lr)  


#MAIN
parameters = {
    "model": deberta_model,
    "train_dataset": train_dataset,
    "dev_dataset" : dev_dataset,
    "optimizer" : optimizer,
    "batch_size" : 16,
    "epochs" : 10
}

train_loop(**parameters)

# %%
test_dataset = NerDataset(test_df)
test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle = True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = model.to(device)
deberta_model.eval()

test_metrics = MetricsTracking()
with torch.no_grad():
  for test_data, test_label in test_dataloader:
    test_label = test_label.to(device)
    mask = test_data['attention_mask'].squeeze(1).to(device)
    input_id = test_data['input_ids'].squeeze(1).to(device)

    output = deberta_model(input_id, mask, test_label)
    loss, logits = output.loss, output.logits

    predictions = logits.argmax(dim= -1)     

    test_metrics.update(predictions, test_label)

test_results = test_metrics.return_avg_metrics(len(test_dataloader))
print(test_results)