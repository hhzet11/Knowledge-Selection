# %%
from google.colab import drive
drive.mount('/content/drive/')

# %%
%cd /content/drive/MyDrive/MultiWoz/alexa-with-dstc9-track1-dataset

# %%
!pip install transformers==2.10.0

# %%
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from copy import deepcopy

import random
import os 
import json

import transformers
from transformers import AutoTokenizer, DistilBertForTokenClassification, ElectraTokenizer, ElectraForTokenClassification

from torch.optim import AdamW
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader
from itertools import chain

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
SPECIAL_TOKENS = {
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "pad_token": "<pad>",
    "additional_special_tokens": ["<speaker1>", "<speaker2>", "<knowledge_sep>", "<knowledge_tag>"],
}
SPECIAL_TOKENS_VALUES = ["<bos>", "<eos>", "<pad>", "<speaker1>", "<speaker2>", "<knowledge_sep>", "<knowledge_tag>"]


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
task = "selection"
dataroot = 'data'
negative_sample_method = 'oracle'

class BaseDataset(torch.utils.data.Dataset): 
  def __init__(self, tokenizer, split_type, labels=True, labels_file = None) :
    
    self.dataroot = dataroot
    self.tokenizer = tokenizer
    self.split_type = split_type

    self.negative_sample_method = negative_sample_method
    self.n_candidates = 2
    
    self.SPECIAL_TOKENS = SPECIAL_TOKENS
    self.SPECIAL_TOKENS_VALUES = SPECIAL_TOKENS_VALUES
    self.bos = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["bos_token"])
    self.eos = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["eos_token"])
    self.pad = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["pad_token"])
    self.speaker1, self.speaker2, self.knowledge_sep, self.knowledge_tag = self.tokenizer.convert_tokens_to_ids(
      self.SPECIAL_TOKENS["additional_special_tokens"]
    )
    self.knowledge_sep_token = self.SPECIAL_TOKENS["additional_special_tokens"][2]
    #self.knowledge_tag_token = self.SPECIAL_TOKENS["additional_special_tokens"][3]

    self.all_response_tokenized = []    
    self.dataset_walker = DatasetWalker(split_type, labels=labels, dataroot = self.dataroot, labels_file=labels_file)
    self.dialogs = self._prepare_conversations()
    self.all_response_tokenized = list(map(eval, set(map(str, self.all_response_tokenized))))

    self.knowledge_reader = KnowledgeReader(self.dataroot, "knowledge.json")
    self.knowledge, self.snippets = self._prepare_knowledge() #knowledge_key baseline_dataset.py에서 추가
    
    self._create_examples()

  def _prepare_conversations(self):
    tokenized_dialogs = []
    for i, (log, label) in enumerate(tqdm(self.dataset_walker)) :
      dialog = {}
      dialog["id"] = i
      dialog["log"] = log
      if label is not None :
        if "response" in label:
          label["response_tokenized"] = self.tokenizer.convert_tokens_to_ids(
              self.tokenizer.tokenize(label["response"])
          )
          self.all_response_tokenized.append(label["response_tokenized"])
      dialog["label"] = label
      tokenized_dialogs.append(dialog)

    return tokenized_dialogs

  def _knowledge_to_string(self, doc, name = ""):
    return doc["body"]

  def _prepare_knowledge(self): 
    knowledge = self.knowledge_reader.knowledge
    self.knowledge_docs = self.knowledge_reader.get_doc_list()
    
    tokenized_snippets = dict()
    for snippet in self.knowledge_docs :
      key = "{}__{}__{}".format(snippet["domain"], str(snippet["entity_id"]) or "", snippet["doc_id"])
      knowledge = self._knowledge_to_string(snippet["doc"], name = snippet["entity_name"] or "")
      
      ##토큰화하는 과정!
      tokenized_knowledge = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(knowledge))
      tokenized_snippets[key] = tokenized_knowledge[:128]

    return knowledge, tokenized_snippets

  def _create_examples(self):
    self.examples = []
    for dialog in tqdm(self.dialogs):
      dialog_id = dialog["id"]
      label = dialog["label"]
      dialog = dialog["log"]
      if label is None:
        label = {"target": False}

      target = label["target"]

      if not target and task != "detection":
        continue
            
      #history tokenize 필요해?
      history = [
        self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(turn["text"]))
        for turn in dialog
      ]

      #response가 generation task에 필요하지 않나?
      gt_resp = label.get("response", "")
      tokenized_gt_resp = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(gt_resp))
      
      #args에서 history_max_utterances :1000000 >> history해결 필요!!!!
      #history_max_tokens & knowledge_max_tokens : 128 값 알아내기
      truncated_history = history[-1000000:]
      truncated_history = truncate_sequences(truncated_history, 128)

      if target:
        if "knowledge" not in label:
          label["knowledge"] = [self.knowledge_docs[0]]

        knowledge = label["knowledge"][0]
        knowledge_key = "{}__{}__{}".format(knowledge["domain"], knowledge["entity_id"], knowledge["doc_id"])
        
        # find snippets with same entity as candidates
        prefix = "{}__{}".format(knowledge["domain"], knowledge["entity_id"])
        knowledge_candidates = [
          cand
          for cand in self.snippets.keys() 
          #if "__".join(cand.split("__")[:-1]) == prefix
          if cand.startswith(prefix)
        ]
                
        if self.split_type == "train" and self.negative_sample_method == "oracle":
          # if there's not enough candidates during training, we just skip this example
          if len(knowledge_candidates) < self.n_candidates:
            continue
        used_knowledge = self.snippets[knowledge_key]
        used_knowledge = used_knowledge[:-1]
      
      else:
        knowledge_candidates = None
        used_knowledge = []

      self.examples.append({
        "history": truncated_history,
        "knowledge": used_knowledge,
        "candidates": knowledge_candidates,
        "response": tokenized_gt_resp,
        "response_text": gt_resp,
        "label": label,
        "knowledge_seeking": target,
        "dialog_id": dialog_id
      })


  def build_input_from_segments(self, knowledge, history, response, with_eos=True):
    """ Build a sequence of input from 3 segments: knowledge, history and last reply """
    instance = {}

    sequence = [[self.bos] + knowledge] + history + [response + ([self.eos] if with_eos else [])]
    sequence_with_speaker = [
      [self.speaker1 if (len(sequence) - i) % 2 == 0 else self.speaker2] + s
      for i, s in enumerate(sequence[1:])
    ]
    sequence = [sequence[0]] + sequence_with_speaker
    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] = [self.speaker2 if i % 2 else self.speaker1 for i, s in enumerate(sequence) for _ in s]
    instance["mc_token_ids"] = len(instance["input_ids"]) - 1
    instance["lm_labels"] = ([-100] * sum(len(s) for s in sequence[:-1])) + [-100] + sequence[-1][1:]

    return instance, sequence

  def __getitem__(self, index):
    raise NotImplementedError
    
  def __len__(self):
    return len(self.examples)

# %%
def pad_ids(arrays, padding, max_length = -1):
  if max_length < 0:
    max_length = max(list(map(len, arrays)))
  
  arrays = [
      array + [padding] * (max_length - len(array))
      for array in arrays
  ]

  return arrays

# %%
class KnowledgeSelectionDataset(BaseDataset):

    def __init__(self, tokenizer, split_type, labels=True, labels_file=None):
        super(KnowledgeSelectionDataset, self).__init__(tokenizer, split_type, labels, labels_file)
        #self.examples = train_examples

    def _knowledge_to_string(self, doc, name=""):
        join_str = " %s " % self.knowledge_sep_token
        return join_str.join([name, doc["title"], doc["body"]])

    def __getitem__(self, index):
        example = self.examples[index]
        
        this_inst = {
            "dialog_id": example["dialog_id"],
            "input_ids": [],
            "token_type_ids": [],
            "mc_token_ids": []
        }

        if self.split_type != "train":
            # if eval_all_snippets is set, we use all snippets as candidates
            if False :
                candidates = list(self.snippets.keys())
            else:
                candidates = example["candidates"]

        else:
            if self.negative_sample_method == "all":
                # 모든 entity 고려 : 너무 많음
                candidates = list(self.snippets.keys())
            elif self.negative_sample_method == "mix":
                #oracle의 결과와 그 만큼의 all을 random으로 설정
                candidates = example["candidates"] + random.sample(list(self.snippets.keys()), k=len(example["candidates"]))
            elif self.negative_sample_method == "oracle":
                # 해당 entity와 관련된 doc만 고려
                candidates = example["candidates"]
            elif self.negative_sample_method == "inDomain" :
                domain = example["label"]["knowledge"][0]["domain"]
                entity_id = example["label"]["knowledge"][0]["entity_id"]
                print(domain)
                can_list = []
                for cand in self.snippets.keys() :
                  can = cand.split("__")
                  if domain in ["hotel", "restaurant"] :
                    if can[0] == domain and int(can[1]) != int(entity_id) :
                      can_list.append(cand)
                  else :
                    can_list = list(self.snippets.keys())
                candidates = example["candidates"] + random.sample(list(can_list), k = len(example["candidates"]))
                  #else :
                    #candidates = example["candidates"] + random.sample(list(self.snippets.keys()), k=len(example["candidates"]))
        
        candidate_keys = candidates
        this_inst["candidate_keys"] = candidate_keys
        candidates = [self.snippets[cand_key] for cand_key in candidates]

        if self.split_type == "train":
            candidates = self._shrink_label_cands(example["knowledge"], candidates) 
        '''
        if example["knowledge"] in candidates : 
          label_idx = candidates.index(example["knowledge"])
        else : 
          label_idx = None
        '''
        
        label_idx = 0
        if example["knowledge"] in candidates :
          label_idx = candidates.index(example["knowledge"])
        this_inst["label_idx"] = label_idx

        for cand in candidates:
            instance, _ = self.build_input_from_segments(
                cand,
                example["history"]
            )
            this_inst["input_ids"].append(instance["input_ids"])
            this_inst["token_type_ids"].append(instance["token_type_ids"])
            this_inst["mc_token_ids"].append(instance["mc_token_ids"])

        return this_inst

    
    def build_input_from_segments(self, knowledge, history):
        """ Build a sequence of input from 2 segments: knowledge and history"""
        instance = {}

        sequence = [[self.bos]] + history
        sequence_with_speaker = [
            [self.speaker1 if (len(sequence) - i) % 2 == 0 else self.speaker2] + s
            for i, s in enumerate(sequence[1:])
        ]
        sequence = [sequence[0]] + sequence_with_speaker + [[self.knowledge_tag] + knowledge + [self.eos]]
        
        instance["input_ids"] = list(chain(*sequence))
        instance["token_type_ids"] = [self.speaker2 if i % 2 else self.speaker1 for i, s in enumerate(sequence[:-1]) for _ in s] + [self.knowledge_tag for _ in sequence[-1]]
        instance["mc_token_ids"] = len(instance["input_ids"]) - 1

        return instance, sequence
    
    #label 부분 축소하기
    def _shrink_label_cands(self, label, candidates):
        shrunk_label_cands = candidates.copy()
        if label in shrunk_label_cands : 
          shrunk_label_cands.remove(label)
        shrunk_label_cands = random.sample(shrunk_label_cands, k = 1) # k = n_cadidates(2) -1
        shrunk_label_cands.append(label)
        random.shuffle(shrunk_label_cands)
        return shrunk_label_cands

    #batch별로 불러오기
    def collate_fn(self, batch):
        input_ids = [ids for ins in batch for ids in ins["input_ids"]]
        token_type_ids = [ids for ins in batch for ids in ins["token_type_ids"]]
        mc_token_ids = [id for ins in batch for id in ins["mc_token_ids"]]
        label_idx = [ins["label_idx"] for ins in batch]

        data_info = {
            "dialog_ids": [ins["dialog_id"] for ins in batch],
            "candidate_keys": [ins["candidate_keys"] for ins in batch]
        }

        batch_size = len(batch)
        n_candidates = len(batch[0]["input_ids"])
        input_ids = torch.tensor(
            pad_ids(input_ids, self.pad)
        ).view(batch_size, n_candidates, -1)
        
        token_type_ids = torch.tensor(
            pad_ids(token_type_ids, self.pad)
        ).view(batch_size, n_candidates, -1)

        lm_labels = torch.full_like(input_ids, -100)
        mc_token_ids = torch.tensor(mc_token_ids).view(batch_size, n_candidates)
        label_idx = torch.tensor(label_idx)

        return input_ids, token_type_ids, mc_token_ids, lm_labels, label_idx, data_info
        #return input_ids, token_type_ids, lm_labels, label_idx, data_info

# %%
def set_seed():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

# %%
from transformers import (
    get_linear_schedule_with_warmup,
    #PreTrainedModel,
    PreTrainedTokenizer,
    BertPreTrainedModel,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
)
from transformers import modeling_roberta
from transformers import XLNetPreTrainedModel as PreTrainedModel
from transformers.modeling_utils import SequenceSummary

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss

from tqdm import tqdm, trange
from typing import Tuple, Dict
from itertools import repeat
from collections import OrderedDict

try:
  from torch.utils.tensorboard import SummaryWriter
except ImportError:
  from tensorboardX import SummaryWriter

ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = getattr(modeling_roberta, 'ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP', None)


# %%
def write_selection_preds(dataset_walker, output_file, data_infos, sorted_pred_ids, topk=5):
    # Flatten the data_infos
    data_infos = [
        {
            "dialog_id": info["dialog_ids"][i],
            "candidate_keys": info["candidate_keys"][i]
        }
        for info in data_infos
        for i in range(len(info["dialog_ids"]))
    ]

    labels = [label for log, label in dataset_walker]
    new_labels = [{"target": False}] * len(dataset_walker)
    
    # Update the dialogs with selected knowledge
    for info, sorted_pred_id in zip(data_infos, sorted_pred_ids):
        dialog_id = info["dialog_id"]
        candidate_keys = info["candidate_keys"]

        snippets = []
        for pred_id in sorted_pred_id[:topk]: 
            selected_cand = candidate_keys[pred_id]
            domain, entity_id, doc_id = selected_cand.split("__")
            snippet = {
                "domain": domain,
                "entity_id": "*" if entity_id == "*" else int(entity_id),
                "doc_id": int(doc_id)
            }
            snippets.append(snippet)
        
        new_label = {"target": True, "knowledge": snippets}
        label = labels[dialog_id]
        if label is None:
            label = new_label
        else:
            label = label.copy()
            if "response_tokenized" in label:
                label.pop("response_tokenized")
            label.update(new_label)

        new_labels[dialog_id] = label

    if os.path.dirname(output_file) and not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    with open(output_file, "w") as jsonfile:
        #logger.info("Writing predictions to {}".format(output_file))
        json.dump(new_labels, jsonfile, indent=2)

    print(new_labels)

# %%
device = "cuda"

def run_batch_selection_train(model, batch):
    batch = tuple(input_tensor.to(device) for input_tensor in batch if isinstance(input_tensor, torch.Tensor))
    input_ids, token_type_ids, mc_token_ids, lm_labels, mc_labels = batch
    model_outputs = model(
        input_ids=input_ids, token_type_ids=token_type_ids,
        mc_token_ids=mc_token_ids, mc_labels=mc_labels
    )
    mc_loss = model_outputs[0]
    lm_logits, mc_logits = model_outputs[1], model_outputs[2]
    return mc_loss, lm_logits, mc_logits, mc_labels


def run_batch_selection_eval(model, batch):
    candidates_per_forward = 16
    batch = tuple(input_tensor.to(device) for input_tensor in batch if isinstance(input_tensor, torch.Tensor))
    input_ids, token_type_ids, mc_token_ids, _, mc_labels = batch
    all_mc_logits = []
    for index in range(0, input_ids.size(1), candidates_per_forward):
        model_outputs = model(
            input_ids=input_ids[0, index:index+candidates_per_forward].unsqueeze(1),
            token_type_ids=token_type_ids[0, index:index+candidates_per_forward].unsqueeze(1),
            mc_token_ids=mc_token_ids[0, index:index+candidates_per_forward].unsqueeze(1)
        )
        mc_logits = model_outputs[1]
        all_mc_logits.append(mc_logits.detach())
    all_mc_logits = torch.cat(all_mc_logits, dim=0).unsqueeze(0)
    return torch.tensor(0.0), torch.tensor([]), all_mc_logits, mc_labels


# %%
class DoubleHeadsModel(PreTrainedModel):
  def __init__(self, config):
    super().__init__(config)
    config.num_labels = 1
    config.summary_activation = None
    config.summary_type = 'cls_index'
    config.summary_proj_to_labels = True

    self.transformer = AutoModel.from_config(config)
    self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    self.multiple_choice_head = SequenceSummary(config)

    self.init_weights()

  def get_output_embeddings(self):
    return self.lm_head

  def forward(
      self,
      input_ids=None,
      past=None,
      attention_mask=None,
      token_type_ids=None,
      position_ids=None,
      head_mask=None,
      inputs_embeds=None,
      mc_token_ids=None,
      labels=None,
      mc_labels=None,
      use_cache=None,
      output_attentions=None,
      output_hidden_states=None,
      **kwargs
  ):
    if "lm_labels" in kwargs:
      labels = kwargs.pop("lm_labels")
    assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

# TODO: find a more simple way to deal with the shape [114-122]
    transformer_outputs = self.transformer(
        input_ids.view((-1, input_ids.shape[2])),
        attention_mask=attention_mask,
        token_type_ids=token_type_ids.view((-1, token_type_ids.shape[2])),
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
    )

    hidden_states = transformer_outputs[0].view((input_ids.shape[0], input_ids.shape[1], input_ids.shape[2], -1))

    lm_logits = self.lm_head(hidden_states)
    mc_logits = self.multiple_choice_head(hidden_states, mc_token_ids).squeeze(-1)

    outputs = (lm_logits, mc_logits) + transformer_outputs[1:]
    if mc_labels is not None:
      loss_fct = CrossEntropyLoss()
      #loss_fct = BCELoss()
      loss = loss_fct(mc_logits.view(-1, mc_logits.size(-1)), mc_labels.view(-1))
      outputs = (loss,) + outputs

    if labels is not None:
      shift_logits = lm_logits[..., :-1, :].contiguous()
      shift_labels = labels[..., 1:].contiguous()
      loss_fct = CrossEntropyLoss()
      loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
      outputs = (loss,) + outputs

    return outputs  # (lm loss), (mc loss), lm logits, mc logits, presents, (all hidden_states), (attentions)

# %%
#global_step = 0
__DEBUG__ = False

def train(train_dataset, eval_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, run_batch_fn_train, run_batch_fn_eval) -> Tuple[int, float]:
    log_dir = os.path.join("runs", "")
    tb_writer = SummaryWriter(log_dir)
    output_dir = log_dir

    train_batch_size = 4
    num_train_epochs = 5
    gradient_accumulation_steps = 1

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size = train_batch_size,
        collate_fn = train_dataset.collate_fn
    )

    t_total = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs

    optimizer = AdamW(model.parameters(), lr = 6.25e-5, eps = 1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps = 0, num_training_steps=t_total
    )

    # Train!
    global_step = 0
    model.zero_grad()
    train_iterator = trange(0, int(num_train_epochs), desc="Epoch")
    set_seed()  # for reproducibility

    for _ in train_iterator:
      local_steps = 0
      tr_loss = 0.0
      epoch_iterator = tqdm(train_dataloader, desc = "Iteration")

      for step, batch in enumerate(epoch_iterator) :
          model.train()
          #loss, _, loss_2, loss_3 = run_batch_fn_train(model, batch)
          #loss, multi_loss = regain_loss(loss, loss_2, loss_3, train_epoch, percentage = global_step / t_total)
          loss, *_ = run_batch_fn_train(model, batch)

          #if gradient_accumulation_steps > 1 :
          #  loss = loss / gradient_accumulation_steps
          loss.backward()
          tr_loss += loss.item()

          if (step + 1) % gradient_accumulation_steps == 0:
              torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
              optimizer.step()
              scheduler.step()
              optimizer.zero_grad()
              global_step += 1
              local_steps += 1
              epoch_iterator.set_postfix(Loss = tr_loss / local_steps)

          if __DEBUG__ and step >= 20 :
              break
          
          #if (step + 1) % gradient_accumulation_steps == 0 :
          #display_infos = OrderedDict([("loss", tr_loss / local_steps)])
          #write_infos = [('loss/total', tr_loss / local_steps)]
          
            
            #tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
            #any(map(tb_writer.add_scalar, *zip(*map(
            #    lambda x, y: (*x, y), write_infos, repeat(global_step)
            #))))

      #train_eval_one_epoch(eval_dataset, model, run_batch_fn_eval, tb_writer, tokenizer)
      results = evaluate(eval_dataset, model, tokenizer, run_batch_fn_eval, desc=str(global_step))
      for key, value in results.items():
        tb_writer.add_scalar("eval/{}".format(key), value, global_step)
      tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
      tb_writer.add_scalar("loss", tr_loss/local_steps, global_step)

      # Save model checkpoint
      output_dir = os.path.join(output_dir, "checkpoint-{}".format(global_step))
      os.makedirs(output_dir, exist_ok=True)
      model_to_save = (
        model.module if hasattr(model, "module") else model
      )  # Take care of distributed/parallel training

      model_to_save.save_pretrained(output_dir)
      tokenizer.save_pretrained(output_dir)
    tb_writer.close()

    return global_step, tr_loss / local_steps


def evaluate(eval_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, run_batch_fn, desc="") -> Dict:
    eval_batch_size = 1
    eval_output_dir = "eval_result"
    os.makedirs(eval_output_dir, exist_ok = True)
    output_file  = "output_result"

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler = eval_sampler,
        batch_size = eval_batch_size,
        collate_fn = eval_dataset.collate_fn
    )

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    data_infos = []
    all_preds = []
    all_labels = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        with torch.no_grad():
            loss, lm_logits, mc_logits, mc_labels = run_batch_fn(model, batch)
            data_infos.append(batch[-1])
            eval_loss += loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps

    all_labels = np.array(all_labels).reshape(-1)
    all_pred_ids = np.array([np.argmax(logits) for logits in all_preds])
    accuracy = np.sum(all_pred_ids == all_labels) / len(all_labels)
    result = {"loss": eval_loss, "accuracy": accuracy}
    
    if output_file :
      sorted_pred_ids = [np.argsort(logits.squeeze())[::-1] for logits in all_preds]
      write_selection_preds(eval_dataset.dataset_walker, output_file, data_infos, sorted_pred_ids, topk=5)
    
    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    with open(output_eval_file, "a") as writer :
      writer.write("***** Eval results %s *****\n" %desc)
      for key in sorted(result.keys()):
        writer.write("%s = %s \n" % (key, str(result[key])))

    return result

# %%
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# %%
#model = AutoModel.from_pretrained("amberoad/bert-multilingual-passage-reranking-msmarco")
#tokenzier = AutoTokenizer.from_pretrained("amberoad/bert-multilingual-passage-reranking-msmarco")
#model = AutoModel.from_pretrained("castorini/monobert-large-msmarco")
#tokenzier = AutoTokenizer.from_pretrained("castorini/monobert-large-msmarco")
dataset_class, model_class, run_batch_fn_train, run_batch_fn_eval = KnowledgeSelectionDataset, DoubleHeadsModel, run_batch_selection_train, run_batch_selection_eval

#model_name 정의필요
model_name = "xlnet-base-cased" #xlnet-large-cased
model_type = "xlnet"
config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens(SPECIAL_TOKENS)

model = model_class.from_pretrained(model_name, config = config)
model.resize_token_embeddings(len(tokenizer))
model.to(device)

dataroot = "data"
train_dataset = dataset_class(tokenizer, split_type="train")
val_dataset = dataset_class(tokenizer, split_type="val")

# %%
dataroot = 'data_eval'
test_dataset = dataset_class(tokenizer, split_type="test")

# %%
global_step, tr_loss = train(train_dataset, val_dataset, model, tokenizer, run_batch_fn_train, run_batch_fn_eval)
print("global_step : %s, average loss = %s", global_step, tr_loss)

# %%
result = {}
result = evaluate(test_dataset, model, tokenizer, run_batch_fn_eval, desc= "")

# %%
class Metric :
  def __init__(self) :
    self.reset()

  def reset(self) :
    self._selection_mrr5 = 0.0
    self._selection_r1 = 0.0
    self._selection_r5 = 0.0

  def _match(self, ref_knowledge, pred_knowledge) :
    result = []
    for pred in pred_knowledge :
      matched = False
      for ref in ref_knowledge :
        if pred["domain"] == ref['domain'] and pred["entity_id"] == ref["entity_id"] and pred["doc_id"] == ref["doc_id"] :
          matched = True
      result.append(matched)
    return result

  def _reciprocal_rank(self, ref_knowledge, hyp_knowledge, k = 5):
    relevance = self._match(ref_knowledge, hyp_knowledge)[:k]

    if True in relevance :
      idx = relevance.index(True)
      result = 1.0 / (idx + 1)
    else :
      result = 0.0

    return result

  def _recall_at_k(self, ref_knowledge, hyp_knowledge, k=5):
    relevance = self._match(ref_knowledge, hyp_knowledge)[:k]

    if True in relevance :
      result = 1.0
    else :
      result = 0.0
    
    return result

  def update(self, ref_obj, hyp_obj) :
    if ref_obj['target'] is True :
      if hyp_obj['target'] is True :
        self._detection_tp += 1

        if 'knowledge' in hyp_obj :
          self._selection_mrr5 += self._reciprocal_rank(ref_obj['knowledge'], hyp_obj['knowledge'], 5)
          self._selection_r1 += self._recall_at_k(ref_obj['knowledge'], hyp_obj['knowledge'], 1)
          self._selection_r1 += self._recall_at_k(ref_obj['knowledge'], hyp_obj['knowledge'], 5)
      else :
        self._detection_fn += 1
    else :
      if hyp_obj['target'] is True :
        self._detection_fp += 1
      else :
        self._detection_tn += 1

  def _compute(self, score_sum):
    if self._detection_tp + self._detection_fp > 0.0:
      score_p = score_sum / (self._detection_tp + self._detection_fp)
    else:
      score_p = 0.0

    if self._detection_tp + self._detection_fn > 0.0:
      score_r = score_sum / (self._detection_tp + self._detection_fn)
    else:
      score_r = 0.0

    if score_p + score_r > 0.0:
      score_f = 2 * score_p * score_r / (score_p + score_r)
    else:
      score_f = 0.0

    return (score_p, score_r, score_f)


  def scores(self) :
    selection_mrr5_p, selection_mrr5_r, selection_mrr5_f = self._compute(self._selection_mrr5)
    selection_r1_p, selection_r1_r, selection_r1_f = self._compute(self._selction_r1)
    selection_r5_p, selection_r5_r, selection_r5_f = self._compute(self._selction_r5)

    scores = {
        'selection' : {
            'mrr@5' : selection_mrr5_f,
            'r@1' : selection_r1_f,
            'r@5' : selection_r5_f,
        }
    }
    return scores

    

# %%
outputfile = "eval_results.txt"
with open(outputfile, 'r') as f:
  output = json.load(f)
  
data = DatasetWalker(dataroot = 'data', dataset = 'test', labels = True)
metric = Metric()

for dialog, pred in zip(data, output):
  ref = dialog['label']
  metric.update(ref, pred)

scorefile = "scores"
scores = metric.scores()
with open(scorefile, 'w') as out :
  json.dump(scores, out, indent = 2)


# %%
