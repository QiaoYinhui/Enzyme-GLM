import json

import torch
from torch.utils.data import Dataset,DataLoader
from src.modeling_glm import GLMConfig,GLMModel
from tokenizers import Tokenizer
import numpy as np
import csv
from src.process_data import build_input_from_ids,recreate_smi
import random

CLS_ID = 108
SEP_ID = 110
STO_ID = 0
MAXK_LENGTH = 1024
seed = 1234

def set_random_seed(seed):
    """Set random seed for reproducability."""

    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.deterministic = True

def acc_of_baniry(pre_labels, labels):

    count = 0
    for pre_label, label in zip(pre_labels, labels):
        count += pre_label == label
    return count * 100.0 / len(labels)


def process_batch(batch):
    _batch_ids, _batch_position_ids, _batch_atention_masks,_names = [], [], [], []

    for sample in batch:
        name = sample['name']
        _names.append(name)
        ids = sample['ids']
        _batch_ids.append(ids)
        attention_mask = sample['attention_mask']
        _batch_atention_masks.append(attention_mask)
        position_id = sample['position_ids']
        _batch_position_ids.append(np.array(position_id))

    return torch.tensor(np.array(_batch_ids)), torch.tensor(np.array(_batch_position_ids)), torch.tensor(np.array(_batch_atention_masks)), _names

class dataset(Dataset):
    def __init__(self,data_file,max_length,tokenizer):
        self.data_file = data_file
        self.max_length = max_length
        self.tokenizer = tokenizer

        with open(self.data_file) as f:
            datas = csv.reader(f)
            self.datas = list(datas)[1:]

    def __getitem__(self,idx):
        # idx is index for a sample
        example = self.datas[idx]
        id = example[0]
        type = example[1]
        sequence = example[2]
        if type == 'protein':
            sequence_ids = self.tokenizer.encode(sequence).ids
            data = build_input_from_ids(sequence_ids, None, self.max_length)

            ids, types, paddings, position_ids, sep, target_ids, loss_masks, attention_mask = data

            sample = {'ids': ids, 'position_ids': position_ids, 'attention_mask': attention_mask, 'name': id}

            return sample
        else:
            _, sequence = recreate_smi(sequence)
            sequence_ids = self.tokenizer.encode(sequence).ids
            data = build_input_from_ids(sequence_ids, None, self.max_length)

            ids, types, paddings, position_ids, sep, target_ids, loss_masks, attention_mask = data

            sample = {'ids': ids, 'position_ids': position_ids, 'attention_mask': attention_mask, 'name': id}

            return sample


    def __len__(self):
        return len(self.datas)

def my_collate(batch):
    new_batch = [{key: value for key, value in sample.items() if key != 'uid'} for sample in batch]
    return new_batch

if __name__ == '__main__':

    set_random_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initialize tokenizer and saved_model
    tokenizer = Tokenizer.from_file("checkpoint/tokenzier/tokenizer.json")
    configuration = GLMConfig()
    Bilingual_model = GLMModel(configuration)

    # load checkpoints
    Bilingual_model.load_state_dict(torch.load('checkpoint/304M_p10s10t10.pt'))
    Bilingual_model.eval()
    Bilingual_model.to(device)

    test_dataset = dataset('data/get_embedding.csv',MAXK_LENGTH,tokenizer=tokenizer)
    dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False,collate_fn=my_collate)

    imbedding_dit = {}
    with torch.no_grad():
        for index,batch in enumerate(dataloader):

            ids,position_id,attention_mask,names = process_batch(batch)
            print(names,ids.tolist())
            ids = ids.to(device)
            position_id = position_id.to(device)
            attention_mask = attention_mask.to(device)
            output= Bilingual_model(ids,position_id,attention_mask) #B*1024*1024
            output = output[0][:,0,:]

            for index,embedding in enumerate(output):
                imbedding_dit[names[index]] = embedding.tolist()

    with open('result/embedding.json', 'w') as f:
        json.dump(imbedding_dit,f, indent=4)