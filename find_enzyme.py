import json

import torch
import random
import numpy as np
from torch.utils.data import Dataset,DataLoader
from src.modeling_glm import GLMConfig,GLMModel,GLMForSequenceClassification
from tokenizers import Tokenizer
import csv
from src.process_data import build_input_from_ids, recreate_smi

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

def get_enzyme_id_seq():
    with open('../data/all_proteins_sequence.json') as f:
        enzyme_id_seq = json.load(f)

    return enzyme_id_seq

class find_enzyme_dataset(torch.utils.data.Dataset):
    def __init__(self,input_path,tokenizer,enzyme_id_seq,max_lehgth=1024):
        self.examples = []
        self.input_path = input_path
        self.tokenizer = tokenizer
        self.enzyme_id_seq = enzyme_id_seq
        self.max_length = max_lehgth
        with open(self.input_path) as f:
            mols = csv.reader(f)
            for index,line in enumerate(mols):
                if index > 0:
                    mol_id = line[0]
                    smiles = line[1]
                    _,re_smiles = recreate_smi(smiles)
                    for k,v in self.enzyme_id_seq.items():
                        example = {'enzyme_id':k,'molecule_id':mol_id,'pro_seq':v,'smi_seq':re_smiles}
                        self.examples.append(example)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        # idx is index for a sample
        example = self.examples[index]
        enzyme_id = example['enzyme_id']
        molecule_id = example['molecule_id']
        pro_seq = example['pro_seq']
        smi_seq = example['smi_seq']


        enzyme_ids = self.tokenizer.encode(pro_seq).ids
        smiles_ids = self.tokenizer.encode(smi_seq).ids

        data = build_input_from_ids(smiles_ids, enzyme_ids, self.max_length)

        ids, types, paddings, position_ids, sep, target_ids, loss_masks, attention_mask = data

        sample = {'ids': ids, 'position_ids': position_ids, 'attention_mask': attention_mask, 'enzyme_id': enzyme_id, 'molecule_id':molecule_id}

        return sample

def my_collate(batch):
    new_batch = [{key: value for key, value in sample.items() if key != 'uid'} for sample in batch]
    return new_batch

def process_batch(batch):
    _batch_ids, _batch_position_ids, _batch_atention_masks,enzyme_ids,molecule_ids = [], [], [], [], []

    for sample in batch:
        enzyme_id = sample['enzyme_id']
        molecule_id = sample['molecule_id']
        enzyme_ids.append(enzyme_id)
        molecule_ids.append(molecule_id)
        ids = sample['ids']
        _batch_ids.append(ids)
        attention_mask = sample['attention_mask']
        _batch_atention_masks.append(attention_mask)
        position_id = sample['position_ids']
        _batch_position_ids.append(np.array(position_id))

    return torch.tensor(np.array(_batch_ids)), torch.tensor(np.array(_batch_position_ids)), \
        torch.tensor(np.array(_batch_atention_masks)), enzyme_ids, molecule_ids

if __name__ == '__main__':

    input_file= 'data/find_enzyme_example.csv'

    set_random_seed(1234)

    top_k = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initialize tokenizer and  saved_model
    tokenizer = Tokenizer.from_file("checkpoint/tokenzier/tokenizer.json")
    configuration = GLMConfig()
    model = GLMModel(configuration)
    Enzyme_GLM_CLS = GLMForSequenceClassification(model, hidden_size=1024, hidden_dropout=0.1, pool_token='cls',
                                                  num_class=1)

    # load checkpoints
    Enzyme_GLM_CLS.load_state_dict(torch.load('checkpoint/enzyme-glm-cls.pt'))
    Enzyme_GLM_CLS.eval()
    Enzyme_GLM_CLS.to(device)

    enzyme_id_seq = get_enzyme_id_seq()

    find_dataset = find_enzyme_dataset(input_file,tokenizer,enzyme_id_seq,1024)
    dataloader = DataLoader(find_dataset, batch_size=32, shuffle=False, collate_fn=my_collate)

    org_resulets = []

    with torch.no_grad():
        for index,batch in enumerate(dataloader):

            print(index)
            ids,position_id,attention_mask,enzyme_ids,molecule_ids = process_batch(batch)
            ids = ids.to(device)
            position_id = position_id.to(device)
            attention_mask = attention_mask.to(device)
            output= Enzyme_GLM_CLS(ids,position_id,attention_mask)
            output = output[0]

            for index,out in enumerate(output):
                org_resulets.append([enzyme_ids[index],molecule_ids[index],out.tolist()])
    results = {}
    for example in org_resulets:
        enzyme_id = example[0]
        molecule_id = example[1]
        score = example[2]
        if molecule_id not in results:
            results[molecule_id] = {}
            results[molecule_id][enzyme_id] = score
        else:
            results[molecule_id][enzyme_id] = score

    result_dit_paixv = {}
    for k, v in results.items():
        sort_dit = sorted(v.items(), key=lambda x: x[1], reverse=True)
        sort_dit_ = sort_dit[:top_k]
        sort_dit_10_dit = {}
        for item in sort_dit_:
            sort_dit_10_dit[item[0]] = item[1]
        result_dit_paixv[k] = sort_dit_10_dit

    print(result_dit_paixv)

