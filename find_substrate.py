import torch
import random
import numpy as np
from torch.utils.data import Dataset,DataLoader
from src.modeling_glm import GLMConfig,GLMModel,GLMForSequenceClassification
from tokenizers import Tokenizer
import csv
from src.process_data import build_input_from_ids

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

def get_all_substrate_and_smi_id_seq():
    SNI_file = "../data/source/all_SMI_plusplus_normalized_TFYG.csv"
    all_substrate = set()
    smi_id_seq = {}
    with open(SNI_file) as g:
        SNI = csv.reader(g)
        for index,line in enumerate(SNI):
            number = line[0]
            all_substrate.add(number)
            smi = line[3]
            if number not in smi_id_seq:
                smi_id_seq[number] = smi
    return all_substrate,smi_id_seq

class find_substrate_dataset(torch.utils.data.Dataset):
    def __init__(self,input_path,tokenizer,smi_id_seq,max_lehgth=1024):
        self.examples = []
        self.input_path = input_path
        self.tokenizer = tokenizer
        self.smi_id_seq = smi_id_seq
        self.max_length = max_lehgth
        with open(self.input_path) as f:
            enzymes = csv.reader(f)
            for index,line in enumerate(enzymes):
                if index > 0:
                    enzyme_id = line[0]
                    sequence = line[1]
                    for k,v in smi_id_seq.items():
                        example = {'enzyme_id':enzyme_id,'molecule_id':k,'pro_seq':sequence,'smi_seq':v}
                        self.examples.append(example)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        # idx is index for sample
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

    input_file= 'data/find_substrate_example.csv'

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

    _,smi_id_seq = get_all_substrate_and_smi_id_seq()

    find_dataset = find_substrate_dataset(input_file,tokenizer,smi_id_seq,1024)
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
        if enzyme_id not in results:
            results[enzyme_id] = {}
            results[enzyme_id][molecule_id] = score
        else:
            results[enzyme_id][molecule_id] = score

    result_dit_paixv = {}
    for k, v in results.items():
        sort_dit = sorted(v.items(), key=lambda x: x[1], reverse=True)
        sort_dit_ = sort_dit[:top_k]
        sort_dit_10_dit = {}
        for item in sort_dit_:
            sort_dit_10_dit[item[0]] = item[1]
        result_dit_paixv[k] = sort_dit_10_dit

    print(result_dit_paixv)

