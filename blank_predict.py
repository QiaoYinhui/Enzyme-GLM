import torch
from torch.utils.data import Dataset,DataLoader
from src.modeling_glm import GLMConfig,GLMModel,GLMForSingleTokenCloze
from tokenizers import Tokenizer
import numpy as np
import csv
from sklearn.metrics import f1_score
from src.process_data import build_input_from_ids,recreate_smi
import random

CLS_ID = 108
SEP_ID = 110
STO_ID = 0
MASK_ID = 109
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
    _batch_ids, _batch_position_ids, _batch_atention_masks, _batch_logit_mask,_labels = [], [], [], [],[]

    for sample in batch:
        label = sample['label']
        _labels.append(label)
        ids = sample['ids']
        _batch_ids.append(ids)
        attention_mask = sample['attention_mask']
        _batch_atention_masks.append(attention_mask)
        position_id = sample['position_ids']
        _batch_position_ids.append(np.array(position_id))
        loss__mask = [5,6]
        _batch_logit_mask.append(loss__mask)

    return torch.tensor(_batch_ids), torch.tensor(_batch_position_ids), torch.tensor(_batch_atention_masks),torch.tensor(_batch_logit_mask), _labels

class dataset(Dataset):
    # The root_dir represents the directory for the data, while label_file denotes the file containing labels.
    def __init__(self,data_file,max_length,tokenizer):
        self.data_file = data_file
        self.max_length = max_length
        self.tokenizer = tokenizer

        with open(self.data_file) as f:
            datas = csv.reader(f)
            self.datas = list(datas)[1:]

    def __getitem__(self,idx):
        # idx is index for every sample
        example = self.datas[idx]
        enzyme = example[1]
        smiles = example[3]
        label = example[4]
        _, re_smi = recreate_smi(smiles)

        enzyme_ids = self.tokenizer.encode(enzyme).ids
        smiles_ids = self.tokenizer.encode(re_smi).ids

        merge_ids = smiles_ids + [SEP_ID] + enzyme_ids + [MASK_ID]

        data = build_input_from_ids(merge_ids,[],self.max_length,add_sep=False)

        ids, types, paddings, position_ids, sep, target_ids, loss_masks, attention_mask = data

        sample = {'ids':ids,'position_ids':position_ids,'attention_mask':attention_mask,'loss_mask':loss_masks,'label':label}

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
    configuration.output_predict = True
    model = GLMModel(configuration)

    Enzyme_GLM_BLANK = GLMForSingleTokenCloze(model)

    # load checkpoints
    Enzyme_GLM_BLANK.load_state_dict(torch.load('checkpoint/enzyme-glm-blank.pt'))
    Enzyme_GLM_BLANK.eval()
    Enzyme_GLM_BLANK.to(device)

    test_dataset = dataset('data/enzyme_substrate_pair_test.csv',MAXK_LENGTH,tokenizer=tokenizer)
    dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False,collate_fn=my_collate)

    true_labels = []
    pre_labels = []
    probabilitys = []

    with torch.no_grad():
        for index,batch in enumerate(dataloader):

            print(index)
            ids,position_id,attention_mask,target_ids,label = process_batch(batch)
            ids = ids.to(device)
            position_id = position_id.to(device)
            attention_mask = attention_mask.to(device)
            target_ids = target_ids.to(device)
            output= Enzyme_GLM_BLANK(ids,position_id,attention_mask,target_ids)
            output = output[0]


            for item in label:
                true_labels.append(int(item))

            for probability in output:
                probabilitys.append(probability)
                pre_labels.append(1 if probability[0] >= probability[1] else 0)


    acc = acc_of_baniry(pre_labels,true_labels)
    F1 = f1_score(true_labels,pre_labels)

    print(acc,F1)

    write_lines = []

    for index,item in enumerate(true_labels):
        write_lines.append([index,item,pre_labels[index],probabilitys[index]])

    with open('result/blank_result.csv', 'w', newline='') as f:
        write = csv.writer(f)
        write.writerows(write_lines)