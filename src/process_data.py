def build_input_from_ids(text_a_ids, text_b_ids,  max_seq_length, add_cls=True,
                         add_sep=True,  add_eos=True):

    eos_id = 0
    cls_id = 108
    sep_id = 110
    ids = []
    types = []
    paddings = []
    # CLS
    if add_cls:
        ids.append(cls_id)
        types.append(0)
        paddings.append(1)
    # A
    len_text_a = len(text_a_ids)
    ids.extend(text_a_ids)
    types.extend([0] * len_text_a)
    paddings.extend([1] * len_text_a)
    # B
    if text_b_ids is not None:
        # SEP
        if add_sep:
            ids.append(sep_id)
            types.append(0)
            paddings.append(1)
        len_text_b = len(text_b_ids)
        ids.extend(text_b_ids)
        types.extend([1] * len_text_b)
        paddings.extend([1] * len_text_b)
    eos_length = 1 if add_eos else 0
    # Cap the size.
    if len(ids) >= max_seq_length - eos_length:
        max_seq_length_m1 = max_seq_length - 1
        ids = ids[0:max_seq_length_m1]
        types = types[0:max_seq_length_m1]
        paddings = paddings[0:max_seq_length_m1]
    end_type = 0 if text_b_ids is None else 1
    if add_eos:
        ids.append(eos_id)
        types.append(end_type)
        paddings.append(1)
    sep = len(ids)
    target_ids = [0] * len(ids)
    loss_masks = [0] * len(ids)
    attention_mask = len(ids)
    position_ids = list(range(len(ids)))
    block_position_ids = [0] * len(ids)
      # Padding.
    padding_length = max_seq_length - len(ids)
    if padding_length > 0:
        ids.extend([eos_id] * padding_length)
        types.extend([eos_id] * padding_length)
        paddings.extend([0] * padding_length)
        position_ids.extend([0] * padding_length)
        block_position_ids.extend([0] * padding_length)
        target_ids.extend([0] * padding_length)
        loss_masks.extend([0] * padding_length)
    position_ids = [position_ids, block_position_ids]
    return ids, types, paddings, position_ids, sep, target_ids, loss_masks, attention_mask

def recreate_smi(smi):
    encoder_dit = {'A':'~','B':'!','C':'{','H':'}','I':'<','K':'>','M':':','N':'$','O':'&','P':'?','R':'_','S':',','T':'"','W':';'} #不行，有些符号smiles本身就有
    # smiles_regex_pattern = r'se|Na|Ag|Mo|Zn|Cu|Se|Mu|Mn|Si|Mg|Ca|Fe|As|Al|Cl|Br|[*#%\)\(\+\-1032547698:=@CBFIHONPSKW\[\]icosndp]|/|\\'
    # smi_list = re.findall(smiles_regex_pattern,smi)
    recreate_smi_list = []
    recreate_smi_str = ''
    for car in smi:
        if car in encoder_dit:
            car = encoder_dit[car]
            recreate_smi_list.append(car)
            recreate_smi_str += car
        else:
            recreate_smi_list.append(car)
            recreate_smi_str += car
    return recreate_smi_list,recreate_smi_str