import os
import json
import random
import string
import unidecode
import torch
import torch.utils.data
import numpy as np
from tasks.data_utils import InputExample
from tqdm import tqdm
from utils import print_rank_0
from data_utils.corpora import punctuation_standardization


def gigaword_detokenize(string, is_target=False):
    _tok_dict = {"(": "-lrb-", ")": "-rrb-",
                 "[": "-lsb-", "]": "-rsb-",
                 "{": "-lcb-", "}": "-rcb-",
                 '&': '&amp;', '<': '&lt;', '>': '&gt;'}
    string = string.replace('UNK', '[UNK]')
    string = string.replace('<unk>', '[UNK]')
    for key, value in _tok_dict.items():
        string = string.replace(value, key)
    # string = string.replace("''", "\"")
    # string = string.replace("``", "\"")
    # string = string.replace("`", "'")
    # string = string.replace(" n't", "n't")
    # string = string.replace(" 's", "'s")
    # string = string.replace(" 'd", "'d")
    # string = string.replace(" 'll", "'ll")
    return string


def cnndm_detokenize(string, is_target=False):
    _tok_dict = {"(": "-LRB-", ")": "-RRB-",
                 "[": "-LSB-", "]": "-RSB-",
                 "{": "-LCB-", "}": "-RCB-"}
    if not is_target:
        string = string.replace("<S_SEP>", "")
    else:
        string = string.replace("<S_SEP>", "[SEP]")
    for key, value in _tok_dict.items():
        string = string.replace(value, key)
    string = string.replace("''", "\"")
    string = string.replace("``", "\"")
    string = string.replace("`", "'")
    string = string.replace(" n't", "n't")
    string = string.replace(" 's", "'s")
    string = string.replace(" 'd", "'d")
    string = string.replace(" 'll", "'ll")
    return string


def blanklm_detokenize(string, is_target=False):
    string = string.replace("_UNK", "[UNK]")
    string = string.replace("<blank>", "[MASK]")
    return string


class SummmaryProcessor:
    def __init__(self, task, data_dir, tokenizer):
        self.task = task
        self.data_dir = data_dir
        self.tokenizer = tokenizer

    def create_examples(self, split):
        if split == "train":
            filename = "train"
        elif split == "dev":
            filename = "val"
        elif split == "test":
            filename = "test"
        else:
            raise NotImplementedError(split)
        print_rank_0(f"Creating {self.task}-{split} dataset from {self.data_dir}")
        if self.task == "gigaword":
            detokenizer = gigaword_detokenize
        elif self.task == "cnn_dm":
            detokenizer = cnndm_detokenize
        else:
            detokenizer = None
        source_texts, target_texts = [], []
        with open(os.path.join(self.data_dir, f"{filename}.source"), encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                line = punctuation_standardization(line)
                line = detokenizer(line) if detokenizer else line
                source_texts.append(line)
        with open(os.path.join(self.data_dir, f"{filename}.target"), encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                line = punctuation_standardization(line)
                line = detokenizer(line, is_target=True) if detokenizer else line
                target_texts.append(line)
        assert len(source_texts) == len(target_texts)
        example_list = []
        for idx, (source_text, target_text) in enumerate(zip(source_texts, target_texts)):
            if (idx + 1) % 20000 == 0:
                print_rank_0(f"Complete {idx + 1} examples")
            guid = "%s-%s" % (split, idx)
            meta = {"ref": self.tokenizer.DecodeIds(self.tokenizer.EncodeAsIds(target_text).tokenization)}
            example = InputExample(guid=guid, text_a=source_text, text_b=target_text, meta=meta)
            if idx < 10:
                print_rank_0((source_text.encode('utf-8'), target_text.encode('utf-8'), meta["ref"].encode('utf-8')))
            example_list.append(example)
        return example_list

class CMRCProcessor:
    def __init__(self, data_dir, tokenizer):
        self.data_dir = data_dir
        self.tokenizer = tokenizer

    def create_examples(self, split):
        if split == "train":
            filename = "train.json"
        elif split == "dev":
            filename = "dev.json"
        elif split == "test":
            filename = "test.json"
        else:
            raise NotImplementedError(split)
        print_rank_0(f"Creating CMRC-{split} dataset from {self.data_dir}")
        example_list = []
        idx = 0
        with open(os.path.join(self.data_dir, filename), encoding='utf-8') as file:
            dataset = json.load(file)
            for article in dataset['data']:
                for paragraph in article['paragraphs']:
                    context = paragraph['context']
                    for qa in paragraph['qas']:
                        question = qa["question"]
                        answers = {answer['text'] for answer in qa["answers"]} if split != 'test' else {"FAKE_ANSWER"}
                        for answer in answers:
                            guid = "%s-%s" % (split, idx)
                            meta = {
                                "answer": answer,
                                "question": question,
                                "ref": self.tokenizer.DecodeIds(self.tokenizer.EncodeAsIds(answer).tokenization)}
                            example = InputExample(guid=guid, text_a=context, meta=meta)
                            if idx < 10:
                                print_rank_0(
                                    (context.encode('utf-8'), answer.encode('utf-8'), meta["ref"].encode('utf-8')))
                            example_list.append(example)
                            idx += 1
        print_rank_0(f"Creating {len(example_list)} examples for {split}")
        return example_list


class SQuADGenerationProcessor:
    def __init__(self, data_dir, tokenizer):
        self.data_dir = data_dir
        self.tokenizer = tokenizer

    def create_examples(self, split):
        if split == "train":
            filename = "train.json"
        elif split == "dev":
            filename = "dev.json"
        elif split == "test":
            filename = "test.json"
        else:
            raise NotImplementedError(split)
        print_rank_0(f"Creating CMRC-{split} dataset from {self.data_dir}")
        example_list = []
        idx = 0
        with open(os.path.join(self.data_dir, filename), encoding='utf-8') as file:
            dataset = json.load(file)
            for article in dataset['data']:
                for paragraph in article['paragraphs']:
                    context = paragraph['context']
                    for qa in paragraph['qas']:
                        question = qa["question"]
                        answers = {answer['text'] for answer in qa["answers"]} if split != 'test' else {"FAKE_ANSWER"}
                        for answer in answers:
                            guid = "%s-%s" % (split, idx)
                            meta = {
                                "answer": answer,
                                "question": question,
                                "ref": self.tokenizer.DecodeIds(self.tokenizer.EncodeAsIds(answer).tokenization)}
                            example = InputExample(guid=guid, text_a=context, meta=meta)
                            if idx < 10:
                                print_rank_0(
                                    (context.encode('utf-8'), answer.encode('utf-8'), meta["ref"].encode('utf-8')))
                            example_list.append(example)
                            idx += 1
        print_rank_0(f"Creating {len(example_list)} examples for {split}")
        return example_list


class SQuADProcessor:
    def __init__(self, data_dir, tokenizer):
        self.data_dir = data_dir
        self.tokenizer = tokenizer

    def create_examples(self, split):
        if split == "train":
            filename = "train.json"
        elif split == "dev":
            filename = "dev.json"
        elif split == "test":
            filename = "test.json"
        else:
            raise NotImplementedError(split)
        print_rank_0(f"Creating SQuAD-{split} dataset from {self.data_dir}")
        example_list = []
        idx = 0
        with open(os.path.join(self.data_dir, filename), encoding='utf-8') as file:
            dataset = json.load(file)
            for paragraphs in dataset:
                for paragraph in paragraphs['paragraphs']:
                    context = paragraph['context']
                    for qa in paragraph['qas']:
                        question = qa["question"]
                        answers = {answer["text"] for answer in qa["answers"]}
                        answer_starts = {answer["text"]: answer["answer_start"] for answer in qa["answers"]}
                        for answer in answers:
                            guid = "%s-%s" % (split, idx)
                            meta = {
                                "answer_start": answer_starts[answer],
                                "answer": answer,
                                "question": question,
                                "ref": self.tokenizer.DecodeIds(self.tokenizer.EncodeAsIds(question).tokenization)}
                            example = InputExample(guid=guid, text_a=context, meta=meta)
                            if idx < 10:
                                print_rank_0(
                                    (context.encode('utf-8'), answer.encode('utf-8'), meta["ref"].encode('utf-8')))
                            example_list.append(example)
                            idx += 1
        print_rank_0(f"Creating {len(example_list)} examples for {split}")
        return example_list


def generate_token_to_char_map(tokens, raw_text, tokenizer):
    # Use heuristics to construct the token to char mapping for BertTokenizer

    def _is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    def _normalize(s):
        # if s in tokenizer.command_token_map:
        #     return s
        return unidecode.unidecode(s.lower())

    def _compare(s1, s2):
        return _normalize(s1) == _normalize(s2)

    tokens = [tokenizer.IdToToken(token) for token in tokens]
    text = raw_text
    token_to_char = []
    char_id = 0
    mismatch = 0
    for i, token in enumerate(tokens):
        while char_id + 1 < len(text) and _is_whitespace(text[char_id]):
            char_id += 1
        assert char_id < len(text)
        if token.startswith("##"):
            token = token[2:].strip()
        if _compare(token, text[char_id:char_id + len(token)]):
            token_to_char.append((char_id, char_id + len(token)))
            char_id += len(token)
        else:
            if token != '[UNK]':
                mismatch += 1
            token_len = 1 if token == '[UNK]' else len(token)
            token_to_char.append((char_id, char_id + token_len))
            char_id += token_len
            if i + 1 < len(tokens):
                pos = text[char_id:].find(tokens[i + 1])
                if pos != -1 and pos < 20:
                    char_id += pos

    return token_to_char


class SQuADProcessor:
    def __init__(self, data_dir, tokenizer, max_src_length, args):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_src_length = max_src_length
        self.task = args.task
        self.args = args
        import transformers
        tokenizer_model_type = self.args.tokenizer_model_type
        if tokenizer_model_type == 'roberta':
            tokenizer_model_type = 'roberta-large'
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.transformer_tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_model_type)

    def create_examples(self, split):

        if split == "train":
            filename = "train-v1.1.json" if self.task == "squad_v1" else "train-v2.0.json"
        elif split == "dev":
            filename = "dev-v1.1.json" if self.task == "squad_v1" else "dev-v2.0.json"
        elif split == "test":
            filename = "dev-v1.1.json" if self.task == "squad_v1" else "dev-v2.0.json"
        else:
            raise NotImplementedError(split)
        print_rank_0(f"Creating SQuAD-{split} dataset from {self.data_dir}")
        example_list = []
        idx = 0
        total_qas = 0
        total_na = 0
        with open(os.path.join(self.data_dir, filename), encoding='utf-8') as file:
            dataset = json.load(file)['data']
            for paragraphs in dataset:
                for paragraph in paragraphs['paragraphs']:
                    context = paragraph['context']
                    context_tokens = self.tokenizer.EncodeAsIds(context).tokenization
                    transformer_encode = self.transformer_tokenizer(context,
                                                                    return_offsets_mapping=True,
                                                                    add_special_tokens=False,
                                                                    verbose=False)
                    assert transformer_encode['input_ids'] == context_tokens
                    token_to_char = transformer_encode['offset_mapping']
                    # if self.tokenizer_type == 'BertWordPieceTokenizer':
                    #     token_to_char = generate_token_to_char_map(context_tokens, context, self.tokenizer)
                    # else:
                    #     token_to_char = None
                    for qa in paragraph['qas']:
                        total_qas += 1
                        question = qa["question"]
                        question_tokens = self.tokenizer.EncodeAsIds(" " + question).tokenization
                        answers = [answer["text"] for answer in qa["answers"]]
                        if len(qa['answers']) == 0:
                            answers = ['N/A']
                        for start in range(0, len(context_tokens), self.max_src_length // 2):
                            length = self.max_src_length - 3 - len(question_tokens)
                            tokens = context_tokens[start:start + length]
                            new_context = self.tokenizer.DecodeIds(tokens)
                            answer = answers[0]
                            answer_tokens_text = self.tokenizer.DecodeIds(self.tokenizer.EncodeAsIds(answer).tokenization)
                            if answer_tokens_text and answer_tokens_text in new_context:
                                # new_context = new_context.replace(answer_tokens_text, answer)
                                pass
                            else:
                                answer = 'N/A'
                            if self.task == 'squad_v1' and answer == 'N/A':
                                continue
                            guid = "%s-%s" % (split, idx)
                            meta = {
                                "context": context,
                                "context_tokens": context_tokens,
                                "token_to_char": token_to_char,
                                "answer": answer,
                                "answers": answers,
                                "question": question,
                                "ref": answer
                            }
                            example = InputExample(guid=guid, text_a=new_context, meta=meta, idx=qa['id'])
                            example_list.append(example)
                            idx += 1
                            total_na += (answer == 'N/A')
                            if len(tokens) < length:
                                break
        print_rank_0(f"Creating {len(example_list)} / {total_qas} examples for {split}. {total_na} N/A")
        return example_list


class XSumProcessor:
    def __init__(self, data_dir, tokenizer):
        self.data_dir = data_dir
        self.tokenizer = tokenizer

    def create_examples(self, split):
        if split == "train":
            key = "train"
        elif split == "dev":
            key = "validation"
        elif split == "test":
            key = "test"
        else:
            raise NotImplementedError(split)
        print_rank_0(f"Creating XSUM-{split} dataset from {self.data_dir}")
        with open(os.path.join(self.data_dir, "XSum-TRAINING-DEV-TEST-SPLIT-90-5-5.json")) as file:
            id_list = json.load(file)
        id_list = id_list[key]
        source_texts, target_texts = [], []
        for i, idx in enumerate(id_list):
            with open(os.path.join(self.data_dir, f"{idx}.summary")) as file:
                key, sentences = None, []
                source_text, target_text = None, None
                for line in file:
                    line = line.strip()
                    if line.startswith("[SN]"):
                        if key is not None:
                            if key == "RESTBODY":
                                source_text = " ".join(sentences)
                            elif key == "FIRST-SENTENCE":
                                target_text = " ".join(sentences)
                        key = line[4:-4]
                        sentences = []
                    elif line:
                        sentences.append(line)
                if key is not None:
                    if key == "RESTBODY":
                        source_text = " ".join(sentences)
                    elif key == "FIRST-SENTENCE":
                        target_text = " ".join(sentences)
                source_texts.append(source_text)
                target_texts.append(target_text)
                if (i + 1) % 1000 == 0:
                    print_rank_0(f"Complete {i + 1} examples")
        assert len(source_texts) == len(target_texts)
        example_list = []
        for idx, (source_text, target_text) in enumerate(zip(source_texts, target_texts)):
            if (idx + 1) % 20000 == 0:
                print_rank_0(f"Complete {idx + 1} examples")
            guid = "%s-%s" % (split, idx)
            meta = {"ref": self.tokenizer.DecodeIds(self.tokenizer.EncodeAsIds(target_text).tokenization)}
            example = InputExample(guid=guid, text_a=source_text, text_b=target_text, meta=meta)
            if idx < 10:
                print_rank_0((source_text.encode('utf-8'), target_text.encode('utf-8'), meta["ref"].encode('utf-8')))
            example_list.append(example)
        return example_list


class Seq2SeqDataset(torch.utils.data.Dataset):
    def __init__(self, args, split, tokenizer):
        self.args = args
        self.task, self.data_dir = args.task.lower(), args.data_dir
        self.max_src_length, self.max_tgt_length = args.src_seq_length, args.tgt_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.dataset_name = split
        if self.task in ["gigaword", "cnn_dm", "cnn_dm_original"]:
            self.processor = SummmaryProcessor(self.task, self.data_dir, tokenizer)
        elif self.task in ["xsum"]:
            self.processor = XSumProcessor(self.data_dir, tokenizer)
        elif self.task in ["squad_generation"]:
            self.processor = SQuADGenerationProcessor(self.data_dir, tokenizer)
        elif self.task in ["squad", "squad_v1"]:
            self.processor = SQuADProcessor(self.data_dir, tokenizer, self.max_src_length, args)
        elif self.task in ['cmrc']:
            self.processor = CMRCProcessor(self.data_dir, tokenizer)
        else:
            raise NotImplementedError(self.task)
        example_list = self.processor.create_examples(split)
        self.example_list = example_list
        self.examples = {example.guid: example for example in example_list}

        print_rank_0(f"Return {len(self.examples)} {split} examples")

    def __len__(self):
        return len(self.example_list)

    def __getitem__(self, idx):
        example = self.example_list[idx]
        cls_id = self.tokenizer.get_command('ENC').Id
        mask_token = 'sMASK' if self.args.task_mask else 'MASK'
        mask_id = self.tokenizer.get_command(mask_token).Id
        pad_id = self.tokenizer.get_command('pad').Id
        sop_id = self.tokenizer.get_command('sop').Id
        eop_id = self.tokenizer.get_command('eop').Id
        if self.task in ["gigaword", "cnn_dm", "cnn_dm_original", "xsum"]:
            source_text, target_text = example.text_a, example.text_b
            source_tokens = self.tokenizer.EncodeAsIds(" " + source_text).tokenization
            prompt = [cls_id, mask_id] + self.tokenizer.EncodeAsIds(" Content:").tokenization
            if len(source_tokens) > self.max_src_length - len(prompt):
                source_tokens = source_tokens[:self.max_src_length - len(prompt)]
            source_tokens = prompt + source_tokens
        elif self.task == "squad_generation":
            source_text = example.text_a
            target_text, answer = example.meta["question"], example.meta["answer"]
            source_tokens = self.tokenizer.EncodeAsIds(source_text.rstrip() + " Question:").tokenization
            answer_tokens = self.tokenizer.EncodeAsIds(" Answer: " + answer).tokenization
            if len(source_tokens) > self.max_src_length - len(answer_tokens) - 2:
                max_src_length = self.max_src_length - len(answer_tokens) - 2
                answer_pattern = self.tokenizer.EncodeAsIds(" " + answer).tokenization

                def sub_finder(mylist, pattern):
                    matches = []
                    for i in range(len(mylist)):
                        if mylist[i] == pattern[0] and mylist[i:i + len(pattern)] == pattern:
                            matches.append(i)
                    return matches

                answer_indices = sub_finder(source_tokens, answer_pattern)
                if len(answer_indices) == 0:
                    print(f"Answer {answer} not exists in the source text")
                    source_tokens = source_tokens[:max_src_length]
                else:
                    start_index = max(answer_indices[0] - max_src_length // 2, 0)
                    source_tokens = source_tokens[start_index: start_index + max_src_length]
            source_tokens = [cls_id] + source_tokens + [mask_id] + answer_tokens
        elif self.task in ["squad", "squad_v1"]:
            source_text = example.text_a
            target_text = example.meta["answer"].strip()
            question = example.meta["question"].strip()
            source_tokens = self.tokenizer.EncodeAsIds(" " + source_text.rstrip()).tokenization
            question_tokens = self.tokenizer.EncodeAsIds(" " + question).tokenization
            period_id = self.tokenizer.TokenToId('.')
            max_src_length = self.max_src_length - len(question_tokens) - 3
            if max_src_length <= 0:
                print(question)
            assert max_src_length > 0
            source_tokens = [cls_id] + question_tokens + [mask_id, period_id] + source_tokens[:max_src_length]
        elif self.task in ["cmrc"]:
            mask_id = self.tokenizer.get_command('MASK').Id
            source_text = example.text_a
            target_text = example.meta["answer"].strip()
            question = example.meta["question"].strip()
            source_tokens = self.tokenizer.EncodeAsIds(source_text.rstrip()).tokenization
            question_tokens = self.tokenizer.EncodeAsIds("问题：" + question + "答案：").tokenization
            max_src_length = self.max_src_length - len(question_tokens) - 2
            if max_src_length <= 0:
                print(question)
                question_tokens = question_tokens[self.max_src_length // 4]
            source_tokens = [cls_id] + question_tokens + [mask_id] + source_tokens[:max_src_length]
        else:
            raise NotImplementedError
        if len(source_tokens) < self.max_src_length:
            source_tokens = source_tokens + [pad_id] * (self.max_src_length - len(source_tokens))
        sep = len(source_tokens)
        position_ids = list(range(len(source_tokens)))
        block_position_ids = [0] * len(source_tokens)
        mask_pos = source_tokens.index(mask_id)
        if self.split == 'train':
            target_tokens = self.tokenizer.EncodeAsIds(" " + target_text).tokenization
            target_tokens = target_tokens + [eop_id]
            if len(target_tokens) > self.max_tgt_length:
                target_tokens = target_tokens[:self.max_tgt_length]
                target_truncated = True
            loss_mask = [1] * len(target_tokens)
            if len(target_tokens) < self.max_tgt_length:
                loss_mask += [0] * (self.max_tgt_length - len(target_tokens))
                target_tokens += [pad_id] * (self.max_tgt_length - len(target_tokens))
            tokens = source_tokens + [sop_id] + target_tokens[:-1]
            loss_mask = [0] * len(source_tokens) + loss_mask
            target_ids = [0] * len(source_tokens) + target_tokens
            position_ids += [mask_pos] * len(target_tokens)
            if self.args.no_block_position:
                block_position_ids += [1] * len(target_tokens)
            else:
                block_position_ids += list(range(1, len(target_tokens) + 1))
            position_ids = [position_ids, block_position_ids]
            sample = {'text': np.array(tokens, dtype=np.int64), 'target': np.array(target_ids, dtype=np.int64),
                      'attention_mask': np.array(sep, dtype=np.int64),
                      'loss_mask': np.array(loss_mask, dtype=np.int64),
                      "position_id": np.array(position_ids, dtype=np.int64), "uid": example.guid}
        else:
            tokens = source_tokens + [sop_id]
            position_ids = position_ids + [mask_pos]
            block_position_ids = block_position_ids + [1]
            position_ids = [position_ids, block_position_ids]
            sample = {'text': np.array(tokens, dtype=np.int64), 'attention_mask': np.array(sep, dtype=np.int64),
                      "position_id": np.array(position_ids, dtype=np.int64), "uid": example.guid}
        return sample


class ExtractionDataset(torch.utils.data.Dataset):
    def __init__(self, args, split, tokenizer):
        self.args = args
        task, data_dir = args.task.lower(), args.data_dir
        self.max_src_length, self.max_tgt_length = args.src_seq_length, args.tgt_seq_length
        self.split = split
        self.tokenizer = tokenizer
        if split == "train":
            filename = "train"
        elif split == "dev":
            filename = "valid"
        elif split == "test":
            filename = "test"
        else:
            raise NotImplementedError(split)
        print_rank_0(f"Creating {task}-{split} dataset from {data_dir}")
        self.dataset_name = split
        source_texts, target_texts = [], []
        with open(os.path.join(data_dir, f"{filename}.source"),
                  encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                source_texts.append(line)
        with open(os.path.join(data_dir, f"{filename}.target"), encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                target_texts.append(line)
        self.examples, self.example_list = {}, []
        for idx, (source_text, target_text) in enumerate(zip(source_texts, target_texts)):
            if (idx + 1) % 20000 == 0:
                print_rank_0(f"Complete {idx + 1} examples")
            guid = "%s-%s" % (split, idx)
            meta = {"ref": target_text}
            example = InputExample(guid=guid, text_a=source_text, text_b=target_text, meta=meta)
            self.examples[guid] = example
            self.example_list.append(example)
        print_rank_0(f"Return {len(self.examples)} {split} examples")

    def __len__(self):
        return len(self.example_list)

    def __getitem__(self, idx):
        example = self.example_list[idx]
        source_text, target_text = example.text_a, example.text_b
        mask_token = 'MASK'
        mask_id = self.tokenizer.get_command(mask_token).Id
        sop_id = self.tokenizer.get_command('sop').Id
        eop_id = self.tokenizer.get_command('eop').Id
        pad_id = self.tokenizer.get_command('pad').Id

        def pad_to(text, max_len, pad_id):
            if len(text) > max_len:
                text = text[:max_len]
            else:
                text = text + [pad_id] * (max_len - len(text))
            return text

        source_tokens = self.tokenizer.EncodeAsIds(source_text).tokenization
        masked_tgt = target_text.split("|")
        source_tokens = pad_to(source_tokens, self.max_src_length, pad_id)
        sep = len(source_tokens)
        position_ids = list(range(len(source_tokens)))
        block_position_ids = [0] * len(source_tokens)
        if self.split == 'train':
            mask_positions = [i for i, x in enumerate(source_tokens) if x == mask_id]
            assert len(mask_positions) <= len(masked_tgt)
            tokens = source_tokens
            target_ids = [0] * len(source_tokens)
            loss_mask = [0] * len(source_tokens)
            for i, mask_pos in enumerate(mask_positions):
                tgt_text = masked_tgt[i]
                tgt_tokens = self.tokenizer.EncodeAsIds(" " + tgt_text).tokenization
                tokens += [sop_id] + tgt_tokens
                target_ids += tgt_tokens + [eop_id]
                loss_mask += [1] * (len(tgt_tokens) + 1)
                position_ids += [mask_pos] * (len(tgt_tokens) + 1)
                block_position_ids += [i + 1 for i in range(len(tgt_tokens) + 1)]
            tokens = pad_to(tokens, self.max_src_length + self.max_tgt_length, pad_id)
            target_ids = pad_to(target_ids, self.max_src_length + self.max_tgt_length, pad_id)
            loss_mask = pad_to(loss_mask, self.max_src_length + self.max_tgt_length, 0)
            position_ids = pad_to(position_ids, self.max_src_length + self.max_tgt_length, 0)
            block_position_ids = pad_to(block_position_ids, self.max_src_length + self.max_tgt_length, 0)
            position_ids = [position_ids, block_position_ids]
            sample = {'text': np.array(tokens, dtype=np.int64), 'target': np.array(target_ids, dtype=np.int64),
                      'attention_mask': np.array(sep, dtype=np.int64),
                      'loss_mask': np.array(loss_mask, dtype=np.int64),
                      "position_id": np.array(position_ids, dtype=np.int64), "uid": example.guid}
        else:
            tokens = source_tokens + [sop_id]
            mask_pos = source_tokens.index(mask_id)
            position_ids = position_ids + [mask_pos]
            block_position_ids = block_position_ids + [1]
            position_ids = [position_ids, block_position_ids]
            sample = {'text': np.array(tokens, dtype=np.int64), 'attention_mask': np.array(sep, dtype=np.int64),
                      "position_id": np.array(position_ids, dtype=np.int64), "uid": example.guid}
        return sample


class BlankLMDataset(torch.utils.data.Dataset):
    def __init__(self, args, split, tokenizer):
        self.args = args
        task, data_dir = args.task.lower(), args.data_dir
        self.max_src_length, self.max_tgt_length = args.src_seq_length, args.tgt_seq_length
        self.split = split
        assert args.tokenizer_type == "BertWordPieceTokenizer"
        self.tokenizer = tokenizer
        if split == "train":
            filename = "train"
        elif split == "dev":
            filename = "valid"
        elif split == "test":
            filename = "test"
        else:
            raise NotImplementedError(split)
        print_rank_0(f"Creating {task}-{split} dataset from {data_dir}")
        self.dataset_name = split
        detokenizer = blanklm_detokenize
        source_texts, target_texts = [], []
        with open(os.path.join(data_dir, f"{filename}.txt"), encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                line = detokenizer(line) if detokenizer else line
                target_texts.append(line)
        if split == 'test':
            with open(os.path.join(data_dir, f"blank/test.maskratio{args.blank_maskratio:.1f}.blank"),
                      encoding='utf-8') as file:
                for line in file:
                    line = line.strip()
                    line = detokenizer(line) if detokenizer else line
                    source_texts.append(line)
        else:
            source_texts = target_texts
        self.examples, self.example_list = {}, []
        for idx, (source_text, target_text) in enumerate(zip(source_texts, target_texts)):
            # if idx > 10000:
            #     break
            if (idx + 1) % 20000 == 0:
                print_rank_0(f"Complete {idx + 1} examples")
            guid = "%s-%s" % (split, idx)
            meta = {"ref": target_text}
            example = InputExample(guid=guid, text_a=source_text, text_b=target_text, meta=meta)
            self.examples[guid] = example
            self.example_list.append(example)
        print_rank_0(f"Return {len(self.examples)} {split} examples")
        self.random = random.Random(args.seed)

    def __len__(self):
        return len(self.example_list)

    def __getitem__(self, idx):
        example = self.example_list[idx]
        source_text, target_text = example.text_a, example.text_b
        mask_token = 'gMASK' if self.args.task_mask else 'MASK'
        mask_id = self.tokenizer.get_command(mask_token).Id
        sop_id = self.tokenizer.get_command('sop').Id
        eop_id = self.tokenizer.get_command('eop').Id
        pad_id = self.tokenizer.get_command('pad').Id
        if self.split in ['train', 'dev']:
            masked_src, masked_tgt = self.mask_text(source_text)
            source_text = masked_src

        def pad_to(text, max_len, pad_id):
            if len(text) > max_len:
                text = text[:max_len]
            else:
                text = text + [pad_id] * (max_len - len(text))
            return text

        source_tokens = self.tokenizer.EncodeAsIds(" " + source_text).tokenization
        source_tokens = pad_to(source_tokens, self.max_src_length, pad_id)
        sep = len(source_tokens)
        position_ids = list(range(len(source_tokens)))
        block_position_ids = [0] * len(source_tokens)
        if self.split in ['train', 'dev']:
            mask_positions = [i for i, x in enumerate(source_tokens) if x == mask_id]
            assert len(mask_positions) <= len(masked_tgt)
            tokens = source_tokens
            target_ids = [0] * len(source_tokens)
            loss_mask = [0] * len(source_tokens)
            for i, mask_pos in enumerate(mask_positions):
                tgt_text = masked_tgt[i]
                tgt_tokens = self.tokenizer.EncodeAsIds(" " + tgt_text).tokenization
                tokens += [sop_id] + tgt_tokens
                target_ids += tgt_tokens + [eop_id]
                loss_mask += [1] * (len(tgt_tokens) + 1)
                position_ids += [mask_pos] * (len(tgt_tokens) + 1)
                block_position_ids += [i + 1 for i in range(len(tgt_tokens) + 1)]
            max_length = self.max_src_length + int(self.max_src_length * self.args.blank_maskratio)
            tokens = pad_to(tokens, max_length, pad_id)
            target_ids = pad_to(target_ids, max_length, pad_id)
            loss_mask = pad_to(loss_mask, max_length, 0)
            position_ids = pad_to(position_ids, max_length, 0)
            block_position_ids = pad_to(block_position_ids, max_length, 0)
            position_ids = [position_ids, block_position_ids]
            sample = {'text': np.array(tokens, dtype=np.int64), 'target': np.array(target_ids, dtype=np.int64),
                      'attention_mask': np.array(sep, dtype=np.int64),
                      'loss_mask': np.array(loss_mask, dtype=np.int64),
                      "position_id": np.array(position_ids, dtype=np.int64), "uid": example.guid}
        else:
            tokens = source_tokens + [sop_id]
            mask_pos = source_tokens.index(mask_id)
            position_ids = position_ids + [mask_pos]
            block_position_ids = block_position_ids + [1]
            position_ids = [position_ids, block_position_ids]
            sample = {'text': np.array(tokens, dtype=np.int64), 'attention_mask': np.array(sep, dtype=np.int64),
                      "position_id": np.array(position_ids, dtype=np.int64), "uid": example.guid}
        return sample

    def mask_text(self, text):
        tokens = text.split()
        mask_ratio = self.args.blank_maskratio
        n = len(tokens)
        indices = sorted(self.random.sample(range(n), int(n * mask_ratio)))
        masked_src, masked_tgt = "", []
        for i, idx in enumerate(indices):
            if i == 0 or idx != indices[i - 1] + 1:
                masked_tgt.append("")
            masked_tgt[-1] += " " + tokens[idx]
            tokens[idx] = "[MASK]"
        for i, token in enumerate(tokens):
            if i != 0 and token == "[MASK]" and tokens[i - 1] == "[MASK]":
                continue
            masked_src += " " + token
        return masked_src, masked_tgt


class CustomizationDataset(torch.utils.data.Dataset):
    def __init__(self, args, split, tokenizer):
        task, data_dir = args.task.lower(), args.data_dir
        self.max_src_length, self.max_tgt_length = args.src_seq_length, args.tgt_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.mask_pad_token = args.mask_pad_token
        self.no_block_position = args.no_block_position
        self.task_mask = args.task_mask
        if split == "train":
            filename = "train"
        elif split == "dev":
            filename = "val"
        elif split == "test":
            filename = "test"
        else:
            raise NotImplementedError(split)
        print_rank_0(f"Creating {task}-{split} dataset from {data_dir}")
        self.dataset_name = split
        source_texts, target_texts = [], []
        with open(os.path.join(data_dir, f"{filename}.source"),
                  encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                source_texts.append(line)
        with open(os.path.join(data_dir, f"{filename}.target"), encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                target_texts.append(line)
        self.examples, self.example_list = {}, []
        for idx, (source_text, target_text) in enumerate(zip(source_texts, target_texts)):
            if (idx + 1) % 20000 == 0:
                print_rank_0(f"Complete {idx + 1} examples")
            guid = "%s-%s" % (split, idx)
            meta = {"ref": target_text}
            example = InputExample(guid=guid, text_a=source_text, text_b=target_text, meta=meta)
            self.examples[guid] = example
            self.example_list.append(example)
        print_rank_0(f"Return {len(self.examples)} {split} examples")

    def __len__(self):
        return len(self.example_list)

    def __getitem__(self, idx):
        example = self.example_list[idx]
        cls_id = self.tokenizer.get_command('ENC').Id
        #mask_token = 'sMASK' if self.task_mask else 'MASK'
        mask_token = 'gMASK' if self.task_mask else 'MASK'
        mask_id = self.tokenizer.get_command(mask_token).Id
        eos_id = self.tokenizer.get_command('eos').Id
        pad_id = self.tokenizer.get_command('pad').Id
        sop_id = self.tokenizer.get_command('sop').Id
        eop_id = self.tokenizer.get_command('eop').Id
        amount_id = 7
        source_text, target_text = example.text_a, example.text_b
        source_tokens = self.tokenizer.EncodeAsIds(source_text).tokenization
        if len(source_tokens) + 3 > self.max_src_length:
            source_tokens = source_tokens[-(self.max_src_length - 3):]
        #source_tokens = [cls_id] + source_tokens + [mask_id, eos_id]
        source_tokens = source_tokens + [amount_id ,mask_id, eos_id]
        context_length = len(source_tokens)
        if len(source_tokens) < self.max_src_length:
            source_tokens = source_tokens + [pad_id] * (self.max_src_length - len(source_tokens))
        sep = len(source_tokens)
        position_ids = list(range(len(source_tokens)))
        block_position_ids = [0] * len(source_tokens)
        mask_pos = source_tokens.index(mask_id)
        if self.split == 'train':
            #target_tokens = self.tokenizer.EncodeAsIds(" " + target_text).tokenization
            target_tokens = self.tokenizer.EncodeAsIds(target_text).tokenization
            target_tokens = target_tokens + [eop_id]
            if len(target_tokens) > self.max_tgt_length:
                target_tokens = target_tokens[:self.max_tgt_length]
            loss_mask = [1] * len(target_tokens)
            if len(target_tokens) < self.max_tgt_length:
                loss_mask += [0] * (self.max_tgt_length - len(target_tokens))
                target_tokens += [pad_id] * (self.max_tgt_length - len(target_tokens))
            if self.mask_pad_token:
                attention_mask = np.zeros((self.max_src_length + self.max_tgt_length,
                                           self.max_src_length + self.max_tgt_length), dtype=np.int64)
                attention_mask[:, :context_length] = 1
                attention_mask[self.max_src_length:, self.max_src_length:] = np.tril(
                    np.ones((self.max_tgt_length, self.max_tgt_length), dtype=np.int64))
                attention_mask = attention_mask[None, :, :]
            else:
                attention_mask = np.array(sep, dtype=np.int64)
            tokens = source_tokens + [sop_id] + target_tokens[:-1]
            loss_mask = [0] * len(source_tokens) + loss_mask
            target_ids = [0] * len(source_tokens) + target_tokens
            position_ids += [mask_pos] * len(target_tokens)
            if self.no_block_position:
                block_position_ids += [1] * len(target_tokens)
            else:
                block_position_ids += list(range(1, len(target_tokens) + 1))
            position_ids = [position_ids, block_position_ids]
            sample = {'text': np.array(tokens, dtype=np.int64), 'target': np.array(target_ids, dtype=np.int64),
                      'attention_mask': attention_mask,
                      'loss_mask': np.array(loss_mask, dtype=np.int64),
                      "position_id": np.array(position_ids, dtype=np.int64), "uid": example.guid}
        else:
            tokens = source_tokens + [sop_id]
            position_ids = position_ids + [mask_pos]
            block_position_ids = block_position_ids + [1]
            position_ids = [position_ids, block_position_ids]
            if self.mask_pad_token:
                attention_mask = np.zeros((self.max_src_length + 1, self.max_src_length + 1), dtype=np.int64)
                attention_mask[:, :context_length] = 1
                attention_mask[-1, -1] = 1
                attention_mask = attention_mask[None, :, :]
            else:
                attention_mask = np.array(sep, dtype=np.int64)
            sample = {'text': np.array(tokens, dtype=np.int64), 'attention_mask': attention_mask,
                      "position_id": np.array(position_ids, dtype=np.int64), "uid": example.guid}
        return sample
