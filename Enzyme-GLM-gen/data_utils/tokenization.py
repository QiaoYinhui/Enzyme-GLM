# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for using and training tokenizers (char, wordpiece, sentencepiece)"""
from collections import namedtuple
import random
import os
import csv
import torch
import itertools

import nltk
from nltk import tokenize as nltk_tokenize
import sentencepiece as spm

from .wordpiece import BertTokenizer, PRETRAINED_VOCAB_ARCHIVE_MAP

from .tokenization_gpt2 import GPT2Tokenizer
from . import sp_tokenizer
from utils import print_rank_0
import regex as re
from tokenizers import Tokenizer as Tokenizer_rengong
from tokenizers.models import BPE


def make_tokenizer(tokenizer_type, corpus, model_path=None, vocab_size=None, model_type=None, pad_token=0,
                   character_coverage=1.0, command_tokens=None, type_tokens=None, fix_command_token=False, **kwargs):
    """
    Helper function to instantiate a tokenizer given common combinations of options.
    帮助程序函数，用于在给定常见选项组合的情况下实例化分词器
    """
    tokenizer_class = tokenizer_type
    if isinstance(tokenizer_class, str): #判断tokenizer_class的数据类型是不是字符串
        tokenizer_class = eval(tokenizer_class)  #eval它接受一个字符串作为参数，将其解释为Python代码，并返回该代码的执行结果
    if tokenizer_class is BertWordPieceTokenizer:
        return BertWordPieceTokenizer(model_type, **kwargs)
    elif tokenizer_class is GPT2BPETokenizer:
        if model_type is None:
            model_type = 'gpt2'
            #model_type = r"/mnt/c/Users/lenovo/Desktop/qyh_vm/GLM_Linux/tokenizer/output/protein/BPE_tokenizer/BPE_tokenizer_protein_all_50256size.json"
        return GPT2BPETokenizer(model_type, **kwargs)
    elif tokenizer_class is ChineseSPTokenizer:
        return ChineseSPTokenizer(fix_command_token=fix_command_token, **kwargs)
    text_tokenizer = tokenizer_class(corpus=corpus, vocab_size=vocab_size, model_path=model_path, model_type=model_type,
                                     pad_token=pad_token, character_coverage=character_coverage)
    return Tokenizer(text_tokenizer, command_tokens, type_tokens)


class Tokenization(object):
    """
    Tokenization object to hold tokenization, (processed text),and original
    text. Can hold tokenization as Ids or tokens.

    It also holds command tokens (pad, unk, etc.) for the tokenization.
    This allows functions to pad/operate on tokenizations without having
    access to the full tokenizer, just the tokenization.

    Several standard array operations are implemented (insert, append, extend).
    """

    def __init__(self, tokenization, text=None, original_text=None, command_tokens=None, asIds=True):
        self.tokenization = tokenization
        self.text = text
        if self.text is None:
            self.text = self.tokenization
        self.original_text = original_text
        if self.original_text is None:
            self.original_text = self.text
        self.command_tokens = command_tokens
        self.asIds = asIds
        self.parse_command_tokens()

    def set_command_tokens(self, command_tokens):
        self.command_tokens = command_tokens
        return self.parse_command_tokens()

    def parse_command_tokens(self):
        if self.command_tokens is None:
            return
        for command_token in self.command_tokens:
            if self.asIds:
                setattr(self, command_token.name, command_token.Id)
            else:
                setattr(self, command_token.name, command_token.token)

    def __getitem__(self, index):
        return self.tokenization[index]

    def __len__(self):
        return len(self.tokenization)

    def insert(self, idx, other):
        if isinstance(other, (CommandToken, TypeToken)):
            self.tokenization.insert(idx, other.Id)
            if idx == 0:
                self.text = other.token + self.text
                self.original_text = other.token + self.original_text
            elif idx == len(self.tokenization) - 1:
                self.text += other.token
                self.original_text += other.token
        elif isinstance(other, Tokenization):
            self.tokenization = self.tokenization[:idx] + other.tokenization + self.tokenization[idx:]
        else:
            self.tokenization = self.tokenization[:idx] + other.tokenization + self.tokenization[idx:]

    def append(self, other):
        if isinstance(other, (CommandToken, TypeToken)):
            self.tokenization.append(other.Id)
            self.text += other.token
            self.original_text += other.token
        elif isinstance(other, Tokenization):
            self.tokenization.extend(other.tokenization)
            self.text += other.text
            self.original_text += other.original_text
        else:
            self.tokenization.append(other)
        return self

    def extend(self, other):
        if isinstance(other, (CommandToken, TypeToken)):
            self.tokenization.append(other.Id)
            self.text += other.token
            self.original_text += other.token
        elif isinstance(other, list) and isinstance(other[0], (CommandToken, TypeToken)):
            self.tokenization.extend([o.Id for o in other])
            self.text += [o.token for o in other]
            self.original_text += [o.token for o in other]
        elif isinstance(other, Tokenization):
            self.tokenization.extend(other.tokenization)
            self.text += other.text
            self.original_text += other.original_text
        else:
            self.tokenization.extend(other)
        return self


"""define some default command tokens for the tokenizer to use"""
token_format = "<{0}>"

COMMAND_TUPLE = namedtuple('CommandToken', ('name', 'token', 'Id'))


def prep_command_tokens(tokenlist, token_format=token_format):
    return [CommandToken(tok[0], token_format.format(tok[0]), tok[1]) for tok in tokenlist]


class CommandToken(object):
    def __init__(self, name, token, Id, lstrip=False, rstrip=False):
        self.name = name
        self.token = token
        self.Id = Id
        self.lstrip = lstrip
        self.rstrip = rstrip

    def __str__(self):
        return str(COMMAND_TUPLE(self.name, self.token, self.Id))

    def __repr__(self):
        return str(COMMAND_TUPLE(self.name, self.token, self.Id))


DEFAULT_COMMAND_TOKENS = [
    ('pad', 0),
    ('eos', 1),
    ('bos', 2),
    ('unk', 3),
    ('sep', 4),
    ('L2R', 5),
    ('ENC', 6),
    ('MASK', 7),
]
DEFAULT_COMMAND_TOKENS = prep_command_tokens(DEFAULT_COMMAND_TOKENS)

"""define some default type tokens for bert training"""

TYPE_TUPLE = namedtuple('TypeToken', ('name', 'token', 'Id'))


def prep_type_tokens(tokenlist, token_format=token_format):
    return [TypeToken(tok[0], token_format.format(tok[0]), tok[1]) for tok in tokenlist]


class TypeToken(object):
    def __init__(self, name, token, Id):
        self.name = name
        self.token = token
        self.Id = Id

    def __str__(self):
        return str(TYPE_TUPLE(self.name, self.token, self.Id))


DEFAULT_TYPE_TOKENS = [
    ('function', 0),
    ('command', 1),
    ('str0', 2),
    ('str1', 3),
    ('str2', 4),
    ('embedding0', 5),
    ('embedding1', 6),
    ('embedding2', 7),
    ('arg0', 8),
    ('arg1', 9),
    ('arg2', 10),
]
DEFAULT_TYPE_TOKENS = prep_type_tokens(DEFAULT_TYPE_TOKENS)


class Tokenizer(object):
    """
    Tokenizer object that handles text tokenization, command tokens, and type tokens.

    Command tokens and text tokens are stored together in one mapping of size
    `len(text_tokenizer)+len(command_tokens)`. Command tokens are stored as first
    `len(command_tokens)` tokens. Token idx is stored at `idx+len(command_tokens)`.

    Token types are stored in a separate mapping of size `len(type_tokens)`.
    """

    def __init__(self, text_tokenizer, command_tokens=None, type_tokens=None):
        # set text tokenizer
        self.text_tokenizer = text_tokenizer
        if not hasattr(self, 'num_text_tokens'):
            # self.num_text_tokens = len(self.text_tokenizer)
            self.num_text_tokens = 50256 - 42

        # set command tokens
        if command_tokens is None:
            command_tokens = DEFAULT_COMMAND_TOKENS   #设置一些默认的token
        self._command_tokens = command_tokens
        self.command_name_map = {tok.name: tok for tok in self._command_tokens}
        self.command_token_map = {tok.token: tok for tok in self._command_tokens}
        self.command_id_map = {tok.Id: tok for tok in self._command_tokens}
        if not hasattr(self, 'num_command_tokens'):
            # self.num_command_tokens = len(self._command_tokens)
            self.num_command_tokens = 42
        if not hasattr(self, 'num_tokens'):
            self.num_tokens = self.num_command_tokens + self.num_text_tokens

        # set type tokens
        if type_tokens is None:
            type_tokens = DEFAULT_TYPE_TOKENS
        self.type_tokens = type_tokens
        self.type_name_map = {tok.name: tok for tok in self.type_tokens}
        self.type_token_map = {tok.token: tok for tok in self.type_tokens}
        self.type_id_map = {tok.Id: tok for tok in self.type_tokens}
        if not hasattr(self, 'num_type_tokens'):
            self.num_type_tokens = len(self.type_tokens)

        # parse tokens and vocabs from tokenizer,从分词器解析标记和词汇
        # self._tokens = list(self.command_token_map.keys()) + list(self.text_tokenizer.tokens)  #_tokens：command_token + text_token
        self._tokens = list(self.command_token_map.keys()) + list(self.text_tokenizer.get_vocab().keys())  # _tokens：command_token + text_token
        self._vocab = {t: Id for Id, t in self.command_id_map.items()}
        self._vocab.update({t: Id + self.num_command_tokens for t, Id in self.text_tokenizer.vocab.items()})

        self._text_tokens = list(self.text_tokenizer.tokens)
        self._text_token_vocab = {t: Id + self.num_command_tokens for t, Id in self.text_tokenizer.vocab.items()}

        self._command_token_tokens = list(self.command_token_map.keys())
        self._command_token_vocab = {t: Id for Id, t in self.command_id_map.items()}

        self._token_types = list(self.type_token_map.keys())
        self._token_type_vocab = {t: Id for Id, t in self.type_id_map.items()}

    def __call__(self, text, process_fn=None):
        """run preprocessing and encode text as Ids，为文本进行编码"""
        return self.EncodeAsIds(text, process_fn=process_fn)

    def __len__(self):
        """total number of tokens"""
        return self.num_tokens

    def get_command(self, name):
        """get command token corresponding to `name`"""
        return self.command_name_map[name]

    def get_type(self, name):
        """get type token corresponding to `name`"""
        return self.type_name_map[name]

    @property
    def tokens(self):
        """list (or iterable) of all tokens for tokenizer"""
        return self._tokens

    @property
    def vocab(self):
        """dictionary mapping tokens to ids for tokenizer"""
        return self._vocab

    @property
    def token_types(self):
        """list (or iterable) of all token types for tokenizer"""
        return self._token_types

    @property
    def token_type_vocab(self):
        """dictionary mapping token types to ids for tokenizer"""
        return self._token_type_vocab

    @property
    def command_tokens(self):
        """list (or iterable) of all command tokens for tokenizer"""
        return self._command_token_tokens

    @property
    def command_token_vocab(self):
        """dictionary mapping command tokens to ids for tokenizer"""
        return self._command_token_vocab

    @property
    def text_tokens(self):
        """list (or iterable) of text tokens for text tokenizer"""
        return self._text_tokens

    @property
    def text_token_vocab(self):
        """dictionary mapping text tokens to ids for text tokenizer"""
        return self._text_token_vocab

    def EncodeAsIds(self, text, process_fn=None):
        """
        encode text using text tokenizer and shift Id values for command tokens
        """
        processed_text = text
        if process_fn is not None:
            processed_text = process_fn(processed_text)

        def split_on_token(tok_extended: CommandToken, text):
            result = []
            tok = tok_extended.token
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
                # CommandToken can control whitespace stripping around them.
                # We use them for GPT2 and Roberta to have different behavior depending on the special token
                # Cf. https://github.com/huggingface/transformers/pull/2778
                # and https://github.com/huggingface/transformers/issues/3788
                # Strip white spaces on the right
                if tok_extended.rstrip and i > 0:
                    # A bit counter-intuitive but we strip the left of the string
                    # since tok_extended.rstrip means the special token is eating all white spaces on its right
                    sub_text = sub_text.lstrip()
                # Strip white spaces on the left
                if tok_extended.lstrip and i < len(split_text) - 1:
                    sub_text = sub_text.rstrip()  # Opposite here

                if i == 0 and not sub_text:
                    result.append(tok)
                elif i == len(split_text) - 1:
                    if sub_text:
                        result.append(sub_text)
                    else:
                        pass
                else:
                    if sub_text:
                        result.append(sub_text)
                    result.append(tok)
            return result

        def split_on_tokens(tok_list, text):
            if not text.strip():
                return []
            if not tok_list:
                return self.text_tokenizer.encode(text)

            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self._command_token_tokens:
                        tokenized_text.extend(split_on_token(tok, sub_text))
                    else:
                        tokenized_text.append(sub_text)
                text_list = tokenized_text

            return list(
                itertools.chain.from_iterable(
                    (
                        self._encode(token) if token not in self._command_token_tokens else [
                            self.command_token_map[token].Id] for token in tokenized_text
                    )
                )
            )

        no_split_tokens = self._command_tokens
        Ids = split_on_tokens(no_split_tokens, processed_text)
        tokenization = Tokenization(Ids, processed_text, text)
        tokenization.set_command_tokens(self._command_tokens)
        return tokenization

    def _encode(self, text):
        raise NotImplementedError

    def EncodeAsTokens(self, text, process_fn=None):
        """
        encode text as tokens using text tokenizer
        """
        tokenization = self.text_tokenizer.EncodeAsTokens(text, process_fn=process_fn)
        tokenization.set_command_tokens(self._command_tokens)
        return tokenization

    def IdToToken(self, Id, type_token=False):
        """convert Id to token accounting for command and type tokens"""
        if isinstance(Id, (TypeToken, CommandToken)):
            return Id.token
        if type_token:
            return self.type_id_map[Id].token
        if Id < self.num_command_tokens:
            return self.command_id_map[Id].token
        return self.text_tokenizer.IdToToken(Id - self.num_command_tokens)

    def TokenToId(self, token, type_token=False):
        """convert token to Id accounting for command and type tokens"""
        if isinstance(token, (TypeToken, CommandToken)):
            return token.Id
        if type_token:
            return self.type_token_map[token].Id
        if token in self.command_token_map:
            return self.command_token_map[token].Id
        return self.text_tokenizer.TokenToId(token) + self.num_command_tokens

    def DecodeIds(self, Ids, type_token=False):
        """
        convert Ids to tokens accounting for command and type tokens, tokens
        are joined and returned as a string.
        """
        if type_token:
            return ' '.join(Id.token if isinstance(Id, TypeToken) else self.type_id_map[Id].token for Id in Ids)
        rtn_strs = []
        current_str = []
        if isinstance(Ids, Tokenization):
            Ids = Ids.tokenization
        for Id in Ids:
            if isinstance(Id, CommandToken):
                rtn_strs.append(self.text_tokenizer.DecodeIds(current_str))
                current_str = []
                rtn_strs.append(Id.token)
            elif Id < self.num_command_tokens:
                rtn_strs.append(self.text_tokenizer.DecodeIds(current_str))
                current_str = []
                rtn_strs.append(self.command_id_map[Id].token)
            else:
                current_str.append(Id - self.num_command_tokens)
        if current_str != []:
            rtn_strs.append(self.text_tokenizer.DecodeIds(current_str))
        return ' '.join(rtn_strs)

    def DecodeTokens(self, Tokens, type_token=False):
        """
        convert tokens to a string accounting for command and type tokens.
        """
        if type_token:
            return ' '.join(t.token if isinstance(t, TypeToken) else t for t in Tokens)
        rtn_strs = []
        current_str = []
        if isinstance(Tokens, Tokenization):
            Tokens = Tokens.tokenization
        for t in Tokens:
            if isinstance(t, CommandToken):
                rtn_strs.append(self.text_tokenizer.DecodeTokens(current_str))
                current_str = []
                rtn_strs.append(t.token)
            elif t in self.command_token_map:
                rtn_strs.append(self.text_tokenizer.DecodeTokens(current_str))
                current_str = []
                rtn_strs.append(t)
            else:
                current_str.append(t)
        if current_str != []:
            rtn_strs.append(self.text_tokenizer.DecodeTokens(current_str))
        return ' '.join(rtn_strs)


class TextTokenizer(object):
    """
    Interface for text tokenizer
    """

    def __init__(self):
        if not hasattr(self, 'num_text_tokens'):
            self.num_text_tokens = 0
        if not hasattr(self, 'num_tokens'):
            self.num_tokens = self.num_text_tokens

    def __call__(self, text, process_fn=None):
        return self.EncodeAsIds(text, process_fn)

    def __len__(self):
        return self.num_text_tokens

    @property
    def tokens(self):
        """list (or iterable) of text tokens for text tokenizer"""
        raise NotImplementedError('TextTokenizer tokens property not implemented')

    @property
    def vocab(self):
        """dictionary mapping tokens to ids"""
        raise NotImplementedError('TextTokenizer vocab property not implemented')

    @staticmethod
    def exists(model_path):
        """check if the filepath for a text tokenizer exists"""
        raise NotImplementedError('TextTokenizer exists method not implemented')

    def Train(self, corpus):
        """train a tokenizer on a data corpus and save model for future use"""
        raise NotImplementedError('TextTokenizer Train not implemented')

    def EncodeAsIds(self, text, process_fn=None):
        """
        Preprocess text and encode as ids. Return a tokenization object with
        original text, processed text, and id tokenization.
        """
        raise NotImplementedError('TextTokenizer EncodeAsIds not implemented')

    def EncodeAsTokens(self, text, process_fn=None):
        """
        Preprocess text and encode as tokens. Return a tokenization object with
        original text, processed text, and token tokenization.
        """
        raise NotImplementedError('TextTokenizer EncodeAsTokens not implemented')

    def IdToToken(self, Id):
        """Convert an Id to Token. Reverse lookup of self.vocab"""
        raise NotImplementedError('TextTokenizer IdToToken not implemented')

    def TokenToId(self, token):
        """Convert a Token to Id. Lookup of self.vocab"""
        raise NotImplementedError('TextTokenizer TokenToId not implemented')

    def DecodeIds(self, Ids):
        """Convert a list or tokenization object of Ids to a text string"""
        raise NotImplementedError('TextTokenizer DecodeIds not implemented')

    def DecodeTokens(self, Tokens):
        """Convert a list or tokenization object of tokens to a text string"""
        raise NotImplementedError('TextTokenizer DecodeTokens not implemented')


class CharacterLevelTokenizer(TextTokenizer):
    """
    Text tokenizer for ASCII-256 Character Level Tokenization.
    """

    def __init__(self, **kwargs):
        self.num_text_tokens = 256
        super(CharacterLevelTokenizer, self).__init__()
        self._tokens = [self.IdToToken(Id) for Id in range(self.num_text_tokens)]
        self._vocab = {t: i for i, t in enumerate(self._tokens)}

    def __len__(self):
        return 256

    @staticmethod
    def exists(model_path):
        return True

    def Train(self, corpus):
        pass

    @property
    def tokens(self):
        return self._tokens

    @property
    def vocab(self):
        return self._vocab

    def EncodeAsIds(self, text, process_fn=None):
        """convert text to ascii 256 Ids"""
        processed_text = text
        if process_fn is not None:
            processed_text = process_fn(processed_text)
            processed_text = str(processed_text)
        tokens = [self.TokenToId(c) for c in processed_text]
        return Tokenization(tokens, processed_text, text)

    def EncodeAsTokens(self, text, process_fn=None):
        """convert text to ascii 256 characters"""
        processed_text = text
        if process_fn is not None:
            processed_text = process_fn(processed_text)
        processed_text = str(processed_text)
        tokens = [c for c in processed_text]
        return Tokenization(tokens, processed_text, text, asIds=False)

    def IdToToken(self, Id):
        """ascii index to character"""
        return chr(Id)

    def TokenToId(self, token):
        """ascii character to index"""
        return ord(token)

    def DecodeIds(self, Ids):
        """converts ascii ids to tokens before joining them into text"""
        if isinstance(Ids, Tokenization):
            Ids = Ids.tokenization
        return ''.join([self.IdToToken(tok) for tok in Ids])

    def DecodeTokens(self, Tokens):
        """just concatenates ascii tokens into text"""
        if isinstance(Tokens, Tokenization):
            Tokens = Tokens.tokenization
        return ''.join(Tokens)


MAX_SENTENCEPIECE_SENTENCES = 100000000


def get_corpus_freq(dataset, filepath, filetype='tsv'):
    """
    Take corpus, split it into sentences, and extract word frequencies.
    Write frequencies to `filepath` as a tsv. Only write the first
    MAX_SENTENCEPIECE_SENTENCES most common words to the file.
    """
    nltk.download('punkt', download_dir="./nltk")
    if filetype == 'tsv':
        delimiter = '\t'
    else:
        delimiter = ','

    print("compute corpus frequency\n", flush=True)

    total_sentence_count = 0
    maxlen = 0
    freqs = {}
    for entry in dataset:
        if isinstance(entry, dict):
            entry = entry['text']
        lines = entry.strip().split('\n')
        for line in lines:
            sentences = nltk_tokenize.sent_tokenize(line)
            total_sentence_count += len(sentences)
            for sentence in sentences:
                maxlen = max(len(line), maxlen)
                for word in sentence.split():
                    if word not in freqs:
                        freqs[word] = 0
                    freqs[word] += 1

    print("length of freqs before truncating " + str(len(freqs)), flush=True)
    print("file path for freq " + str(filepath), flush=True)

    freqs_sorted = {}
    counter = 0
    for word, count in sorted(freqs.items(), key=lambda x: x[1], reverse=True):
        if counter >= MAX_SENTENCEPIECE_SENTENCES:
            break
        counter += 1
        freqs_sorted[word] = count

    print("length of freqs after trancating " + str(len(freqs_sorted)), flush=True)

    with open(filepath, 'w') as f:
        writer = csv.writer(f, delimiter=delimiter)
        for k, v in freqs_sorted.items():
            writer.writerow([str(k), str(v)])

    return total_sentence_count, maxlen


class SentencePieceTokenizer(TextTokenizer):
    """Trains and uses sentencepiece for text tokenization"""

    def __init__(self, model_type='bpe', vocab_size=None, corpus=None, model_path=None, character_coverage=1.0,
                 **kwargs):
        self.character_coverage = character_coverage
        self.model_type = model_type.lower()
        self.spm_model = model_path
        self.num_text_tokens = vocab_size
        make_train = not SentencePieceTokenizer.exists(self.spm_model)
        if make_train:
            assert corpus is not None and self.num_text_tokens is not None
            self.Train(corpus, self.num_text_tokens)
        self._tokens = []
        self._vocab = {}
        self.load_spm_model()
        super(SentencePieceTokenizer, self).__init__()

    def __len__(self):
        return self.num_text_tokens

    @property
    def tokens(self):
        return self._tokens

    @property
    def vocab(self):
        return self._vocab

    @staticmethod
    def exists(model_path):
        if model_path is None:
            return False
        # check if path exists
        dne = not os.path.exists(model_path)
        # check if path.model exists
        if dne and not model_path.endswith('.model'):
            dne = not os.path.exists(model_path + '.model')
        return not dne

    def load_spm_model(self):
        """load sentencepiece model and parse vocab"""
        if not os.path.exists(self.spm_model) and not self.spm_model.endswith('.model'):
            self.spm_model = self.spm_model + '.model'
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(self.spm_model)
        self.vocab_size = self.num_text_tokens = len(self.sp)
        self._tokens = [self.IdToToken(t) for t in range(self.vocab_size)]
        self._vocab = {t: i for i, t in enumerate(self._tokens)}

    def Train(self, corpus, num_text_tokens):
        """train sentencepiece model on corpus using word frequencies"""
        self.num_text_tokens = num_text_tokens
        use_model_path = self.spm_model
        random_hash = str(random.randint(0, 2147483647))
        if use_model_path is None:
            use_model_path = random_hash
        if use_model_path.endswith('.model'):
            use_model_path = use_model_path[:use_model_path.rfind('.model')]
        input_path = use_model_path + '.tsv.' + random_hash
        line_count, maxlenline = get_corpus_freq(corpus, input_path)
        line_count = min(line_count, MAX_SENTENCEPIECE_SENTENCES)
        print('line count used as input_sentence_size ', line_count, flush=True)
        print('training sentencepiece model', flush=True)
        train_string = '--input={file_path} --model_prefix={model_prefix} --vocab_size={vocab_size}' \
                       + ' --model_type={model_type} --character_coverage={character_coverage} ' \
                       + '--input_sentence_size={input_sentence_size} ' \
                       + '--input_format=tsv'
        train_string = train_string.format(file_path=input_path, model_prefix=use_model_path,
                                           vocab_size=num_text_tokens,
                                           model_type=self.model_type, character_coverage=self.character_coverage,
                                           input_sentence_size=int(line_count))  # , #)#,
        print("calling spm.SentencePieceTrainer.Train(%s)" % (train_string), flush=True)
        spm.SentencePieceTrainer.Train(train_string)
        os.remove(input_path)
        self.spm_model = use_model_path + '.model'
        print('sentencepiece model written to ' + self.spm_model, flush=True)

    def EncodeAsIds(self, text, process_fn=None):
        """convert text to sentencepiece Ids"""
        processed_text = text
        if process_fn is not None:
            processed_text = process_fn(processed_text)
        tokens = self.sp.EncodeAsIds(processed_text)
        return Tokenization(tokens, processed_text, text)

    def EncodeAsTokens(self, text, process_fn=None):
        """convert text to sentencepiece tokens"""
        processed_text = text
        if process_fn is not None:
            processed_text = process_fn(processed_text)
        tokens = self.sp.EncodeAsTokens(processed_text)
        return Tokenization(tokens, processed_text, text, asIds=False)

    def IdToToken(self, Id):
        """convert Id to sentencpiece token"""
        return self.sp.IdToPiece(Id)

    def TokenToId(self, token):
        """convert sentencpiece token to Id"""
        return self.sp.PieceToId(token)

    def DecodeIds(self, Ids):
        """converts ids to a text string"""
        if isinstance(Ids, Tokenization):
            Ids = Ids.tokenization
        return self.sp.DecodeIds(Ids)

    def DecodeTokens(self, Tokens):
        """converts sentencepiece tokens to a text string"""
        if isinstance(Tokens, Tokenization):
            Tokens = Tokens.tokenization
        return self.sp.DecodeTokens(Tokens)


class BertWordPieceTokenizer(Tokenizer):
    """
    Loads a pretrained WordPiece tokenizer from `cache_dir` for tokenization
    in BERT training. Default to bert-large-uncased tokenizer.
    """

    def __init__(self, tokenizer_model_type=None, cache_dir=None, add_block_symbols=False, add_sentinel_token=0,
                 add_task_mask=False, add_decoder_mask=False, **kwargs):
        # default to bert-large-uncased tokenizer
        if tokenizer_model_type not in PRETRAINED_VOCAB_ARCHIVE_MAP:
            tokenizer_model_type = 'bert-large-uncased'
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print('loading BertWordPieceTokenizer (', tokenizer_model_type, ') from cache_dir ', cache_dir)
        do_lower_case = not ('-cased' in tokenizer_model_type or 'chinese' in tokenizer_model_type)
        self.text_tokenizer = BertTokenizer.from_pretrained(tokenizer_model_type, do_lower_case=do_lower_case,
                                                            cache_dir=cache_dir)
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print('loaded', tokenizer_model_type)
        # disable max len warnings by increasing max len
        self.text_tokenizer.max_len = int(1e12)

        # set command tokens from wordpiece tokenizer values
        self.num_command_tokens = 6
        self.num_tokens = len(self.text_tokenizer.vocab)
        self.num_text_tokens = self.num_tokens - 5
        self.num_type_tokens = 2

        self._command_tokens = [
            CommandToken('pad', '[PAD]', self.text_tokenizer.vocab['[PAD]']),
            CommandToken('ENC', '[CLS]', self.text_tokenizer.vocab['[CLS]']),
            CommandToken('MASK', '[MASK]', self.text_tokenizer.vocab['[MASK]']),
            CommandToken('unk', '[UNK]', self.text_tokenizer.vocab['[UNK]']),
            CommandToken('sep', '[SEP]', self.text_tokenizer.vocab['[SEP]']),
            CommandToken('eos', '[PAD]', self.text_tokenizer.vocab['[PAD]']),
        ]
        if add_block_symbols:
            self._command_tokens.extend([
                CommandToken('sop', '<|startofpiece|>', self.num_tokens),
                CommandToken('eop', '<|endofpiece|>', self.num_tokens + 1)
            ])
            self.num_tokens += 2
            self.num_command_tokens += 2
            if add_task_mask:
                self._command_tokens.extend([
                    CommandToken('gMASK', '[gMASK]', self.num_tokens),
                    CommandToken('sMASK', '[sMASK]', self.num_tokens + 1)
                ])
                self.num_tokens += 2
                self.num_command_tokens += 2
            if add_decoder_mask:
                self._command_tokens.extend([
                    CommandToken('dBLOCK', '[dBLOCK]', self.num_tokens)
                ])
                self.num_tokens += 1
                self.num_command_tokens += 1
        if add_sentinel_token > 0:
            for i in range(1, add_sentinel_token):
                self._command_tokens.extend([CommandToken(f'MASK{i}', f'[MASK{i}]', self.num_tokens),
                                             CommandToken(f'sop{i}', f'<|startofpiece{i}|>', self.num_tokens + 1)])
                self.num_tokens += 2
                self.num_command_tokens += 2
        self.command_name_map = {tok.name: tok for tok in self._command_tokens}
        self.command_token_map = {tok.token: tok for tok in self._command_tokens}
        self.command_id_map = {tok.Id: tok for tok in self._command_tokens}

        # set type tokens
        self.type_tokens = [
            TypeToken('str0', '<str0>', 0),
            TypeToken('str1', '<str1>', 1),
        ]
        self.type_name_map = {tok.name: tok for tok in self.type_tokens}
        self.type_token_map = {tok.token: tok for tok in self.type_tokens}
        self.type_id_map = {tok.Id: tok for tok in self.type_tokens}

        # parse tokens and vocabs from tokenizer

        self._tokens = list(self.text_tokenizer.vocab.keys())
        self._vocab = {k: v for k, v in self.text_tokenizer.vocab.items()}

        self._text_tokens = list(self._tokens)
        self._text_token_vocab = {k: v for k, v in self.text_tokenizer.vocab.items()}

        self._command_token_tokens = list(self.command_token_map.keys())
        self._command_token_vocab = {t: Id for Id, t in self.command_id_map.items()}

        self._token_types = list(self.type_token_map.keys())
        self._token_type_vocab = {t: Id for Id, t in self.type_id_map.items()}

    def _encode(self, text):
        tokens = self.text_tokenizer.tokenize(text)
        ids = self.text_tokenizer.convert_tokens_to_ids(tokens)
        return ids

    def EncodeAsTokens(self, text, process_fn=None):
        """convert wordpiece token to Id"""
        processed_text = text
        if process_fn is not None:
            processed_text = process_fn(processed_text)
        tokens = self.text_tokenizer.tokenize(processed_text)
        return Tokenization(tokens, processed_text, text, asIds=False)

    def IdToToken(self, Id, type_token=False):
        """convert Id to sentencpiece token"""
        if isinstance(Id, (TypeToken, CommandToken)):
            return Id.token
        if type_token:
            return self.type_id_map[Id].token
        if Id in self.command_id_map:
            return self.command_id_map[Id].token
        return self.text_tokenizer.ids_to_tokens[Id]

    def TokenToId(self, token, type_token=False):
        """convert sentencpiece token to Id"""
        if isinstance(token, (TypeToken, CommandToken)):
            return token.Id
        if type_token:
            return self.type_token_map[token].Id
        token = token.strip()
        return self.text_tokenizer.vocab[token]

    def DecodeIds(self, Ids, type_token=False):
        """converts ids to wordpiece tokens and joins them as a text string"""
        if type_token:
            return ' '.join(Id.token if isinstance(Id, TypeToken) else self.type_id_map[Id].token for Id in Ids)
        if isinstance(Ids, Tokenization):
            Ids = Ids.tokenization
        Tokens = []
        for Id in Ids:
            if Id in self.command_id_map:
                Tokens.append(self.command_id_map[Id].token)
            elif Id in self.text_tokenizer.ids_to_tokens:
                Tokens.append(self.text_tokenizer.ids_to_tokens[Id])
        new_tokens = []
        for token in Tokens:
            if token.startswith('##') and len(new_tokens) > 0:
                new_tokens[-1] += token[2:]
            else:
                new_tokens.append(token)
        return ' '.join(new_tokens)

    def DecodeTokens(self, Tokens, type_token=False):
        """converts wordpiece tokens to a text string"""
        if type_token:
            return ' '.join(t.token if isinstance(t, TypeToken) else t for t in Tokens)
        if isinstance(Tokens, Tokenization):
            Tokens = Tokens.tokenization
        return ' '.join(Tokens)

'''
class GPT2BPETokenizer(Tokenizer):
    def __init__(self, model_type_or_path, cache_dir=None, add_block_symbols=True, add_task_mask=True,
                 add_decoder_mask=False, **kwargs):
        # self.text_tokenizer = GPT2Tokenizer.from_pretrained(model_type_or_path,
        #                                                     cache_dir=cache_dir)

        #修改加载tokenizer的代码，改为自己训练的tokenizer
        
        tokenizer_path = r"/mnt/d/111QYH/progect/GLM/GLM-main/tokenizer/output/protein/BPE_tokenizer/BPE_tokenizer_proANDsmi_plus_4096size.json"
        tokenizer_1 = Tokenizer_rengong(BPE())
        self.text_tokenizer = tokenizer_1.from_file(tokenizer_path)

        from tokenizers import Tokenizer
        from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM, AutoModelForMultipleChoice
        from transformers import Trainer, TrainingArguments
        new_tokenizer = Tokenizer.from_file(
            r"/mnt/d/111QYH/progect/GLM/GLM-main/tokenizer/output/protein/BPE_tokenizer/new_tok/ProSmi_34999.json")
        # new_tokenizer = Tokenizer.from_file(r"D:\PyCharm Community Edition 2023.1\data\DNAgpt\dnagpt-main\data\human2_formal.json")
        # 或者下面方法
        from transformers import GPT2TokenizerFast
        self.text_tokenizer = GPT2TokenizerFast(tokenizer_object=new_tokenizer)

        # disable max len warnings by increasing max len
        self.text_tokenizer.max_len = int(1e12)
        # self.num_tokens = len(self.text_tokenizer.encoder)
        self.num_tokens = len(self.text_tokenizer.get_vocab())
        self.num_type_tokens = 2
        if model_type_or_path.startswith('roberta'):
            self.num_command_tokens = 6
            self.num_text_tokens = self.num_tokens - 3
            self._command_tokens = [
                CommandToken('pad', '<|endoftext|>', self.text_tokenizer.encoder['</s>']),
                CommandToken('eos', '<|endoftext|>', self.text_tokenizer.encoder['</s>']),
                CommandToken('sep', '[SEP]', self.text_tokenizer.encoder['</s>']),
                CommandToken('ENC', '[CLS]', self.text_tokenizer.encoder['<s>']),
                CommandToken('MASK', '[MASK]', self.text_tokenizer.encoder['<mask>'], lstrip=True),
                CommandToken('unk', '[UNK]', self.text_tokenizer.encoder['<unk>'])
            ]
            if add_block_symbols:
                self._command_tokens.extend([
                    CommandToken('sop', '<|startofpiece|>', self.num_tokens),
                    CommandToken('eop', '<|endofpiece|>', self.num_tokens + 1)
                ])
                self.num_tokens += 2
                self.num_command_tokens += 2
        else:
            self.num_command_tokens = 2
            self.num_text_tokens = self.num_tokens - 1
            self._command_tokens = [
                # CommandToken('pad', '<|endoftext|>', self.text_tokenizer.encoder['<|endoftext|>']),
                # CommandToken('eos', '<|endoftext|>', self.text_tokenizer.encoder['<|endoftext|>'])
                CommandToken('pad', '[PAD]', self.text_tokenizer.get_vocab()['[PAD]']),
                CommandToken('eos', '[PAD]', self.text_tokenizer.get_vocab()['[PAD]'])
            ]
            if add_block_symbols:
                self._command_tokens.extend([
                    CommandToken('sop', '[startofpiece]', self.num_tokens),
                    CommandToken('eop', '[endofpiece]', self.num_tokens + 1),
                    CommandToken('ENC', '[Cls]', self.num_tokens + 2),
                    CommandToken('MASK', '[Mask]', self.num_tokens + 3, lstrip=True),
                    CommandToken('sep', '[Sep]', self.num_tokens + 4),
                    CommandToken('unk', '[Unk]', self.num_tokens + 5)
                ])
                self.num_tokens += 6
                self.num_command_tokens += 6
        if add_block_symbols:
            if add_task_mask:
                self._command_tokens.extend([
                    CommandToken('gMASK', '[gMASK]', self.num_tokens, lstrip=True),
                    CommandToken('sMASK', '[sMASK]', self.num_tokens + 1, lstrip=True)
                ])
                self.num_tokens += 2
                self.num_command_tokens += 2
            if add_decoder_mask:
                self._command_tokens.extend([
                    CommandToken('dBLOCK', '[dBLOCK]', self.num_tokens)
                ])
                self.num_tokens += 1
                self.num_command_tokens += 1
        if True:  #添加蛋白序列/分子smiles token
            self._command_tokens.extend([
                CommandToken('pro', '[pro]', self.num_tokens, lstrip=True),
                CommandToken('smi', '[smi]', self.num_tokens + 1, lstrip=True)
            ])
            self.num_tokens += 2
            self.num_command_tokens += 2
        self.command_name_map = {tok.name: tok for tok in self._command_tokens}
        self.command_token_map = {tok.token: tok for tok in self._command_tokens}
        self.command_id_map = {tok.Id: tok for tok in self._command_tokens}

        self.type_tokens = [
            TypeToken('str0', '<str0>', 0),
            TypeToken('str1', '<str1>', 1),
        ]
        self.type_name_map = {tok.name: tok for tok in self.type_tokens}
        self.type_token_map = {tok.token: tok for tok in self.type_tokens}
        self.type_id_map = {tok.Id: tok for tok in self.type_tokens}

        # self._tokens = list(self.text_tokenizer.encoder.keys())
        self._tokens = list(self.text_tokenizer.get_vocab().keys())
        # self._vocab = {k: v for k, v in self.text_tokenizer.encoder.items()}
        self._vocab = {k: v for k, v in self.text_tokenizer.get_vocab().items()}

        self._text_tokens = list(self._tokens)
        # self._text_token_vocab = {k: v for k, v in self.text_tokenizer.encoder.items()}
        self._text_token_vocab = {k: v for k, v in self.text_tokenizer.get_vocab().items()}

        self._command_token_tokens = list(self.command_token_map.keys())
        self._command_token_vocab = {t: Id for Id, t in self.command_id_map.items()}

        self._token_types = list(self.type_token_map.keys())
        self._token_type_vocab = {t: Id for Id, t in self.type_id_map.items()}

        # for idx, tok in self.command_id_map.items():
        #     self.text_tokenizer.decoder[idx] = tok.token
    
    def get_vocab(self):
        return self._vocab

    def get_tokens(self):
        return self._tokens
    def get_commend_token(self):
        return self._command_tokens

    def EncodeAsIds(self, text, process_fn=None):
        processed_text = text
        if process_fn is not None:
            processed_text = process_fn(processed_text)

        def split_on_token(tok_extended: CommandToken, text):
            result = []
            tok = tok_extended.token
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
                # CommandToken can control whitespace stripping around them.
                # We use them for GPT2 and Roberta to have different behavior depending on the special token
                # Cf. https://github.com/huggingface/transformers/pull/2778
                # and https://github.com/huggingface/transformers/issues/3788
                # Strip white spaces on the right
                if tok_extended.rstrip and i > 0:
                    # A bit counter-intuitive but we strip the left of the string
                    # since tok_extended.rstrip means the special token is eating all white spaces on its right
                    sub_text = sub_text.lstrip()
                # Strip white spaces on the left
                if tok_extended.lstrip and i < len(split_text) - 1:
                    sub_text = sub_text.rstrip()  # Opposite here

                if i == 0 and not sub_text:
                    result.append(tok)
                elif i == len(split_text) - 1:
                    if sub_text:
                        result.append(sub_text)
                    else:
                        pass
                else:
                    if sub_text:
                        result.append(sub_text)
                    result.append(tok)
            return result

        def split_on_tokens(tok_list, text):
            if not text.strip():
                return []
            if not tok_list:
                return self.text_tokenizer.encode(text)

            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self._command_token_tokens:
                        tokenized_text.extend(split_on_token(tok, sub_text))
                    else:
                        tokenized_text.append(sub_text)
                text_list = tokenized_text

            return list(
                itertools.chain.from_iterable(
                    (
                        self.text_tokenizer.encode(token).ids if token not in self._command_token_tokens else [
                            self.command_token_map[token].Id] for token in tokenized_text
                    )
                )
            )

        no_split_tokens = self._command_tokens
        Ids = split_on_tokens(no_split_tokens, processed_text)
        tokenization = Tokenization(Ids, processed_text, text)
        tokenization.set_command_tokens(self._command_tokens)
        return tokenization

    def _encode(self, text):
        return self.text_tokenizer.encode(text)

    def EncodeAsTokens(self, text, process_fn=None):
        processed_text = text
        if process_fn is not None:
            processed_text = process_fn(processed_text)
        tokens = []
        for token in re.findall(self.text_tokenizer.pat, processed_text):
            token = ''.join(self.text_tokenizer.bye_encoder[b] for b in token.encode('utf-8'))
            tokens.extend(bpe_token for bpe_token in self.text_tokenizer.bpe(token).split(' '))
        tokenization = Tokenization(tokens, processed_text, text, asIds=False)
        tokenization.set_command_tokens(self._command_tokens)
        return tokenization

    def DecodeAsTokens(self, Ids):
        return [self.IdToToken(x) for x in Ids]

    def IdToToken(self, Id, type_token=False):
        if isinstance(Id, (TypeToken, CommandToken)):
            return Id.token
        if type_token:
            return self.type_id_map[Id].token
        if Id in self.command_id_map:
            return self.command_id_map[Id].token
        #return self.text_tokenizer.decoder[Id]
        return self.text_tokenizer.id_to_token(Id)

    def TokenToId(self, token, type_token=False):
        if isinstance(token, (TypeToken, CommandToken)):
            return token.Id
        if type_token:
            return self.type_token_map[token].Id
        return self.text_tokenizer.get_vocab()[token]

    def DecodeIds(self, Ids, type_token=False):
        if type_token:
            return ' '.join(Id.token if isinstance(Id, TypeToken) else self.type_id_map[Id].token for Id in Ids)
        if isinstance(Ids, Tokenization):
            Ids = Ids.tokenization
        return self.text_tokenizer.decode(Ids)

    def DecodeTokens(self, Tokens, type_token=False):
        if type_token:
            return ' '.join(t.token if isinstance(t, TypeToken) else t for t in Tokens)
        if isinstance(Tokens, Tokenization):
            Tokens = Tokens.tokenization
        return self.text_tokenizer.decode([self.TokenToId(tok) for tok in Tokens])
'''

class GPT2BPETokenizer(Tokenizer):
    def __init__(self, model_type_or_path, cache_dir=None, add_block_symbols=True, add_task_mask=True,
                 add_decoder_mask=False, **kwargs):

        from tokenizers import Tokenizer
        new_tokenizer = Tokenizer.from_file(
            r"/home/qyh_vm/GLM_Linux/data/my_tokenizer/pro_22+smi_64_tokenzier/tokenizer.json")
        # new_tokenizer = Tokenizer.from_file(r"D:\PyCharm Community Edition 2023.1\data\DNAgpt\dnagpt-main\data\human2_formal.json")
        # 或者下面方法
        from transformers import GPT2TokenizerFast
        self.text_tokenizer = GPT2TokenizerFast(tokenizer_object=new_tokenizer)

        #self.text_tokenizer = tokenizer
        self.encoder = self.text_tokenizer.get_vocab()
        self.decoder = {v: k for k, v in self.encoder.items()}

        # disable max len warnings by increasing max len
        self.text_tokenizer.max_len = int(1e12)
        # self.num_tokens = len(self.text_tokenizer.encoder)
        self.num_tokens = len(self.encoder)
        self.num_type_tokens = 2
        if model_type_or_path.startswith('roberta'):
            self.num_command_tokens = 6
            self.num_text_tokens = self.num_tokens - 3
            self._command_tokens = [
                CommandToken('pad', '<|endoftext|>', self.text_tokenizer.encoder['</s>']),
                CommandToken('eos', '<|endoftext|>', self.text_tokenizer.encoder['</s>']),
                CommandToken('sep', '[SEP]', self.text_tokenizer.encoder['</s>']),
                CommandToken('ENC', '[CLS]', self.text_tokenizer.encoder['<s>']),
                CommandToken('MASK', '[MASK]', self.text_tokenizer.encoder['<mask>'], lstrip=True),
                CommandToken('unk', '[UNK]', self.text_tokenizer.encoder['<unk>'])
            ]
            if add_block_symbols:
                self._command_tokens.extend([
                    CommandToken('sop', '<|startofpiece|>', self.num_tokens),
                    CommandToken('eop', '<|endofpiece|>', self.num_tokens + 1)
                ])
                self.num_tokens += 2
                self.num_command_tokens += 2
        else:
            self.num_command_tokens = 2
            self.num_text_tokens = self.num_tokens - 1
            self._command_tokens = [
                CommandToken('pad', '[endoftext]', self.encoder['[endoftext]']),
                CommandToken('eos', '[endoftext]', self.encoder['[endoftext]'])
            ]
        if add_block_symbols:
            self._command_tokens.extend([
                CommandToken('sop', '[startofpiece]', self.num_tokens),
                CommandToken('eop', '[endofpiece]', self.num_tokens + 1),
                CommandToken('ENC', '[ClS]', self.num_tokens + 2),
                CommandToken('MASK', '[Mask]', self.num_tokens + 3),
                CommandToken('sep', '[Sep]', self.num_tokens + 4),
                CommandToken('unk', '[Unk]', self.num_tokens + 5)
            ])
            self.num_tokens += 6
            self.num_command_tokens += 6
        if add_block_symbols:
            if add_task_mask:
                self._command_tokens.extend([
                    CommandToken('gMASK', '[gMASK]', self.num_tokens, lstrip=True),
                    CommandToken('sMASK', '[sMASK]', self.num_tokens + 1, lstrip=True)
                ])
                self.num_tokens += 2
                self.num_command_tokens += 2
            if add_decoder_mask:
                self._command_tokens.extend([
                    CommandToken('dBLOCK', '[dBLOCK]', self.num_tokens)
                ])
                self.num_tokens += 1
                self.num_command_tokens += 1
        self.command_name_map = {tok.name: tok for tok in self._command_tokens}
        self.command_token_map = {tok.token: tok for tok in self._command_tokens}
        self.command_id_map = {tok.Id: tok for tok in self._command_tokens}

        self.type_tokens = [
            TypeToken('str0', '<str0>', 0),
            TypeToken('str1', '<str1>', 1),
        ]
        self.type_name_map = {tok.name: tok for tok in self.type_tokens}
        self.type_token_map = {tok.token: tok for tok in self.type_tokens}
        self.type_id_map = {tok.Id: tok for tok in self.type_tokens}

        self._tokens = list(self.encoder.keys())
        self._vocab = {k: v for k, v in self.encoder.items()}

        self._text_tokens = list(self._tokens)
        self._text_token_vocab = {k: v for k, v in self.encoder.items()}

        self._command_token_tokens = list(self.command_token_map.keys())
        self._command_token_vocab = {t: Id for Id, t in self.command_id_map.items()}

        self._token_types = list(self.type_token_map.keys())
        self._token_type_vocab = {t: Id for Id, t in self.type_id_map.items()}

        for idx, tok in self.command_id_map.items():
            self.decoder[idx] = tok.token

        self.command_name_map = {tok.name: tok for tok in self._command_tokens}

    def EncodeAsIds(self, text, process_fn=None):
        processed_text = text
        if process_fn is not None:
            processed_text = process_fn(processed_text)

        def split_on_token(tok_extended: CommandToken, text):
            result = []
            tok = tok_extended.token
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
                # CommandToken can control whitespace stripping around them.
                # We use them for GPT2 and Roberta to have different behavior depending on the special token
                # Cf. https://github.com/huggingface/transformers/pull/2778
                # and https://github.com/huggingface/transformers/issues/3788
                # Strip white spaces on the right
                if tok_extended.rstrip and i > 0:
                    # A bit counter-intuitive but we strip the left of the string
                    # since tok_extended.rstrip means the special token is eating all white spaces on its right
                    sub_text = sub_text.lstrip()
                # Strip white spaces on the left
                if tok_extended.lstrip and i < len(split_text) - 1:
                    sub_text = sub_text.rstrip()  # Opposite here

                if i == 0 and not sub_text:
                    result.append(tok)
                elif i == len(split_text) - 1:
                    if sub_text:
                        result.append(sub_text)
                    else:
                        pass
                else:
                    if sub_text:
                        result.append(sub_text)
                    result.append(tok)
            return result

        def split_on_tokens(tok_list, text):
            if not text.strip():
                return []
            if not tok_list:
                return self.text_tokenizer.encode(text)

            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self._command_token_tokens:
                        tokenized_text.extend(split_on_token(tok, sub_text))
                    else:
                        tokenized_text.append(sub_text)
                text_list = tokenized_text

            return list(
                itertools.chain.from_iterable(
                    (
                        self.text_tokenizer.encode(token) if token not in self._command_token_tokens else [
                            self.command_token_map[token].Id] for token in tokenized_text
                    )
                )
            )

        no_split_tokens = self._command_tokens
        Ids = split_on_tokens(no_split_tokens, processed_text)
        tokenization = Tokenization(Ids, processed_text, text)
        tokenization.set_command_tokens(self._command_tokens)
        return tokenization

    def _encode(self, text):
        return self.text_tokenizer.encode(text)

    def EncodeAsTokens(self, text, process_fn=None):
        processed_text = text
        if process_fn is not None:
            processed_text = process_fn(processed_text)
        tokens = []
        for token in re.findall(self.text_tokenizer.pat, processed_text):
            token = ''.join(self.text_tokenizer.bye_encoder[b] for b in token.encode('utf-8'))
            tokens.extend(bpe_token for bpe_token in self.text_tokenizer.bpe(token).split(' '))
        tokenization = Tokenization(tokens, processed_text, text, asIds=False)
        tokenization.set_command_tokens(self._command_tokens)
        return tokenization

    def DecodeAsTokens(self, Ids):
        return [self.IdToToken(x) for x in Ids]

    def IdToToken(self, Id, type_token=False):
        if isinstance(Id, (TypeToken, CommandToken)):
            return Id.token
        if type_token:
            return self.type_id_map[Id].token
        if Id in self.command_id_map:
            return self.command_id_map[Id].token
        return self.decoder[Id]

    def TokenToId(self, token, type_token=False):
        if isinstance(token, (TypeToken, CommandToken)):
            return token.Id
        if type_token:
            return self.type_token_map[token].Id
        return self.encoder[token]

    def DecodeIds(self, Ids, type_token=False):
        if type_token:
            return ' '.join(Id.token if isinstance(Id, TypeToken) else self.type_id_map[Id].token for Id in Ids)
        if isinstance(Ids, Tokenization):
            Ids = Ids.tokenization
        return self.text_tokenizer.decode(Ids)

    def DecodeTokens(self, Tokens, type_token=False):
        if type_token:
            return ' '.join(t.token if isinstance(t, TypeToken) else t for t in Tokens)
        if isinstance(Tokens, Tokenization):
            Tokens = Tokens.tokenization
        return self.text_tokenizer.decode([self.TokenToId(tok) for tok in Tokens])

    def get_command(self, name):
        """get command token corresponding to `name`"""
        return self.command_name_map[name]

class ChineseSPTokenizer(Tokenizer):
    def __init__(self, add_block_symbols=False, add_task_mask=False, add_decoder_mask=False, fix_command_token=False,
                 **kwargs):
        self.text_tokenizer = sp_tokenizer.from_pretrained()

        self.num_command_tokens = 0
        self.num_text_tokens = self.text_tokenizer.sp.vocab_size()
        self.num_tokens = self.num_text_tokens
        self.num_type_tokens = 2

        self._command_tokens = [
            CommandToken('pad', '<|endoftext|>', self.num_text_tokens),
            CommandToken('eos', '<|endoftext|>', self.num_text_tokens),
            CommandToken('sep', '[SEP]', self.num_text_tokens + 1),
            CommandToken('ENC', '[CLS]', self.num_text_tokens + 2),
            CommandToken('MASK', '[MASK]', self.num_text_tokens + 3, lstrip=True),
            CommandToken('unk', '[UNK]', self.num_text_tokens + 4)
        ]
        self.num_tokens += 5
        self.num_command_tokens += 6
        if add_block_symbols:
            self._command_tokens.extend([
                CommandToken('sop', '<|startofpiece|>', self.num_tokens + 1),
                CommandToken('eop', '<|endofpiece|>', self.num_tokens + 2)
            ])
            if fix_command_token:
                self.num_tokens += 3
            else:
                self.num_tokens += 2
            self.num_command_tokens += 2
            if add_task_mask:
                if fix_command_token:
                    self._command_tokens.extend([
                        CommandToken('sMASK', '[sMASK]', self.num_tokens, lstrip=True),
                        CommandToken('gMASK', '[gMASK]', self.num_tokens + 1, lstrip=True)
                    ])
                else:
                    self._command_tokens.extend([
                        CommandToken('gMASK', '[gMASK]', self.num_tokens, lstrip=True),
                        CommandToken('sMASK', '[sMASK]', self.num_tokens + 1, lstrip=True)
                    ])
                self.num_tokens += 2
                self.num_command_tokens += 2
            if add_decoder_mask:
                self._command_tokens.extend([
                    CommandToken('dBLOCK', '[dBLOCK]', self.num_tokens)
                ])
                self.num_tokens += 1
                self.num_command_tokens += 1
        self.command_name_map = {tok.name: tok for tok in self._command_tokens}
        self.command_token_map = {tok.token: tok for tok in self._command_tokens}
        self.command_id_map = {tok.Id: tok for tok in self._command_tokens}
        print_rank_0({tok.name: tok.Id for tok in self._command_tokens})
        self.type_tokens = [
            TypeToken('str0', '<str0>', 0),
            TypeToken('str1', '<str1>', 1),
        ]
        self.type_name_map = {tok.name: tok for tok in self.type_tokens}
        self.type_token_map = {tok.token: tok for tok in self.type_tokens}
        self.type_id_map = {tok.Id: tok for tok in self.type_tokens}

        # self._tokens = list(self.text_tokenizer.encoder.keys())
        # self._vocab = {k:v for k,v in self.text_tokenizer.encoder.items()}
        #
        # self._text_tokens = list(self._tokens)
        # self._text_token_vocab = {k:v for k,v in self.text_tokenizer.encoder.items()}

        self._command_token_tokens = list(self.command_token_map.keys())
        self._command_token_vocab = {t: Id for Id, t in self.command_id_map.items()}

        self._token_types = list(self.type_token_map.keys())
        self._token_type_vocab = {t: Id for Id, t in self.type_id_map.items()}

    def _encode(self, text):
        ids = self.text_tokenizer.encode(text)
        return ids

    def EncodeAsTokens(self, text, process_fn=None):
        processed_text = text
        if process_fn is not None:
            processed_text = process_fn(processed_text)
        tokens = self.text_tokenizer.tokenize(processed_text)
        tokenization = Tokenization(tokens, processed_text, text, asIds=False)
        tokenization.set_command_tokens(self._command_tokens)
        return tokenization
        # return Tokenization(tokens, processed_text, text, asIds=False)

    def IdToToken(self, Id, type_token=False):
        if isinstance(Id, (TypeToken, CommandToken)):
            return Id.token
        if type_token:
            return self.type_id_map[Id].token
        if Id in self.command_id_map:
            return self.command_id_map[Id].token
        elif Id in self.type_id_map:
            return self.type_id_map[Id].token
        else:
            return self.text_tokenizer.convert_id_to_token(int(Id))

    def TokenToId(self, token, type_token=False):
        if isinstance(token, (TypeToken, CommandToken)):
            return token.Id
        if type_token:
            return self.type_token_map[token].Id
        return self.text_tokenizer.convert_token_to_id(token)

    def DecodeIds(self, Ids, type_token=False):
        if type_token:
            return ' '.join(Id.token if isinstance(Id, TypeToken) else self.type_id_map[Id].token for Id in Ids)
        if isinstance(Ids, Tokenization):
            Ids = Ids.tokenization
        Ids = list(map(int, Ids))
        pieces = []
        last = 0
        for i, token_id in enumerate(Ids):
            if token_id in self.command_id_map:
                pieces.append(Ids[last: i])
                pieces.append(token_id)
                last = i + 1
        pieces.append(Ids[last:])
        text = ""
        for piece in pieces:
            if isinstance(piece, int):
                text += self.command_id_map[piece].token
            elif piece:
                text += self.text_tokenizer.decode(piece)
        return text

    def DecodeTokens(self, Tokens, type_token=False):
        if type_token:
            return ' '.join(t.token if isinstance(t, TypeToken) else t for t in Tokens)
        if isinstance(Tokens, Tokenization):
            Tokens = Tokens.tokenization
        return self.text_tokenizer.decode([self.TokenToId(tok) for tok in Tokens])
