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

"""Pretrain GPT2"""

# Flag to use Pytorch ddp which uses overlapping communication and computation.
from datetime import datetime
import os
import random
import math

import torch.distributed
from filelock import FileLock
import numpy as np
import torch

import deepspeed
from contextlib import ExitStack
from arguments import get_args
from configure_data import configure_data, prepare_tokenizer, build_multi_task_dataset
import mpu
import pathlib

from train_utils import setup_model_and_optimizer, train_step
from utils import Timers
from utils import save_checkpoint
from utils import load_checkpoint
from utils import report_memory
from utils import print_and_save_args
from utils import print_rank_0
from utils import get_sample_writer, get_log_dir, get_hostname
import torch.distributed as dist
import joblib
import json

def get_masks_and_position_ids(data,
                               eod_token,
                               reset_position_ids,
                               reset_attention_mask,
                               loss_mask=None,
                               attention_mask=None,
                               set_loss_mask=False,
                               mem_length=None):
    # Extract batch size and sequence length.
    batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if mem_length:
        if attention_mask is None:
            attention_mask = torch.ones((1, seq_length, seq_length + mem_length), device=data.device)
        attention_mask = torch.tril(torch.triu(attention_mask, 1 - seq_length + mem_length), mem_length)
    else:
        if reset_attention_mask:
            att_mask_batch = batch_size
        else:
            att_mask_batch = 1
        if attention_mask is None:
            attention_mask = torch.ones((att_mask_batch, seq_length, seq_length), device=data.device)
        attention_mask = torch.tril(attention_mask)
    attention_mask = attention_mask.unsqueeze(1)

    # Loss mask.
    if loss_mask is None:
        loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    if set_loss_mask:
        loss_mask[data == eod_token] = 0.0
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Loop through the batches:
        for b in range(batch_size):

            # Find indecies where EOD token is.
            eod_index = position_ids[b, data[b] == eod_token]
            # Detach indecies from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.clone()

            # Loop through EOD indecies:
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
                # Mask attention loss.
                if reset_attention_mask:
                    attention_mask[b, 0, (i + 1):, :(i + 1)] = 0
                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i + 1):] -= (i + 1 - prev_index)
                    prev_index = i + 1

    return attention_mask, loss_mask, position_ids


def get_two_batch(data, args):
    keys = ['text', 'target', 'loss_mask']
    datatype = torch.int64
    # Broadcast data.
    data_b = mpu.broadcast_data(keys, data, datatype)
    source_tokens = data_b['text'].long()
    target_tokens = data_b['target'].long()
    loss_mask = data_b['loss_mask'].float()
    labels = target_tokens[:, 1:].contiguous()
    loss_mask = loss_mask[:, 1:].contiguous()
    target_tokens = target_tokens[:, :-1].contiguous()
    _, _, source_position_ids = get_masks_and_position_ids(
        source_tokens,
        args.eod_token,
        reset_position_ids=False,
        reset_attention_mask=False,
        loss_mask=None,
        attention_mask=None,
        set_loss_mask=False)
    target_mask, _, target_position_ids = get_masks_and_position_ids(
        target_tokens,
        args.eod_token,
        reset_position_ids=False,
        reset_attention_mask=False,
        loss_mask=None,
        attention_mask=None,
        set_loss_mask=False)
    if args.fp16:
        target_mask = target_mask.half()
    return source_tokens, target_tokens, source_position_ids, target_position_ids, labels, target_mask, loss_mask


def get_batch(data, args):
    ''' get_batch subdivides the source data into chunks of
    length args.seq_length. If source is equal to the example
    output of the data loading example, with a seq_length limit
    of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘
    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the data loader. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM. A Variable representing an appropriate
    shard reset mask of the same dimensions is also returned.
    '''
    # Items and their type.
    keys = ['text', 'loss_mask']
    if args.transformer_xl or args.block_lm:
        keys += ['target', 'attention_mask']
    if args.block_lm:
        keys += ['position_id']
    datatype = torch.int64

    # Broadcast data.
    data_b = mpu.broadcast_data(keys, data, datatype)   #把数据广播到各个GPU
    # Unpack.
    if args.transformer_xl:
        tokens = data_b['text'].long()
        labels = data_b['target'].long()
        attention_mask = data_b['attention_mask'].float()
        loss_mask = data_b['loss_mask'].float()
    elif args.block_lm:
        tokens = data_b['text'].long()
        labels = data_b['target'].long()
        attention_mask = data_b['attention_mask'].long()
        loss_mask = data_b['loss_mask'].float()
        position_ids = data_b['position_id'].long()
    else:
        tokens_ = data_b['text'].long()
        loss_mask = data_b['loss_mask'].float()
        labels = tokens_[:, 1:].contiguous()
        loss_mask = loss_mask[:, 1:].contiguous()
        tokens = tokens_[:, :-1].contiguous()
        attention_mask = None

    # Get the masks and postition ids.
    if not args.block_lm:
        attention_mask, loss_mask, position_ids = get_masks_and_position_ids(
            tokens,
            args.eod_token,
            args.reset_position_ids,
            args.reset_attention_mask,
            loss_mask=loss_mask,
            attention_mask=attention_mask,
            mem_length=args.mem_length,
            set_loss_mask=not args.transformer_xl)
        # Convert
        if args.fp16:
            attention_mask = attention_mask.half()
    return tokens, labels, loss_mask, attention_mask, position_ids


tokenizer = None


def forward_step(data_iterator, model, args, timers, mems):
    """Forward step."""

    # Get the batch.
    timers('batch generator').start()
    timers('data loader').start()
    rand = random.Random(args.iteration * mpu.get_data_parallel_world_size() + mpu.get_data_parallel_rank())
    if data_iterator[1] and rand.random() < args.multi_task_ratio:
        data = next(data_iterator[1]) if data_iterator[1] else None
        data["mode"] = "multi-task"
    else:
        data = next(data_iterator[0]) if data_iterator[0] else None
    # print_rank_0("data iterator")
    timers('data loader').stop()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data, args)
    timers('batch generator').stop()

    # print_rank_0("get batch")

    def print_masked_text(batch_id):
        block_position_ids = position_ids[:, 1]
        position_ids_ = position_ids[:, 0]
        sep = attention_mask.item() if torch.numel(attention_mask) == 1 else attention_mask[batch_id].item()
        text, last_segment = "", []
        for i, token_id in enumerate(tokens[batch_id, :sep].tolist()):
            token = tokenizer.IdToToken(token_id)
            if token.startswith('[MASK') or token.endswith('MASK]'):
                if last_segment:
                    text += tokenizer.DecodeIds(last_segment)
                    last_segment = []
                text += f" [{position_ids_[batch_id, i].item()}, {token}]"
            else:
                last_segment.append(token_id)
        if last_segment:
            text += tokenizer.DecodeIds(last_segment)
        print(text.encode('utf-8'))
        last_index = None
        for i in range(sep, tokens.size(1)):
            if tokenizer.IdToToken(tokens[batch_id, i].item()).startswith("<|startofpiece"):
                if last_index is not None:
                    print(tokenizer.DecodeIds(tokens[batch_id, last_index: i].tolist()).encode('utf-8'), "|",
                          tokenizer.DecodeIds(labels[batch_id, last_index: i].tolist()).encode('utf-8'),
                          position_ids_[batch_id, last_index: i].tolist(),
                          block_position_ids[batch_id, last_index:i].tolist())
                last_index = i
        if last_index is not None:
            print(tokenizer.DecodeIds(tokens[batch_id, last_index:].tolist()).encode('utf-8'), "|",
                  tokenizer.DecodeIds(labels[batch_id, last_index:].tolist()).encode('utf-8'),
                  position_ids_[batch_id, last_index:].tolist(), block_position_ids[batch_id, last_index:].tolist())

    if data is not None and "mode" in data:
        mode = data['mode']
    else:
        mode = 'bert'

    logits, *mems = model(tokens, position_ids, attention_mask, *mems)
    losses = mpu.vocab_parallel_cross_entropy(logits.contiguous().float(),
                                              labels)
    loss_mask = loss_mask.view(-1)
    loss = torch.sum(losses.view(-1) * loss_mask)
    if loss_mask.sum().item() > 0:
        loss = loss / loss_mask.sum()

    return loss, mems, mode


def report_iteration_metrics(summary_writer, optimizer, lr, loss, elapsed_time, step, total_step, args):
    log_string = ' iteration {:8d}/{:8d} |'.format(step, total_step)
    log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(elapsed_time)
    log_string += ' learning rate {:.3E} |'.format(lr)
    log_string += ' lm loss {:.6E} |'.format(loss)
    if args.fp16:
        log_string += ' loss scale {:.1f} |'.format(
            optimizer.cur_scale if args.deepspeed else optimizer.loss_scale)
    print_rank_0(log_string)
    if summary_writer is not None:
        summary_writer.add_scalar(f'Train/lr', lr, step)
        summary_writer.add_scalar(f'Train/train_loss', loss, step)
        summary_writer.add_scalar(f'Train/elapsed_time', elapsed_time, step)


def report_evaluate_metrics(summary_writer, prefix, loss, ppl, gpt_loss, bert_loss, sent_loss, multi_loss, step):
    string = ' validation loss at {}'.format(prefix)
    string += ' | LM loss: {:.6E}'.format(loss)
    string += ' | LM PPL: {:.6E}'.format(ppl)
    if gpt_loss != 0:
        string += ' | GPT loss: {:.6E}'.format(gpt_loss)
    if bert_loss != 0:
        string += ' | BERT loss: {:.6E}'.format(bert_loss)
    if sent_loss != 0:
        string += ' | Sent loss: {:.6E}'.format(sent_loss)
    if multi_loss != 0:
        string += ' | Multi loss: {:.6E}'.format(multi_loss)
    length = len(string) + 1
    print_rank_0('-' * 100)
    print_rank_0('-' * length)
    print_rank_0(string)
    print_rank_0('-' * length)
    if summary_writer is not None:
        summary_writer.add_scalar(f'Train/valid_ppl', ppl, step)
        summary_writer.add_scalar(f'Train/valid_loss', loss, step)
        if gpt_loss != 0:
            summary_writer.add_scalar(f'Train/valid_gpt_loss', gpt_loss, step)
        if bert_loss != 0:
            summary_writer.add_scalar(f'Train/valid_bert_loss', bert_loss, step)
        if sent_loss != 0:
            summary_writer.add_scalar(f'Train/valid_sent_loss', sent_loss, step)
        if multi_loss != 0:
            summary_writer.add_scalar(f'Train/valid_multi_loss', multi_loss, step)


def train(model, optimizer, lr_scheduler,
          train_data_iterator, val_data_iterator, timers, args, summary_writer=None):
    """Train the model."""

    # Turn on training mode which enables dropout.
    model.train()

    # Tracking loss.
    total_lm_loss = 0.0

    # Iterations.
    skipped_iters = 0

    timers('interval time').start()
    report_memory_flag = True
    mems = []
    while args.iteration < args.train_iters:

        lm_loss, skipped_iter, mems = train_step(train_data_iterator,
                                                 model,
                                                 optimizer,
                                                 lr_scheduler,
                                                 args, timers, mems=mems, forward_step_func=forward_step)
        skipped_iters += skipped_iter
        args.iteration += 1

        # Update losses.
        total_lm_loss += lm_loss.data.detach().float()

        # Logging.
        if args.iteration % args.log_interval == 0:
            learning_rate = optimizer.param_groups[0]['lr']
            avg_lm_loss = total_lm_loss.item() / args.log_interval
            elapsed_time = timers('interval time').elapsed()
            report_iteration_metrics(summary_writer, optimizer, learning_rate, avg_lm_loss,
                                     elapsed_time * 1000.0 / args.log_interval, args.iteration, args.train_iters, args)
            total_lm_loss = 0.0
            if report_memory_flag:
                report_memory('after {} iterations'.format(args.iteration))
                report_memory_flag = False
            # for i in range(torch.distributed.get_world_size()):
            #     if i == torch.distributed.get_rank():
            #         print(get_hostname())
            #         timers.log(['forward', 'backward', 'optimizer',
            #                     'batch generator', 'data loader'],
            #                    normalizer=args.log_interval, reset=False)
            #     torch.distributed.barrier()
            if args.deepspeed or args.DDP_impl == 'torch':
                timers.log(['forward', 'backward', 'optimizer',
                            'batch generator', 'data loader'],
                           normalizer=args.log_interval)
            else:
                timers.log(['forward', 'backward', 'allreduce', 'optimizer',
                            'batch generator', 'data loader'],
                           normalizer=args.log_interval)
        # Checkpointing
        if args.save and args.save_interval and args.iteration % args.save_interval == 0:
            save_checkpoint(args.iteration, model, optimizer, lr_scheduler, args)

        # Evaluation
        if args.eval_interval and args.iteration % args.eval_interval == 0 and args.do_valid:
            prefix = 'iteration {}'.format(args.iteration)
            evaluate_and_print_results(
                prefix, val_data_iterator, model, args, timers, verbose=False, step=args.iteration,
                summary_writer=summary_writer, forward_step_func=forward_step)

    return args.iteration, skipped_iters


def evaluate(data_iterator, model, args, timers, forward_step_func, verbose=False):
    """Evaluation."""
    # Turn on evaluation mode which disables dropout.
    model.eval()

    total_lm_loss, total_gpt_loss, total_bert_loss, total_sent_loss, total_multi_loss = 0, 0, 0, 0, 0
    gpt_iters, bert_iters, sent_iters, multi_iters = 0, 0, 0, 0
    mems = []
    with torch.no_grad():
        iteration = 0
        while iteration < args.eval_iters:
            iteration += 1
            if verbose and iteration % args.log_interval == 0:
                print_rank_0('Evaluating iter {}/{}'.format(iteration, args.eval_iters))
            # Forward evaluation.
            lm_loss, mems, mode = forward_step_func(data_iterator, model, args, timers, mems=mems)

            '''when contiguous memory optimizations are enabled, the buffers
            allocated by the optimizations are deallocated during backward pass
            in the absence of backward pass the buffers should be reset after each
            forward pass'''
            if args.deepspeed and args.deepspeed_activation_checkpointing:
                deepspeed.checkpointing.reset()

            lm_loss = lm_loss.data.detach().float().item()
            total_lm_loss += lm_loss
            if mode == 'gpt':
                total_gpt_loss += lm_loss
                gpt_iters += 1
            elif mode == 'bert':
                total_bert_loss += lm_loss
                bert_iters += 1
            elif mode == 'sentence':
                total_sent_loss += lm_loss
                sent_iters += 1
            elif mode == 'multi-task':
                total_multi_loss += lm_loss
                multi_iters += 1
    # Move model back to the train mode.
    model.train()
    # Reduce across processes.
    loss_data = torch.cuda.FloatTensor(
        [total_lm_loss, total_gpt_loss, total_bert_loss, total_sent_loss, total_multi_loss, gpt_iters, bert_iters,
         sent_iters, multi_iters])
    torch.distributed.all_reduce(loss_data, group=mpu.get_data_parallel_group())
    loss_data = loss_data.tolist()
    total_lm_loss = loss_data[0] / args.eval_iters / (args.world_size / args.model_parallel_size)
    total_gpt_loss = loss_data[1] / loss_data[5] if loss_data[5] > 0 else 0
    total_bert_loss = loss_data[2] / loss_data[6] if loss_data[6] > 0 else 0
    total_sent_loss = loss_data[3] / loss_data[7] if loss_data[7] > 0 else 0
    total_multi_loss = loss_data[4] / loss_data[8] if loss_data[8] > 0 else 0
    return total_lm_loss, total_gpt_loss, total_bert_loss, total_sent_loss, total_multi_loss


def evaluate_and_print_results(prefix, data_iterator, model,
                               args, timers, forward_step_func, verbose=False, step=None, summary_writer=None):
    """Helper function to evaluate and dump results on screen."""
    lm_loss, gpt_loss, bert_loss, sent_loss, multi_loss = evaluate(data_iterator, model, args, timers, verbose=verbose,
                                                                   forward_step_func=forward_step_func)

    lm_ppl = math.exp(min(20, lm_loss))
    report_evaluate_metrics(summary_writer, prefix, lm_loss, lm_ppl, gpt_loss, bert_loss, sent_loss, multi_loss, step)

    return lm_loss


'''
    Optional DeepSpeed Activation Checkpointing features
    Gives access to partition activations, contiguous memory optimizations
    and cpu checkpointing.

    Activation checkpoint requires keep track of the random states
    and setting the random seed for each MP process. Megatron uses
    mpu.get_cuda_rng_tracker and mpu.model_parallel_cuda_manual_seed
    for keeping track of the random states and setting the random seeds.
    Since they are used in places outside of activation checkpointing,
    we overwrite them to maintain consistency.

    This must be done before all the calls to mpu.model_parallel_cuda_manual_seed
    '''


def set_deepspeed_activation_checkpointing(args):
    deepspeed.checkpointing.configure(mpu, deepspeed_config=args.deepspeed_config, num_checkpoints=args.num_layers)
    mpu.checkpoint = deepspeed.checkpointing.checkpoint
    mpu.get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
    mpu.model_parallel_cuda_manual_seed = deepspeed.checkpointing.model_parallel_cuda_manual_seed


def initialize_distributed(args):
    """Initialize torch.distributed."""

    # Manually set the device ids. #收到设置设备id
    device = args.rank % torch.cuda.device_count()  #torch.cuda.device_count，获取当前计算机上可用的CUDA（Compute Unified Device Architecture）设备的数量
    if args.local_rank is not None:
        device = args.local_rank
    torch.cuda.set_device(device) #选择在那个GPU上执行后续的深度学习操作
    # Call the init process 当说"Call the init process"时，通常是指执行一个操作，以使系统重新启动或重新初始化
    init_method = 'tcp://'
    args.master_ip = os.getenv('MASTER_ADDR', 'localhost') #获取名称为"MASTER_ADDR"的环境变量的值，如果没有则默认为localhost
    args.master_port = os.getenv('MASTER_PORT', '6000')    #获取名称为“MASTER_PORT”的环境变量的值，如果没有则默认为6000
    init_method += args.master_ip + ':' + args.master_port  # tcp://localhost:6000
    if hasattr(deepspeed, "init_distributed"): #检查名为deepspeed的模块或对象是否具有名为init_distributed的属性或方法
        deepspeed.init_distributed(dist_backend=args.distributed_backend) #初始化分布式训练环境，dist_backend='nccl'，分布式后端
    else:
        torch.distributed.init_process_group(
            backend=args.distributed_backend,
            world_size=args.world_size, rank=args.rank,
            init_method=init_method)

    # Set the model-parallel / data-parallel communicators.
    mpu.initialize_model_parallel(args.model_parallel_size)

    # Optional DeepSpeed Activation Checkpointing Features
    #
    if hasattr(args, "deepspeed") and args.deepspeed and args.deepspeed_activation_checkpointing:
        set_deepspeed_activation_checkpointing(args)


def set_random_seed(seed):
    """Set random seed for reproducability."""

    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        mpu.model_parallel_cuda_manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def get_train_val_test_data(args, tokenizer):
    """Load the data on rank zero and boradcast number of tokens to all GPUS.
    这里应该是说数据是有级别的，将来会被最先利用的数据级别为0，优先将这类数据加载到GPU中
    """

    (train_data, val_data, test_data) = (None, None, None)
    # Data loader only on rank 0 of each model parallel group.
    if mpu.get_model_parallel_rank() == 0:
        data_config = configure_data()
        if args.block_lm:
            data_set_type = "Block"
        elif args.transformer_xl:
            data_set_type = "GPT-XL"
        else:
            data_set_type = "GPT2"
        data_config.set_defaults(data_set_type=data_set_type, transpose=False)
        train_data, val_data, test_data = data_config.apply(args, tokenizer)  #data_config.apply 会返回 dataloader

        data_counts = torch.cuda.LongTensor([int(args.do_train), int(args.do_valid), int(args.do_test)])
    else:
        data_counts = torch.cuda.LongTensor([0, 0, 0])

    # Broadcast num tokens.
    torch.distributed.broadcast(data_counts,
                                mpu.get_model_parallel_src_rank(),
                                group=mpu.get_model_parallel_group())
    args.do_train = data_counts[0].item()
    args.do_valid = data_counts[1].item()
    args.do_test = data_counts[2].item()

    return train_data, val_data, test_data


def main():
    """Main training program."""

    # Disable CuDNN.
    torch.backends.cudnn.enabled = False #禁用CuDNN
    # Timer.
    timers = Timers()

    # Arguments.
    args = get_args()
    args.mem_length = args.mem_length if args.transformer_xl else 0  #如果transformer==True，mem_length为args的值，否则为0
    if args.load and not args.new_save_directory:   #如果load 并且 new_save_directory不为空
        args.experiment_name = os.path.basename(os.path.normpath(args.load))   #experiment_name 等于load目录的最后一级id
    else:
        args.experiment_name = args.experiment_name + datetime.now().strftime("%m-%d-%H-%M")  #experiment_name 等于 args设置的默认id + 时间(年月日分钟)
    if args.save:
        args.save = os.path.join(args.save, args.experiment_name)  #如果设置的有save路径,存储路径设置为save+实验名称
    # Pytorch distributed.
    initialize_distributed(args)   #GPU分配，模型模型并行化，deepspeed相关

    # Random seeds for reproducability. 用于重现性的随机种子
    set_random_seed(args.seed)  #固定训练过程中生成的随机数，使得实验具有重现性

    # Data stuff.  数据内容
    global tokenizer
    tokenizer = prepare_tokenizer(args)  #设置分词器，并且分词器具有编码文本的功能
    
    train_data, val_data, test_data, = get_train_val_test_data(args, tokenizer)  #获取训练，验证，测试数据集，搞不懂数据是从哪里加载出来的
    
     
    
    print(train_data)
    for i,item in enumerate(train_data):
        if i <= 0:
            dit = {}
            print(type(item))
            text = item['text'].tolist()
            for seq in text:
                str_seq = ""
                tokens = tokenizer.DecodeAsTokens(seq)
                #for token in tokens:
                #   str_seq += token
                print(tokens)
            target = item['target']
            loss_mask = item['loss_mask']
            position_id = item['position_id']
            attention_mask = item['attention_mask']
            dit['text']=text
            dit['target']=target.tolist()
            dit['loss_mask']=loss_mask.tolist()
            dit['position_id']=position_id.tolist()
            dit['attention_mask']=attention_mask.tolist()
            print(text)
            print(dit)
   



    

    

if __name__ == "__main__":
    main()
