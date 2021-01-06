import os
from typing import Dict, List

import conllu

import torch
from torch import tensor, cuda
from torch.utils.data import Dataset


# Whether to train on a gpu
train_on_gpu = cuda.is_available()
print(f'Train on gpu: {train_on_gpu}')


# Number of gpus
if train_on_gpu:
    gpu_count = cuda.device_count()
    print(f'{gpu_count} gpus detected.')
    if gpu_count > 1:
        multi_gpu = True
    else:
        multi_gpu = False
else:
    multi_gpu = None

# Parameters
if train_on_gpu:
    params = {
        'batch_size': 8,
        # 'shuffle': True,
        'num_workers': 6
    }
else:
    params = {
        'batch_size': 2,
    }


class Args:
    def __init__(self, maxlen=512, list_deprel_main=[], list_deprel_aux=[], list_pos=[], split_deprel=False, punct=False, mode="train"):
        self.maxlen = maxlen
        self.list_deprel_main = list_deprel_main
        self.list_deprel_aux = list_deprel_aux
        self.list_pos = list_pos
        self.split_deprel = split_deprel
        self.punct = punct
        self.mode = mode


def dep_parse_data_collator(features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    batch: Dict[str, torch.Tensor] = {}
    batch['input_ids'] = torch.stack([f[0] for f in features])
    batch['attention_mask'] = torch.stack([f[2] for f in features])

    subwords_start = torch.stack([f[1] for f in features])
    idx_convertor = torch.stack([f[3] for f in features])
    poss = torch.stack([f[4] for f in features])
    heads = torch.stack([f[5] for f in features])
    deprels_main = torch.stack([f[6] for f in features])
    # print("KK subwords_start", subwords_start.size())
    # print("KK idx_convertor", idx_convertor.size())
    batch['labels'] = torch.stack(
        [subwords_start, idx_convertor, poss, heads, deprels_main], dim=1)

    # print("KK labels.size(", batch["labels"].size())
    # input("input")
    return batch


class ConlluDataset(Dataset):
    def __init__(self, path_folder, tokenizer, args: Args) -> None:
        self.tokenizer = tokenizer
        self.args = args

        # self.separate_deprel = args.separate_deprel
        self.separate_deprel = True

        self.CLS_token_id = tokenizer.cls_token_id
        self.SEP_token_id = tokenizer.sep_token_id

        # Load all the sequences from the file
        # TODO : make a generator
        self.sequences = []
        for conllu_name in os.listdir(path_folder):
            path_conllu = os.path.join(path_folder, conllu_name)
            with open(path_conllu, 'r') as infile:
                # TODO : add a checker for conllu well formating
                self.sequences += conllu.parse(infile.read())
        
        self.drm2i, self.i2drm = self._mount_dr2i(self.args.list_deprel_main)

        self.pos2i, self.i2pos = self._mount_pos2i(self.args.list_pos)

        self.n_labels_main = len(self.drm2i)

        if self.args.split_deprel:
            self.dra2i, self.i2dra = self._mount_dr2i(
                self.args.list_deprel_aux)
            self.n_labels_aux = len(self.dra2i)

    def __len__(self):
        return(len(self.sequences))

    def __getitem__(self, index):
        return(self._get_processed(self.sequences[index]))

    def _mount_dr2i(self, list_deprel):
        i2dr = {}
        dr2i = {}

        for idx, deprel in enumerate(list_deprel):
            i2dr[idx] = deprel
            dr2i[deprel] = idx

        return dr2i, i2dr

    def _mount_pos2i(self, list_pos):
        # list_pos = []
        # for sequence in self.sequences:
        #     for token in sequence:
        #         list_pos.append(token['upostag'])
        sorted_set_pos = sorted(set(list_pos))

        pos2i = {}
        i2pos = {}

        for i, pos in enumerate(sorted_set_pos):
            pos2i[pos] = i
            i2pos[i] = pos

        self.list_pos = sorted_set_pos

        return pos2i, i2pos

    def _pad_list(self, l, padding_value):
        if len(l) > self.args.maxlen:
            print(l, len(l))
            raise Exception(
                "The sequence is bigger than the size of the tensor")

        return l + [padding_value]*(self.args.maxlen-len(l))

    def _trunc(self, tensor):
        if len(tensor) >= self.args.maxlen:
            tensor = tensor[:self.args.maxlen-1]

        return tensor

    def _get_input(self, sequence):
        sequence_ids = [self.CLS_token_id]
        subwords_start = [-1]
        idx_convertor = [0]
        tokens_len = [1]

        for token in sequence:
            if type(token['id']) != int:
                # print(token['id'])
                continue

            form = ""
            # if self.args.increment_unicode:
            #     for character in token['form']:
            #         form += chr(ord(character) + 3000)
            # else:
            form = token['form']
            token_ids = self.tokenizer.encode(form, add_special_tokens=False)
            idx_convertor.append(len(sequence_ids))
            tokens_len.append(len(token_ids))
            subword_start = [1] + [0] * (len(token_ids)-1)

            sequence_ids += token_ids
            subwords_start += subword_start

        sequence_ids = self._trunc(sequence_ids)
        subwords_start = self._trunc(subwords_start)
        idx_convertor = self._trunc(idx_convertor)

        sequence_ids = sequence_ids + [self.SEP_token_id]

        sequence_ids = tensor(self._pad_list(sequence_ids, 0))
        subwords_start = tensor(self._pad_list(subwords_start, -1))
        idx_convertor = tensor(self._pad_list(idx_convertor, -1))
        attn_masks = tensor([int(token_id > 0) for token_id in sequence_ids])

        return sequence_ids, subwords_start, attn_masks, idx_convertor, tokens_len

    def _get_output(self, sequence, tokens_len):
        poss = [-1]
        heads = [-1]
        deprels_main = [-1]
        deprels_aux = [-1]
        skipped_tokens = 0
        for n_token, token in enumerate(sequence):
            if type(token['id']) != int:
                # print(token['id'])
                skipped_tokens += 1
                continue

            # if len(tokens_len) == n_token+1:
            #     print("sequence", sequence)
            #     print("tokens_len", tokens_len)
            token_len = tokens_len[n_token + 1 - skipped_tokens]

            pos = [self.pos2i.get(
                token['upostag'], self.pos2i['none'])] + [-1]*(token_len-1)
            head = [sum(tokens_len[:token['head']])] + [-1]*(token_len-1)
            deprel_main, deprel_aux = normalize_deprel(
                token['deprel'], split_deprel=self.args.split_deprel)
            deprel_main = [self.drm2i.get(
                deprel_main, self.drm2i['none'])] + [-1]*(token_len-1)

            poss += pos
            heads += head
            deprels_main += deprel_main

            if self.args.split_deprel:
                deprel_aux = [self.dra2i.get(
                    deprel_aux, self.dra2i['none'])] + [-1]*(token_len-1)
                deprels_aux += deprel_aux
        # try:
        # except:
        #     print(sequence)
        #     print(sequence.metadata)
        heads = self._trunc(heads)
        deprels_main = self._trunc(deprels_main)
        poss = self._trunc(poss)

        poss = tensor(self._pad_list(poss, -1))
        heads = tensor(self._pad_list(heads, -1))
        deprels_main = tensor(self._pad_list(deprels_main, -1))

        heads[heads == -1] = self.args.maxlen - 1
        heads[heads >= self.args.maxlen-1] = self.args.maxlen - 1

        if self.args.split_deprel:
            deprel_aux = self._trunc(deprel_aux)
            deprels_aux = tensor(self._pad_list(deprels_aux, -1))

        if not self.args.punct:
            is_punc_tensor = [deprels_main == self.drm2i["punct"]]
            heads[is_punc_tensor] = self.args.maxlen - 1
            deprels_main[is_punc_tensor] = -1

            if self.args.split_deprel:
                deprels_aux[is_punc_tensor] = -1

        if not self.args.split_deprel:
            deprels_aux = deprels_main.clone()

        return poss, heads, deprels_main, deprels_aux

    def _get_processed(self, sequence):
        sequence_ids, subwords_start, attn_masks, idx_convertor, token_lens = self._get_input(
            sequence)

        if self.args.mode == 'predict':
            return sequence_ids, subwords_start, attn_masks, idx_convertor

        else:
            poss, heads, deprels_main, deprels_aux = self._get_output(
                sequence, token_lens)

            return sequence_ids, subwords_start, attn_masks, idx_convertor, poss, heads, deprels_main, deprels_aux


def normalize_deprel(deprel, split_deprel):
    # change for taking only before @
    deprel = deprel.replace("@", ":")
    if split_deprel:
        deprels = deprel.split(":")
        deprel_main = deprels[0]
        if len(deprels) > 1:
            deprel_aux = deprels[1]
        else:
            deprel_aux = 'none'

        return deprel_main, deprel_aux

    else:
        return deprel, 'none'


def create_deprel_lists(*paths, split_deprel):
    for path in paths:
        with open(path, "r", encoding="utf-8") as infile:
            result = conllu.parse(infile.read())

        list_deprel_main = []
        list_deprel_aux = []
        for sequence in result:
            for token in sequence:
                deprel_main, deprel_aux = normalize_deprel(
                    token['deprel'], split_deprel=split_deprel)
                list_deprel_main.append(deprel_main)
                list_deprel_aux.append(deprel_aux)

    list_deprel_main.append('none')
    list_deprel_aux.append('none')
    list_deprel_main = sorted(set(list_deprel_main))
    list_deprel_aux = sorted(set(list_deprel_aux))
    return list_deprel_main, list_deprel_aux


def create_pos_list(*paths):
    list_pos = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as infile:
            result = conllu.parse(infile.read())

        for sequence in result:
            for token in sequence:
                list_pos.append(token['upostag'])
    list_pos.append('none')
    list_pos = sorted(set(list_pos))
    return list_pos
