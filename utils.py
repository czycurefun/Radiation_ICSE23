import os
import random
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from params import param
import numpy as np


def read_data(file_path_dataset):
    return pd.read_csv(file_path_dataset, delimiter='\t')

def readDataFromFile(data_file):
    datas = []
    with open(data_file, 'r', encoding='utf-8') as reader:
        lines = reader.readlines()
        for line in lines:
            # pair = line.strip().split('\t')
            pair = line.strip().split('^^')
            datas.append(pair)
    return datas

def make_cuda(tensor):
    """Use CUDA if it's available."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def init_random_seed(manual_seed):
    """Init random seed."""
    if manual_seed is None:
        seed = random.randint(1, 10000)
    else:
        seed = manual_seed
    print("use random seed: {}".format(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_model(net, restore=None):

    # restore model weights
    if restore is not None and os.path.exists(restore):
        net.load_state_dict(torch.load(restore))
        print("Restore model from: {}".format(os.path.abspath(restore)))

    # check if cuda is available
    if torch.cuda.is_available():
        cudnn.benchmark = True
        net.cuda()
    return net


def save_model(net, filename):
    if not os.path.exists(param.model_root):
        os.makedirs(param.model_root)
    torch.save(net.state_dict(),
               os.path.join(param.model_root, filename))
    print("save pretrained model to: {}".format(os.path.join(param.model_root,
                                                             filename)))


def get_data_loader(sequences, labels, maxlen=None):
    # dataset and data loader
    text_dataset = TextDataset(sequences, labels, maxlen)

    text_data_loader = DataLoader(
        dataset=text_dataset,
        batch_size=param.batch_size,
        shuffle=False)

    return text_data_loader


class TextDataset(Dataset):
    def __init__(self, sequences, labels, maxlen):

        input_ids = list()
        attention_mask = list()
        token_type_ids = list()
        for each in sequences:
            input_ids.append(each["input_ids"])
            attention_mask.append(each['attention_mask'])
            token_type_ids.append(each['token_type_ids'])
        self.input_ids = torch.LongTensor(input_ids).cuda()
        self.attention_mask = torch.LongTensor(attention_mask).cuda()
        self.token_type_ids = torch.LongTensor(token_type_ids).cuda()
        self.labels = torch.LongTensor(labels).cuda()
        self.dataset_size = len(self.input_ids)

    def __getitem__(self, index):
        input_ids, attention_mask, token_type_ids, label = self.input_ids[index], self.attention_mask[index], self.token_type_ids[index], self.labels[index]
        return input_ids, attention_mask, token_type_ids, label

    def __len__(self):
        return self.dataset_size
