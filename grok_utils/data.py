import os
import blobfile as bf
import itertools
from torch import Tensor, LongTensor
from typing import Tuple, List, Dict, Any, Union, Optional
import random
import sys
import math
import torch
import numpy as np

#Set Global variables

VALID_OPERATORS={
    "+": "addition",
    "-": "subtraction",
    "*": "muliplication",
    "/": "division",
    "**2+": "squarepoly",
    "**3+": "cubepoly",
    "x**2+y**2_mod_97": "quad1",
    "x**2+y**2+x*y_mod_97": "quad2",
    "x**2+y**2+x*y+x_mod_97": "quad3",
    "x**3+x*y_mod_97": "cube1",
    "x**3+x*y**2+y_mod_97": "cube2",
    "(x._value//y)if(y._value%2==1)else(x-y)_mod_97": "mix1",
    "s5": "s5",
    "s5conj": "s5conj",
    "s5aba": "s5aba",
    "+*": "even-addition_odd-multiplication",
    "+-": "even-addition_odd-subtraction",
    "sort": "sort",
    "reverse": "reverse",
    "copy": "copy",
}


EOS = "<|eos|>"
EQ_TOKEN = "="
MODULUS = 97
NUMS = list(range(MODULUS))

DEFAULT_DATA_DIR = "data"


class ArithmeticTokenizer:
    """Stores the list of token text to token id mappings and converts between them"""

    token_file= "tokens.txt"

    def __init__(self, data_dir=DEFAULT_DATA_DIR)->None:
        self.token_file=bf.join(data_dir, self.token_file)
        self.itos = self.get_tokens()
        self.stoi : Dict[str, int] = dict([(s, i) for i, s in enumerate(self.itos)])

    @classmethod
    def get_tokens(cls):
        tokens = (
                [EOS, EQ_TOKEN]
                + list(sorted(list(VALID_OPERATORS.keys())))
                + list(map(str, NUMS))
        )
        return tokens

    def __len__(self) -> int:
        """returns the vocabulary size"""
        return len(self.itos)

    def _encode(self, obj:str)-> Tensor:
        print([self.stoi[t] for t in obj.split(" ")])
        return LongTensor([self.stoi[t] for t in obj.split(" ")])

    def encode(self, obj) -> Tensor:
        """
        Convert a string of text into a rank-1 tensor of token ids
        or convert a list of strings of text into a rank-2 tensor of token ids

        :param obj: the string or list of strings to convert
        :returns: a tensor of the token ids
        """
        if isinstance(obj, str):
            return self._encode(obj)
        elif isinstance(obj, list):
            return torch.stack([self._encode(s) for s in obj], dim=0)
        else:
            raise NotImplementedError
    
    def decode(self, tensor):
        """
        Converts a tensor of token ids into a string of text

        :param tensor: a tensor of the token ids
        :returns: string of these tokens.
        """

        return " ".join([self.itos[i] for i in tensor.long()])

class ArithmeticDataset:
    """Data set of arithmetic equations"""
    def __init__(self,name, data, train, data_dir) -> None:
        self.tokenizer = ArithmeticTokenizer(data_dir)
        self.name = name
        self.train = train
        if isinstance(data, list):
            self.data = self.tokenizer.encode(data)
        else:
            self.data = data
        

    @classmethod
    def _make_operation_data(cls, operator: str) -> list[str]:
        
        #Calculates the cartesian product of the set of numbers by itself
        operands = NUMS
        tuples = itertools.product(operands, repeat=2)
        
        #Calculte the result of the operation
        eq = []
        for a,b in tuples:
            
            #Division
            if operator == '/':
                if b==0:
                    continue
                else:
                    c=a
                    a=(b*c) % MODULUS

            #addition
            elif operator == '+':
                c=(a+b) % MODULUS
            

            eq.append(f'{a} {operator} {b} = {c}')
            

        return eq

    @classmethod
    def make_data(cls, operator: str, shuffle=True, seed=42) -> list[str]:
        
        assert operator in VALID_OPERATORS
        rng = np.random.RandomState(seed=seed)
        data= cls._make_operation_data(operator)
        if shuffle:
            rng.shuffle(data)
        return [EOS + " " + eq + " " + EOS for eq in data]
        
    @classmethod
    def splits(cls, train_pct:float, operator:str, data_dir:str= DEFAULT_DATA_DIR, tr_in_context:int=0, val_in_context:int=0):
        """
        Creates the validation and training datasets:

        :param tr_in_context: the number of in context examples in each equation in the training dataset
        :param val_in_context: the number of in context examples in each equation in the validation dataset
        :param operator: the operator of the equations
        :param train_pct: percentage of total equations used for the training set (between 0 and 1)
        :param data_dir: the output data dir
        :returns: (train_dataset, validation_dataset)
        """
        assert (0<train_pct) and (train_pct<1)
        
        eq=cls.make_data(operator)
        ds_name = VALID_OPERATORS[operator]
        train_rows = round(len(eq) * train_pct)
        
        tr_eq = eq[:train_rows]
        val_eq = eq[train_rows:]
        
        val_ds = []
        if val_in_context > 0:
            for i in range(len(val_eq)):
                random_samples = " ".join(random.sample(tr_eq, tr_in_context))
                val_ds.append(random_samples + " " + val_eq[i])
        else:
            val_ds=val_eq

        tr_ds = []
        if tr_in_context > 0:
            for i in range(len(tr_eq)):
                random_samples = random.sample(tr_eq, tr_in_context)
                random_samples = " ".join(random_samples)
                tr_ds.append(random_samples + " " + tr_eq[i])

                print("----------- eq " + str(i)+"------------------")
                print(random_samples)
                print(tr_ds[i])
        else:
            tr_ds=tr_eq

        train_ds = cls(ds_name, tr_ds, train = True, data_dir = data_dir)
        val_ds = cls(ds_name, val_ds, train = False, data_dir = data_dir)

        return train_ds, val_ds

    def __len__(self) -> int:
        """
        :returns: total number of equations in this dataset
        """
        return self.data.shape[0]

class ArithmeticIterator(torch.utils.data.IterableDataset):
    """
    An iterator over batches of data in an ArithmeticDataset
    """

    def __init__(
        self,
        dataset: ArithmeticDataset,
        device: torch.device,
        shuffle: bool = True,
    ) -> None:
        """
        :param dataset: the dataset to iterate over
        :param device: the torch device to send batches to
        :param shuffle: whether or not to randomly shuffle the dataset
        """
        self.dataset = dataset
        self.batchsize = min(512, math.ceil(len(dataset) / 2.0)) 
        self.device = device
        self.reset_iteration(shuffle=shuffle)


    def reset_iteration(self, shuffle=True):
        self.index = 0
        if shuffle and self.dataset.train:
            self.permutation = torch.randperm(len(self.dataset))
        else:
            self.permutation = torch.arange(len(self.dataset))

    def __iter__(self):
        """
        :returns: this iterator
        """
        return self

    def __next__(self) -> Dict[str, Tensor]:
        """
        Returns one batch of data.

        :raises: StopIteration when we're out of data
        :returns: batch tensor of shape (self.batchsize, tokens_per_eq)
        """

        batch_begin = self.index * self.batchsize
        if batch_begin > len(self.dataset) - 1:
            self.reset_iteration()
            raise StopIteration
        indices = self.permutation[batch_begin : batch_begin + self.batchsize]
        text = self.dataset.data[indices, :-1]
        target = self.dataset.data[indices, 1:]
        batch = {"text": text.to(self.device), "target": target.to(self.device)}
        self.index += 1
        return batch

    def __len__(self) -> int:
        """
        :returns: the total number of batches
        """
        return math.ceil(len(self.dataset) / self.batchsize)
