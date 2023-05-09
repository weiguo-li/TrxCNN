# data is stored in format data.fastq/gene/genes_name/trascripts_name/fastq
# only 4 element "A", "G", "C", "T"
# currently there are 12340 trascripts (转录体)
# time: 2 GB file is saved within aobut 1 minute and the whole gene or transcripts are about 18 GB

import torch 
import pandas as pd
from torch.utils.data import Dataset
from torchtext.vocab import vocab,Vocab
from collections import Counter, OrderedDict

import torch.nn.functional as F 

# origin_data_path = "../data.fastq/gene/"
# new_DataPath = "../Data/fastq.csv"
new_DataPath = "../Data_full/fastq.csv"




save_d__vocab_path = "../Data_full/"


Deoxynucleotide_ACID_vocab = ['A','T','C','G']


def _map_sequence_to_int(seq_item,vocab):
    """
        Args:
            seq_item: str
            vocab : Vocab 

        Return: list of numbericial representation of the tokens in the sequence
    """

    sequence = [c for c in seq_item]
    sequence.pop() # remove the "\n" in the end 

    return vocab.lookup_indices(sequence)


def _map_labels_to_int(label_item,vocab):
    """
    Args:
        label_item : str


    Return: int 
    """

    return vocab.__getitem__(label_item)


# define the Fastq Dataset class 
class FastqDataset(Dataset):
    
    def __init__(self,path_to_fastq_csv = new_DataPath) -> None:
        """
            reading in the raw data and preprocessing it as needed
            ### we do not need to store or process the whole data, what we need to do is to be able to process one sample at a time 

        """
        # super().__init__()
        self.path_to_fastq_csv = path_to_fastq_csv
        self.raw_data = pd.read_csv(path_to_fastq_csv,header=None)  #LOCAL variable 
        self.make_vocab()
        # self.preprocessed_data = None
        # self.preprocessed_data = self._preprocess_data(raw_data) 

    def __len__(self):
        pass
        return len(self.raw_data)

    def __getitem__(self, index) :
        """
        The feature will be a tensor representing the input sequence of tokens, 
        and the label will be a tensor representing the output label(s). 

        pytorch and tensorflow have different input size to conv1d() and sometimes we need to use tensor.prmute()
        to make it suitable for conv1d 
        I do such thing in collate function : my_collaote_fun in utile.py
        """

        feature = self.raw_data.iloc[index,0]

        label = self.raw_data.iloc[index,1]

        # More costomizable 
        # add the length to it and return it 
        length = len(feature) -1  # -1 because of "\n"

        feature = _map_sequence_to_int(feature,self.Deoxynucleotide_vocab)
        label = _map_labels_to_int(label,self.label_vocab)
        feature = torch.tensor(feature)
        label = torch.tensor(label)
        # return self.preprocessed_data[index]
        return F.one_hot(feature,num_classes = 4), label, length

    

    def make_vocab(self):
        
        labels = set(self.raw_data[1].values)
        self.label_vocab = None

        # to make the label a static file so that model will not be changed from time to time
        try:
            self.label_vocab = torch.load(save_d__vocab_path + "label_vocab.pt")
            print("vocabulary loaded sucessuful from the existing vocab")
        except FileNotFoundError:
            labels = set(self.raw_data[1].values)
            self.label_vocab = vocab(OrderedDict([(token , 1) for token in labels]))
            torch.save(self.label_vocab,save_d__vocab_path + "label_vocab.pt")
            print("a vocab was saved succussfully. ")
        else:
            pass

        self.Deoxynucleotide_vocab = vocab(OrderedDict([(token,1) for token in Deoxynucleotide_ACID_vocab]))


    def get_vocab(self):
        return self.label_vocab

    # def _preprocess_data(self):
    #     """
    #     Args:
    #         self.raw_data: Dataframe

    #     get the raw data from the .csv file and make tokenization to get its corresponding numericial representation
    #     and one-hot encoding to the original sequence 

    #     """
    #     # using Pytorch tensort to store the numerical data 
        

    #     return 




