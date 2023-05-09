from torch.nn.utils.rnn import pad_sequence
import torch
import torch.optim as optim
import hparams

Hyparams = hparams.hparams_set_train()


def my_collate_fn(batch):
    """
    Args:
        batch : list of data samples(tensors)
        here I achieve bucket sequenceing. 
        by divide batches in to 4 different sub batches
        so when you set the hyparamter batch size, it is better set it to the multiple of 4 
    """
    batch = sorted(batch, key=lambda x: len(x[0]))

    features = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    length = [item[2] for item in batch]

    subatch_size = int(len(batch) / 4) if len(batch) > 4 else len(batch)
  
    start_index = 0
    sub_padded_batches = []
    sub_labels = []
    sub_length = []

    while start_index < len(batch):
        end_index = min(start_index+subatch_size,len(batch))
        tmp_batch = features[start_index:end_index]
        if(len(tmp_batch) == 0):
            print("warninggggggggggggggggggg")
            print(start_index,end_index)
            print(f"batch size is {len(batch)}")
        padded_features = pad_sequence(
            tmp_batch, batch_first=True).permute(0, 2, 1).float()
        sub_padded_batches.append(padded_features)
        sub_labels.append(torch.tensor(labels[start_index:end_index]))
        sub_length.append(length[start_index:end_index])

        start_index = end_index

    # Perform Padding
    # paddding_value = 99 # this number must be greater than 1 (2,3,4, 5,)
    # padded_features = pad_sequence(features,batch_first=True).permute(0,2,1)  # 或者使用tensor.transpose(2,1)

    #
    # generate_mask
    # mask = (padded_features != 99).float()

    # sort to achieve buketing

    # return padded_features.float(), torch.tensor(labels), length
    return sub_padded_batches,sub_labels,sub_length

    # mask 不是静态的，它会随着forward 不停的变换尺寸

    # print(f"the type of batch is {type(batch)}")


def my_collate_fn2(batch):
    """
    Args:
        batch : list of data samples(tensors)
        here I achieve bucket sequenceing. 
        by divide batches in to 4 different sub batches
        so when you set the hyparamter batch size, it is better set it to the multiple of 4 
    """
    batch = sorted(batch, key=lambda x: len(x[0]))

    features = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    length = [item[2] for item in batch]


    # Perform Padding
    # paddding_value = 99 # this number must be greater than 1 (2,3,4, 5,)
    padded_features = pad_sequence(features,batch_first=True).permute(0,2,1)  # 或者使用tensor.transpose(2,1)

    #
    # generate_mask
    # mask = (padded_features != 99).float()

    # sort to achieve buketing

    return padded_features.float(), torch.tensor(labels), length



def lr_lamba(epoch):
    """
    warmp up and expenetial decay lambada function
    """
    # turn_point = int(0.2 * Num_epochs)
    turn_point = 1
    # debug
    # print(f"the epoch  in lr_lamba is {epoch}") #  this function will be called when I initialize the LambdaLR
    if (epoch <= turn_point):
        # warm up
        return epoch * (Hyparams["max_lr_rate"] - Hyparams["init_lr_rate"])/(turn_point)
    else:
        # expenetial decay
        return Hyparams["decay_rate"] ** (epoch - turn_point)


def lr_lamba_warmup(epoch):
    turn_point = int(0.3 * Hyparams["num_epochs"])
    epoch

    pass
