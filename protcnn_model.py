"""
Model architecture:

    
Predication Network 
        ^
        |
        |
    Embedding Network ----->{initial covolution ---> residual network ---> Pollinng along sequence dimen --> class prediction}
        ^                                               x N times
        |
        |
    Input Network
        

"""


import torch.nn as nn
import torch
import math


class Conv1d_with_mask(nn.Module):
    """
        ### build a convotion building block with mask 
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1) -> None:
        # super(Conv1d_with_mask,self).__init__()
        super().__init__()

        self.conv_layer = nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation, padding="same")

    def forward(self, x, length):  # mask 不好传进去，那就将(x,mask) 传进去,但这样也很难搞，所以修改residual block 部分，不用nn.sequence
        """
        length : list with length being batch_size
        """
        # 传mask 没用，得转length
        # x, mask = x 
        # first time zero_pad using mask --> element-wise multiplication

        mask = torch.ones_like(x)      # 猜想，第一次这里的卷积可做可不做，做了反而是多余的 因为能保证输入如果是padding都是0 的那么输出也都是满足padding部分都是0
        for i , len_of_seq in enumerate(length):
            mask[i][:,len_of_seq:] = 0                   # the i_th sequence in the batch

        with torch.no_grad():
            x = x * mask   # with torch.no_grad() ? 需要吗

        x = self.conv_layer(x)   # 找到了bug 忘了赋值

        # second time mask also called  re-zero padding (a technique)  这个techinique 跟paper里面所要表示的可能不是一个意思
        with torch.no_grad():
            mask = torch.ones_like(x)
            for i , len_of_seq in enumerate(length):
                mask[i][:,len_of_seq:] = 0 
            x = x * mask # 这样直接用是错的，因为维度已经变了，所以要重新计算mask

        return x

# def Conv_with_mask(batch_sequence, sequence_length,mask,num_nits,dialtion_rate,kernel_size):  #
#     """
#     ### build a convotion building block with mask
#     Args:
#         batch_sequence : Tensor of size (batch_size,input_embedding_size ,sequence_max_length)
#         sequence_length : list of size(batch_size)
#         mask : same size as batch_sequence

#     """
#     Mask_Conv1d = nn.Conv1d()

#     # create a mask with the same shape as padded data or batch in this example : batch_sequence
#     # mask = torch.ones_like(batch_sequence)   #
#     # mask[:,:,batch_sequence.size() : ]  = 0
#     # do not need to do such thing, I have transer generating mask to utilis.py


#     # first time mask element-wise multiplication
#     padding_zeroed =

#     conved = Mask_Conv1d(padding_zeroed)

#     # second time mask also called  re-zero padding (a technique)  这个techinique 跟paper里面所要表示的可能不是一个意思

#     return


class Residual_Block(nn.Module):
    def __init__(self, in_channels, layer_index, hparams) -> None:
        """
            hparams: dictionary from str to hyper parameters 

        """
        super().__init__()
        # calculate the parameter for bulidng bolocks like con1d etc.

        shifted_layer_index = layer_index - hparams["first_dilated_layer"] + 1  # 这个first_dataled_layer 有啥用呢？
        dilation_rate = max(1, hparams["dilation_rate"] ** shifted_layer_index)
        num_bottleneck_units = math.floor(
            hparams["resnet_bottleneck_factor"] * hparams["filters"])

        # self.residual_block = nn.Sequential(
        #     nn.BatchNorm1d(num_features=in_channels),
        #     nn.ReLU(),
        #     Conv1d_with_mask(in_channels, num_bottleneck_units,
        #                      kernel_size=hparams["kernel_size"], dilation=dilation_rate),
        #     nn.BatchNorm1d(num_features=num_bottleneck_units),
        #     nn.ReLU(),

        #     # bottleneck covolution
        #     Conv1d_with_mask(num_bottleneck_units, hparams["filters"], kernel_size = 1, dilation=1)
        # )

        # using the following to substitue the nn.sequential above to make most of "mask"
        self.batch_norm_1 =  nn.BatchNorm1d(num_features=in_channels)
        self.active_fn = nn.ReLU()
        self.diate_conv =  Conv1d_with_mask(in_channels, num_bottleneck_units,
                             kernel_size=hparams["kernel_size"], dilation=dilation_rate)
        self.batch_norm_2 =  nn.BatchNorm1d(num_features=num_bottleneck_units)
        self.bottleneck_conv= Conv1d_with_mask(num_bottleneck_units, hparams["filters"], kernel_size = 1, dilation=1)
    



    def forward(self, x, length):
        # identity = x   # the same name regulation as offical document
        # out = self.residual_block(x)
        # Because vonlution will not change the length of each sampel, so the mask can be used many times
        out = self.batch_norm_1(x)
        out = self.active_fn(out)
        out = self.diate_conv(out,length)
        out = self.batch_norm_2(out)
        out = self.active_fn(out)
        out = self.bottleneck_conv(out,length)
        # pass  # mask 应该怎末传进去？solved

        # return out + identity  # skip connection
        return out + x

class DNA_Model(nn.Module):
    def __init__(self, hparams,num_output_classes) -> None:
        """
            Build the neural network 
        """
        super().__init__()

        # "build a initial convolution "
        self.init_conv = Conv1d_with_mask(
            4, hparams["filters"], hparams["kernel_size"], dilation=1)
        
        # residual part 
        pass  # 用列表的方式有问题，只有第一个residual block 的输入的in_channel 是初始的embedding 4，而后面每一个residual
                    #block 的inchanel 是固定不变，所以要改写，让不变的inchaneel用ModuleLlist 生成，第一个residual block就不用了
              # rewirte

        # self.first_res_block = Residual_Block(4,layer_index=0,hparams=hparams)

        self.residual_blocks = nn.ModuleList([Residual_Block(hparams["filters"], layer_index, hparams) for layer_index in range(hparams["num_layers"])])
        # self.flatten = nn.Flatten()
        # max pool along the sequence dimension
            # 首先要取padding 化，将sequence 以外的长度不考虑，重置为0 就可以了
            # by seting kernetl__size and stride equal to the length of the sequence dimension 
        #self.max_pool = nn.MaxPool1d()  # max_pool 应该是个动态变换的，它不需要learnable 的参数，所以将其放再for_ward() 那边试试



        # preciation part : 先maxpool 再 接一个fully connected layer 
        #    max pool here ....figuratively 
        ##########################################################
        self.predic_layer = nn.Linear(hparams["filters"],num_output_classes)

    def forward(self, x, length):
        """
            Args:
                x : size(batch,input_embedding,max_length)
        """
        
        #pool_layer = nn.MaxPool1d(x.size()[-1],x.size()[-1]) # 该layer的大小是根据每次不同的输入动态的改变的

        # 这里可以不使用这个layer，用一个函数实现，而不是使用一个类，可能会增加batch size 的大小，然后增加训练的速度
        # 这里可能存在这样的改进，但是增不增加模型batch_size 不得而知？

        #----------------------------------------------------- forward
        out = self.init_conv(x,length)    # bug : 经过卷积之后，维度竟然没有发生改变 忘了在Conv1d_with_mask 赋值

        # out = self.first_res_block(out,mask)
        
        for resnet_block in self.residual_blocks:
            out = resnet_block(out,length)

        # max pool here  这里可能没有必要，因为一个conv_with bask 里面有两次mask操作
        with torch.no_grad():
            mask = torch.ones_like(out)   # mask = torch.ones_like(x)  这里写错了忘改了
            for i , len_of_seq in enumerate(length):
                mask[i][:,len_of_seq:] = 0 
            out = out * mask

        #out = pool_layer(out)  # now the size of out is Size([batch_size, input_sembedding,1])
        out =torch.max_pool1d(out,x.size()[-1],x.size()[-1]) # if use this sequeeze_, then do not use nn.flatten() but this will 
                                                                # cause problems when the one sample point was left -==> size mismatch
        
        out = torch.nn.Flatten()(out)
        #logits = self.predic_layer(out) # use logits to follow the regular name accoring the maching learning by tensorflow and pytorch
                                        # perhaps this will make the memory of the GPU not effectively because the variable "out" still remain alive
        out  = self.predic_layer(out)
        return out
    
