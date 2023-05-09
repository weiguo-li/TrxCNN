# define a dict as a set of hyper parameters


def hparams_set_train():
    hparams = {}

    hparams["first_dilated_layer"] = 1
    hparams["dilation_rate"] = 3
    hparams["resnet_bottleneck_factor"]  = 0.5  #之前为了占位用了 None 可害惨我了 
    hparams["filters"] = 800  # default 1100
    hparams["kernel_size"] = 3 # i use 4 instead of 3 to debug and I find a bug 
                                #and rewrite it in module protcnn_model.py where I generate residual blocks
    hparams["num_layers"] = 4  # orinally 5 
    hparams["lr_rate"] = 0.0005  # 0.0005 oringanl 
    hparams["max_norm"] = 1 # the max norm value
    hparams["init_lr_rate"] = 0.0005  # 0.0005 oringanl 0.00005
    hparams["max_lr_rate"] = 0.0006
    hparams["decay_rate"] = 0.95
    hparams["num_epochs"] = 40
    hparams["bt_size"] = 32

    return hparams 

# {'first_dilated_layer': 1, 'dilation_rate': 3, 'resnet_bottleneck_factor': 0.5, 'filters': 1100, 
#  'kernel_size': 3, 'num_layers': 4, 'lr_rate': 0.0005, 'max_norm': 1, 
#  'init_lr_rate': 8e-05, 'max_lr_rate': 0.0006, 'decay_rate': 0.95, 'num_epochs': 40, 'bt_size': 16}

def hparams_set_2(): 
    hparams = {}

    hparams["first_dilated_layer"] = 1
    hparams["dilation_rate"] = 1
    return hparams 

