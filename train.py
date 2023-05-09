import protcnn_model
from torch.utils.data import DataLoader , random_split  
import utils
import fastq_dataset
import hparams
import torch.nn as nn
import torch.optim as optim
import torch
import datetime
import os

# accelerating with mixed precision
scaler = torch.cuda.amp.GradScaler()
##########################################

# check whether or not we can use GPU accelarator these tow line comes from offical tutorial 
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
device = "cuda:0"
print(f"Using {device} device")

# model_save_path  = "./training_info"
current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
train_flag = "_clip_lr5e5_decayR0997"
model_save_path = "./saved_model/" + current_time + train_flag
print(f"model will be saved in{model_save_path}")
os.makedirs(model_save_path,exist_ok=True)

# path to data
train_stage = ("../Data_full/ninety_percentile_train.csv","../Data_full/train.csv")
test_stage = "../Data_full/test.csv"



# Hyparameter
# -------------------------------------------------------------------------------------------
# set the ratio of data for each set
Hyparams = hparams.hparams_set_train()
print(Hyparams)

train_ratio = 0.7
dev_ratio = 0.25
test_ratio = 0.1
Batch_size = Hyparams["bt_size"]
Num_epochs = Hyparams["num_epochs"]
# -------------------------------------------------------------------------------------------
# hyper parameter above 


# # loading the whole dataset
# print("begin to load the Fast dataset ")
# Data_whole = fastq_dataset.FastqDataset()
# print("the data is loaded successfully")




# # calculate the size of each set
# num_train = int(len(Data_whole) * train_ratio)
# num_dev = int(len(Data_whole) * dev_ratio)
# num_test = len(Data_whole) - num_train - num_dev


# # split data into train ,dev and test -- and set the data to a static by settingt the seed manually
# train_dataset, dev_dataset, test_datatest = random_split(Data_whole,[num_train,num_dev,num_test],generator=torch.Generator().manual_seed(40))



# train_dataloader = DataLoader(train_dataset,shuffle=True,batch_size=Batch_size,collate_fn=utils.my_collate_fn)
# test_dataloader = DataLoader(test_datatest,shuffle=True,batch_size=Batch_size,collate_fn=utils.my_collate_fn)
train_dataloader = None
test_dataloader = None

# # set  the model 
# My_model  = protcnn_model.DNA_Model(Hyparams,len(Data_whole.get_vocab())).to(device)

# loss_fn = nn.CrossEntropyLoss()
# optimiser = optim.Adam(My_model.parameters(),lr=Hyparams["init_lr_rate"])

def lr_lambda(epoch):
    turn_point = int(0.3 * Num_epochs)
    # turn_point = 1
    # debug 
    #print(f"the epoch  in lr_lamba is {epoch}") #  this function will be called when I initialize the LambdaLR
    if (epoch <= turn_point):
        # warm up 
        # return epoch * (Hyparams["max_lr_rate"] - Hyparams["init_lr_rate"])/(turn_point)
        return (epoch) / turn_point
    else :
        # expenetial decay 
        return  Hyparams["decay_rate"] ** (epoch - turn_point)

def lr_lambda2(epoch):

    warmup_epochs = int(0.3 * Num_epochs)
    
    if epoch <= warmup_epochs:
        return (epoch + 1)/warmup_epochs
    else : 
        return Hyparams["decay_rate"]**(epoch - warmup_epochs)

# lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser,lr_lambda=lr_lambda2)
lr_scheduler = None
#  if epoch < (0.3 * Num_epochs) else 1
# lr_schedule = optim.lr_scheduler.LambdaLR(optimiser,lr_lamba)
# lr_schedule = optim.lr_scheduler.LambdaLR(optimiser,lambda epoch :  
#                                           epoch / (0.3 * Num_epochs) if epoch < (0.3 * Num_epochs) else 1 )
# lr_schedule2 = optim.lr_scheduler.ExponentialLR(optimiser,gamma=0.9)



def train(dataloader,model,loss_fn,optimiser):

    
    # test for GPU memory 
    # mem_allocated = torch.cuda.max_memory_allocated(device=device)
    # print(f"The memory used before training: Maximum GPU memory allocated: {mem_allocated/(1024**2):.2f} MB")
    
    size = len(dataloader.dataset)

    for epoch in range(Num_epochs):

        # set model to train mode 
        # model.train()
        model.train()
        print(f"start traing ===> epoch is {epoch}")
        llrr = optimiser.param_groups[0]["lr"]
        # llrr =  optimiser.state_dict()['param_groups'][0]["lr"]
        print(f"the learning rate of epoch is ----{llrr}")

        for batch, (features, labels,length) in enumerate(dataloader):
            """
            features consist of a list of sub bathes, each of which has similar sequence lenth 
            """
            mask = length # use length to generate the mask to violate
            max_length = max(length) 

            optimiser.zero_grad()
            loss_whole = 0.0

            for sub_feature,sub_label,sub_length in zip(features,labels,length):
                features,labels = sub_feature.to(device),sub_label.to(device)
                       # forward pass
                # ...
                with torch.cuda.amp.autocast():
                    pred = model(features,sub_length)
                    loss = loss_fn(pred,labels)
                    loss_whole += loss
                
                # backward pass and optimise the parameter and accumulate the gradient. 
                # ...
                scaler.scale(loss).backward()   

              
            #clip the gradient 
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(),Hyparams["max_norm"])
            # optimiser.step()
            scaler.step(optimiser)
            scaler.update() # update the scale for the next generation

            

# ------------------------------------------------------------------------------------------------------------
            # features, labels= features.to(device), labels.to(device)
            # mem_allocated = torch.cuda.memory_allocated(device=device)
            # print(f"memory used PRE_beofre zeor_grad: Maximum GPU memory allocated: {mem_allocated/1024**2:.2f} MB",end= " <-->")
            



            # optimiser.zero_grad()

            # mem_allocated = torch.cuda.memory_allocated(device=device)
            # print(f"memory used PRE: Maximum GPU memory allocated: {mem_allocated/1024**2:.2f} MB")

            # forward pass
                # ...
            # with torch.cuda.amp.autocast():
            #     pred = model(features,mask)
            #     loss = loss_fn(pred,labels)
            # mem_allocated = torch.cuda.memory_allocated(device=device)
            # print(f"memory used after forward: Maximum GPU memory allocated: {mem_allocated/1024**2:.2f} MB")

            # backward pass and optimise the parameter
                # ...
            # scaler.scale(loss).backward()

            # mem_allocated = torch.cuda.memory_allocated(device=device)
            # print(f"memory used after backword: Maximum GPU memory allocated: {mem_allocated/1024**2:.2f} MB")

            #clip the gradient 
            # scaler.unscale_(optimiser)
            # torch.nn.utils.clip_grad_norm_(model.parameters(),Hyparams["max_norm"])

            # # optimiser.step()
            # scaler.step(optimiser)
            # scaler.update() # update the scale for the next generation

            # mem_allocated = torch.cuda.memory_allocated(device=device)
            # print(f"memory used after paraupdate: Maximum GPU memory allocated: {mem_allocated/1024**2:.2f} MB")
        
            # inspect info:
            if batch % 100 == 0 :
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                loss, current_number = loss_whole.item(), (batch + 1) * len(features) * 4
                print(f"loss: {loss:>7f}  [{current_number:>5d}/{size:>5d} ==> epoch : {epoch}]  current time: {current_time}")
                # print(length)
            del loss
            del pred

            # test for GPU memory 
            # mem_allocated = torch.cuda.max_memory_allocated(device=device)
            # print(f"The max length of this batch is {max(length)} ",end=" ")
            # print(f"Epoch {epoch}: Maximum GPU memory allocated: {mem_allocated/1024**2:.2f} MB")

        torch.save(model.state_dict(),os.path.join(model_save_path, f"my_model_epoch[{epoch}].pth"))
        print(f"Model in epoch: [{epoch}] saved successfully")

        # lr_scheduler.step()

        test(test_dataloader,model,loss_fn)

def test(dataloader,model,loss_fn,inspect = False):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    torch.cuda.empty_cache()
    # set the model to eval mode
    model.eval()
    # test for GPU memory 
    # mem_allocated = torch.cuda.max_memory_allocated(device=device)
    # print(f"During test :begining: Maximum GPU memory allocated: {mem_allocated/1024**2:.2f} MB")

    test_loss, correct = 0, 0
    wrong_tensor = torch.tensor([0])
    with torch.no_grad():
        for features,labels,length in dataloader: 
            mask = length
            max_length = max(length)
            
            for sub_feature, sub_label, sub_length in zip(features,labels,length):

                features, labels = sub_feature.to(device), sub_label.to(device)
                pred = model(features,sub_length)  # logits   
                
                test_loss += loss_fn(pred,labels).item()
                correct += (pred.argmax(1) == labels).type(torch.float).sum().item()


            # test for GPU memory 
            # mem_allocated = torch.cuda.max_memory_allocated(device=device)
            # print(f"During test11: Maximum GPU memory allocated: {mem_allocated/1024**2:.2f} MB")


            # pred = model(features,mask)  # logits
            # test_loss += loss_fn(pred,labels).item()
            # correct += (pred.argmax(1) == labels).type(torch.float).sum().item()


            if inspect:
                pass
                wrong_classification = labels[~(pred.argmax(1) == labels)]
                wrong_tensor = torch.cat((wrong_tensor,wrong_classification),dim = 0)
            # -------------------------------------------------
            # test for GPU memory 
            # mem_allocated = torch.cuda.max_memory_allocated(device=device)
            
            # print(f"the max lenght of batch is f{max(length)}")
            # print(f"During test12: Maximum GPU memory allocated: {mem_allocated/1024**2:.2f} MB")

    test_loss /= num_batches
    correct /= size

    print(f"Test Error: \n Accuracy: {(100 * correct):0.1f}%, Avg loss : {test_loss:>8f}\n")   # 后续把时间信息也加上

    # torch.save(wrong_tensor,"./tensor_inspect/wrong_tensor1.pt")
    # print(wrong_tensor)

def train_whole():

    global test_dataloader
    global train_dataloader
    global lr_scheduler

    test_dataset = fastq_dataset.FastqDataset(test_stage)
    test_dataloader = DataLoader(test_dataset,shuffle=True,batch_size=Batch_size,collate_fn=utils.my_collate_fn)


    # set  the model 
    My_model  = protcnn_model.DNA_Model(Hyparams,len(test_dataset.get_vocab())).to(device)
    My_model.load_state_dict(torch.load("./saved_model/2023-04-30-20:43:07_clip_lr5e5_decayR0997/my_model_epoch[29].pth"))
    loss_fn = nn.CrossEntropyLoss()
    optimiser = optim.Adam(My_model.parameters(),lr=Hyparams["init_lr_rate"])

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser,lr_lambda=lr_lambda2)
    # two stage training 
    stage_num = 1
    while stage_num< 2:
        print(f"current stage is {stage_num}")

        # loading the whole dataset
        print("begin to load the Fast dataset ")
        train_dataset = fastq_dataset.FastqDataset(train_stage[stage_num])
        train_dataloader = DataLoader(train_dataset,shuffle=True,batch_size=Batch_size,collate_fn=utils.my_collate_fn)
        print(f"the data from {train_stage[stage_num]} is loaded successfully")

        train(dataloader=train_dataloader,model=My_model,loss_fn=loss_fn,optimiser=optimiser)

        stage_num += 1

    
def test_saved_model(): # to test the saved model

    """
    to find top K 
    """
    test_dataset = fastq_dataset.FastqDataset(test_stage)
    test_dataloader = DataLoader(test_dataset,shuffle=True,batch_size=Batch_size,collate_fn=utils.my_collate_fn)


    # load the model 
    My_model  = protcnn_model.DNA_Model(Hyparams,len(test_dataset.get_vocab())).to(device)
    My_model.load_state_dict(torch.load("./saved_model/2023-04-01-13:00:17_clip_lr5e5_decayR0997/my_model_epoch[39].pth"))
    loss_fn = nn.CrossEntropyLoss()
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    torch.cuda.empty_cache()

    # set the model to eval mode
    My_model.eval()


    test_loss, correct = 0, 0
    wrong_tensor = torch.tensor([])


    inspect = False
    topk_index = torch.tensor([])
    num_correct_preds = 0
    topk_in  = 10
    with torch.no_grad():
        for batch_num, (features,labels,length) in enumerate(test_dataloader): 
            mask = length
            max_length = max(length)
            
            for sub_feature, sub_label, sub_length in zip(features,labels,length):

                features, labels = sub_feature.to(device), sub_label.to(device)
                pred = My_model(features,sub_length)  # logits   
                

                _,topk_inds = torch.topk(pred,k = topk_in,dim=1)
                # topk_index = torch.cat((topk_index,topk_inds.cpu()),dim=0)
                correct_preds = torch.sum( sub_label.view(-1,1) == topk_inds.cpu(),dim = 1 )
                num_correct_preds += torch.sum(correct_preds).item()

                test_loss += loss_fn(pred,labels).item()
                correct += (pred.argmax(1) == labels).type(torch.float).sum().item()

            if inspect:
                wrong_classification = labels[~(pred.argmax(1) == labels)]
                wrong_tensor = torch.cat((wrong_tensor,wrong_classification),dim = 0)

            if(batch_num%100 == 0):
                current_total = (batch_num+1) * len(features) * 4 
                print(f"process : {current_total:>5d} / {size:>5d} ------> {num_correct_preds / current_total * 100:0.1f}%")

    print(f"The whole accuracy with top{topk_in} is {num_correct_preds / size * 100:0.1f}%")

    test_loss /= num_batches
    correct /= size

    print(f"Test Error: \n Accuracy: {(100 * correct):0.1f}%, Avg loss : {test_loss:>8f}\n")   # 后续把时间信息也加上

    # torch.save(wrong_tensor,"./tensor_inspect/wrong_tensor1.pt")
    # torch.save(topk_index,"./tensor_indpect/topk_index1.pt")
    # print(wrong_tensor)
    return topk_index



if __name__ == "__main__":
    train_whole()
    # test_saved_model()


    # test(test_dataloader,My_model,loss_fn)



def debug(option):
    # mem_allocated = torch.cuda.memory_allocated(device=device)
    # print(f"The memory used before training and testing: Maximum GPU memory allocated: {mem_allocated/1024**2:.2f} MB")
    if option == 1: # train the model 
        train_whole()
    elif option ==2: # test the model only 
        # test(test_dataloader,My_model,loss_fn)
        return test_saved_model()
    elif option == 3:
        pass
        # by setting the return_indices argument to True when create a dataloader inorder to get the index
        # load the model 
        # My_model.load_state_dict(torch.load("./saved_model/2023-04-01-13:00:17_clip_lr5e5_decayR0997/my_model_epoch[39].pth"))
        # test(test_dataloader,My_model,loss_fn)

        # pass
    





# 注：
"""
1. 加入 add clip gradient                           已将加入
2. 模型选择 :model selection                       （暂时没有）
3. 使用warm Up 在不同阶段都有动态的learning rate      已经完善
4. 模型保存的时候：指定路径的文件不存在应该先生成目录   已经完善
"""