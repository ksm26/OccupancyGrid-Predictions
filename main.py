'''
Usgae: Main script for training and running the experiments
'''
import os, cv2, re, copy 
import shutil
import argparse
import numpy as np
import math
import torch
torch.cuda.empty_cache()
import time
import datetime
import collections
import torch.nn as nn
from torch.utils import data
from torch.utils.data.dataloader import DataLoader
from models.Model import Network_combinedStaticSemantic, Network_static_Semantic, Network_static_full, Network_standard_static_full, Network_ConvLSTM_combinedStaticSemantic
from dataloader import NuscenesDataset
from sklearn.metrics import f1_score
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from skimage.metrics import peak_signal_noise_ratio 
from test_model import test_combinedStaticSemantic, test_StaticSemantic, test_StaticFull, test_StandardLSTM_StaticFull

maindir = '/home/khushdeep/Desktop/star_predrnn'

parser = argparse.ArgumentParser(description='Predicting Future Occupancy Grids in Dynamic Environment with Spatio-Temporal Learning')
# training/test
parser.add_argument('--is_training', type=int, default=1)
parser.add_argument('--use_std_rnn', type=int, default=0, help='Use standard ConvLSTM instead of SpatioTemporalLSTM, this is an baseline')
parser.add_argument('--device', type=str, default='cuda')

# Load Checkpoints
parser.add_argument('--saveckpt_step', type=int, default=1, help='Save checkpoint after certain number of epochs') 
parser.add_argument('--pretrained_network', type=str, default='', help='Path to pretrained model')

# Data loading
parser.add_argument('--traindata', type=str, default=f'{maindir}/dataset/nuscenes_train', help='Path to training dataset')
parser.add_argument('--testdata', type=str, default=f'{maindir}/dataset/nuscenes_test', help='Path to testing dataset')

# Separate PredRNN Static and Semantic predictions from Static and Full images
parser.add_argument('--saveckpt_path', type=str, default=f'{maindir}/checkpoints/StaticFull', help='Path to save checkpoints')
parser.add_argument('--savetestimages', type=str, default=f'{maindir}/results/StaticFull', help='Path to save Test images')

# parameters input to model 
parser.add_argument('--seq_len', type=int, default=14, help='Total length of the sequence')
parser.add_argument('--input_len', type=int, default=8, help='Input length to the prediction model')
parser.add_argument('--seqimg_gap', type=int, default=2, help='Gap between two sequences (least value: 1)')
parser.add_argument('--img_channels', type=int, default=1)
parser.add_argument('--img_width', type=int, default=600, help='Original image width')
parser.add_argument('--img_height', type=int, default=600, help='Original image height')
parser.add_argument('--resize_img', type=int, default=1, help='Bool for resizing the original image')
parser.add_argument('--resize_img_ht', type=int, default=256, help='Resized image height ')
parser.add_argument('--resize_img_wd', type=int, default=256, help='Resized image width')

# Model
parser.add_argument('--model_name', type=str, default='predrnn')
parser.add_argument('--num_hidden', type=list, default=[4,4,4,4], help= "Number of hidden layers")
#parser.add_argument('--num_hidden', type=list, default=[64,64,64,64], help= "Number of hidden layers")

# Only one of the below has to be true 
parser.add_argument('--use_combinedStaticSemantic', type=int, default=0, help= "Combined Static and Semantic objects")
parser.add_argument('--use_StaticSemantic',         type=int, default=1, help= "Separate Static and Semantic objects")
parser.add_argument('--use_StaticFull',            type=int, default=0, help= "Input is static and full image and predict separate Static and Semantic objects")
parser.add_argument('--use_semantic_masking', type=int, default=0, help='Use masking for semantic labels in separate static-semantic-prediction')

parser.add_argument('--filter_size',  type=int,  default=5)
parser.add_argument('--stride',       type=int,  default=1)

# Reverse Scheduled Sampling 
parser.add_argument('--reverse_scheduled_sampling', type=int, default=1, help='Boolean for choosing the training scheme' )
parser.add_argument('--r_sampling_step1', type=int, default=25000)
parser.add_argument('--r_sampling_step2', type=int, default=50000)
parser.add_argument('--r_exp_alpha',      type=float, default=5000)

# Scheduled Sampling
parser.add_argument('--scheduled_sampling',     type=int, default=1)
parser.add_argument('--sampling_stop_iter',     type=int, default=50000) 
parser.add_argument('--sampling_start_value',   type=float, default=1.0)
parser.add_argument('--sampling_changing_rate', type=float, default=0.00002)

# Optimization
parser.add_argument('--k_loss', type=float, default=10)
parser.add_argument('--lr', type=float, default=0.0003)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--epochs', type=int, default=30)   
parser.add_argument('--display_batch_interval', type=int, default=20, help='Display the batch after an interval')
parser.add_argument('--optim_step', type=int, default=8)

args = parser.parse_args()
print("\n|",args) 

# ------------------------------------ TRAINING FUNCTIONS -------------------------------------------

def training_combinedStaticSemantic(network, data_loader, numiterations, start):
    time_elapsed = 0
    loss_hist = collections.deque(maxlen=100)

    for idx, batch in enumerate(data_loader):
        
        bwTensor = batch["bwTensor"].to(device)
        predictions = network(bwTensor, numiterations) # pass the sequence tensor to Network

        # Considering different losses  
        loss = MSE_criterion(predictions, bwTensor[:, 1:]) 
        loss_hist.append(float(loss))
        loss.backward()

        if idx % args.optim_step == 0:   
            optimizer.step()
            optimizer.zero_grad() 
            
        if idx % args.display_batch_interval == 0:
            print("Training loss is ", np.mean(loss_hist))
            time_elapsed = int(round(time.time())) - start
            print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            print("Batches done are ", idx)
            print("Total training time has been ", time_elapsed, " seconds")
        numiterations += 1 
    
    return numiterations

def training_static_full(network, data_loader, numiterations, start):
    time_elapsed = 0
    loss_hist = collections.deque(maxlen=100)
    for idx, batch in enumerate(data_loader):
        
        staticgridTensor, bwTensor, SemanticgridTensor =  batch["staticgridTensor"].to(device), batch["bwTensor"].to(device), batch["SemanticgridTensor"].to(device)

        # pass the sequence tensor to Network
        predictions_static, predictions_full = network(staticgridTensor, bwTensor, numiterations)

        loss_static = MSE_criterion(predictions_static, staticgridTensor[:, 1:]) 
        loss_Semantic = MSE_criterion(predictions_full, SemanticgridTensor[:, 1:]) 
        loss = loss_static + args.k_loss * loss_Semantic
        loss_hist.append(float(loss))
        loss.backward()

        if idx % args.optim_step == 0:   
            optimizer.step()
            optimizer.zero_grad() 
            
        if idx % args.display_batch_interval == 0:
            print("static loss is ", float(loss_static))
            print("Semantic loss is ", float(loss_Semantic))
            print("Training loss is ", np.mean(loss_hist))
            time_elapsed = int(round(time.time())) - start
            print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            print("Batches done are ", idx)
            print("Total training time has been ", time_elapsed, " seconds")
        numiterations += 1 
    
    return numiterations

def training_Masked_StaticSemantic(network, data_loader, numiterations, start):
    time_elapsed = 0
    loss_hist = collections.deque(maxlen=100)
    for idx, batch in enumerate(data_loader):
        
        staticgridTensor, SemanticgridTensor =  batch["staticgridTensor"].to(device), batch["SemanticgridTensor"].to(device)

        # pass the sequence tensor to Network
        predictions_static, predictions_Semantic = network(staticgridTensor, SemanticgridTensor, numiterations)

        # Considering different losses  
        loss_static = MSE_criterion(predictions_static, staticgridTensor[:, 1:]) 

        Masked_tensor = copy.deepcopy(SemanticgridTensor)
        Masked_tensor = torch.where(Masked_tensor==1, 1, 10)

        loss_Semantic = torch.mean(torch.square(predictions_Semantic- SemanticgridTensor[:, 1:])*Masked_tensor[:, 1:])

        loss = loss_static + args.k_loss * loss_Semantic
        loss_hist.append(float(loss))
        loss.backward()

        if idx % args.optim_step == 0:   
            optimizer.step()
            optimizer.zero_grad() 
            
        if idx % args.display_batch_interval == 0:
            print("static loss is ", float(loss_static))
            print("Semantic loss is ", float(loss_Semantic))
            print("Training loss is ", np.mean(loss_hist))
            time_elapsed = int(round(time.time())) - start
            print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            print("Batches done are ", idx)
            print("Total training time has been ", time_elapsed, " seconds")
        numiterations += 1 
    
    return numiterations

def training_StaticSemantic(network, data_loader, numiterations, start):
    time_elapsed = 0
    loss_hist = collections.deque(maxlen=100)
    for idx, batch in enumerate(data_loader):
        
        staticgridTensor, SemanticgridTensor =  batch["staticgridTensor"].to(device), batch["SemanticgridTensor"].to(device)

        # pass the sequence tensor to Network
        predictions_static, predictions_Semantic = network(staticgridTensor, SemanticgridTensor, numiterations)

        # Considering different losses  
        loss_static = MSE_criterion(predictions_static, staticgridTensor[:, 1:]) 
        loss_Semantic = MSE_criterion(predictions_Semantic, SemanticgridTensor[:, 1:]) 
        loss = loss_static + args.k_loss * loss_Semantic
        loss_hist.append(float(loss))
        loss.backward()

        if idx % args.optim_step == 0:   
            optimizer.step()
            optimizer.zero_grad() 
            
        if idx % args.display_batch_interval == 0:
            print("static loss is ", float(loss_static))
            print("Semantic loss is ", float(loss_Semantic))
            print("Training loss is ", np.mean(loss_hist))
            time_elapsed = int(round(time.time())) - start
            print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            print("Batches done are ", idx)
            print("Total training time has been ", time_elapsed, " seconds")
        numiterations += 1 
    
    return numiterations

# ---------------------------- TRAINING WRAPPER -----------------------------------------------------
def train_wrapper(network):
    print("\n| Training...")
    if args.pretrained_network :    
        print("\n| Loading the pre-trained model from ", args.pretrained_network)
        stats = torch.load(args.pretrained_network)
        network.load_state_dict(stats['net_param'])

    # initialization of the Custom Dataset class 
    train = NuscenesDataset(args.traindata, args) 
    print("\n| Length of Training dataset sequences is ", len(train))
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=4)

    if args.pretrained_network:    
        initepoch = int(re.findall(r'\d+', args.pretrained_network)[0]) + 1
        numiterations = initepoch * len(train) + 1
    else:   
        initepoch = 1
        numiterations = 0

    for numepoch in range(initepoch, args.epochs+1):
        start = int(round(time.time()))
        print("\nCurrent Epoch is ", numepoch)

        # Choose the training function according to set parameters 
        if args.use_StaticSemantic:    
            numiterations = training_StaticSemantic(network, train_loader, numiterations, start)
        elif args.use_combinedStaticSemantic:   
            numiterations = training_combinedStaticSemantic(network, train_loader, numiterations, start)
        elif args.use_StaticFull:   
            numiterations = training_static_full(network, train_loader, numiterations, start) 
        elif args.use_semantic_masking:
            numiterations = training_Masked_StaticSemantic(network, train_loader, numiterations, start) 
        
        #------------  SAVE the model --------------------
        if numepoch % args.saveckpt_step == 0:   
            stats = {}
            stats['net_param'] = network.state_dict()
            checkpoint_path = os.path.join(args.saveckpt_path, 'model.ckpt'+'-'+str(numepoch))
            torch.save(stats, checkpoint_path)
            print("save model to %s" % checkpoint_path)

# ---------------------------- MAIN -----------------------------------------------------

if __name__ == "__main__":

    os.makedirs(args.saveckpt_path, exist_ok = True)
    os.makedirs(args.savetestimages, exist_ok = True)
    if args.reverse_scheduled_sampling: print("\n| Reverse scheduled sampling is used")
    else:   print("\n| Reverse scheduled sampling is not used")

    print("\n| Initializing the model")
    device = torch.device(args.device)

    if args.use_std_rnn: 
        if args.use_StaticFull: 
            print("| Architecture: Standard RNN with separate Static and Full scenes as input and predicting separate Static and Semantic objects black and white images")
            network = Network_standard_static_full(args)
        if args.use_combinedStaticSemantic:
            print("| Architecture: Standard RNN with combined Static-Semantic objects black and white images")
            network = Network_ConvLSTM_combinedStaticSemantic(args)
    else:
        if args.use_combinedStaticSemantic:
            print("| Architecture: PredRNN with combined Static-Semantic objects black and white images")
            network = Network_combinedStaticSemantic(args)
        elif args.use_StaticSemantic: 
            print("| Architecture: PredRNN with separate Static and Semantic objects black and white images")
            network = Network_static_Semantic(args)
        elif args.use_StaticFull: 
            print("| Architecture: PredRNN with separate Static and Full scenes as input and predicting separate Static and Semantic objects black and white images")
            network = Network_static_full(args)
        elif args.use_semantic_masking:
            print("| Architecture: PredRNN with separate Static and Masked Semantic scenes as input and predicting separate Static and Semantic objects black and white images")
            network = Network_static_Semantic(args)

    network.to(device)

    MSE_criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr, amsgrad=True)

    if args.is_training:    train_wrapper(network)
    else:   
        if args.use_std_rnn: 
            if args.use_StaticFull: test_StandardLSTM_StaticFull(args, network) # Test for baseline ConvLSTM
            elif args.use_combinedStaticSemantic:  test_combinedStaticSemantic(args, network) 
        else:  
            if args.use_combinedStaticSemantic:  test_combinedStaticSemantic(args, network) 
            elif args.use_StaticSemantic:  test_StaticSemantic(args, network)
            elif args.use_StaticFull:   test_StaticFull(args, network)
            