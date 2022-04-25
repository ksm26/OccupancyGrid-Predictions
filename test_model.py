'''
Usgae: Main script for training and running the experiments
'''
import os, cv2, re
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
 

# ---------------------------- TESTING -----------------------------------------------------
def test_combinedStaticSemantic(args, network):
    MSE_criterion = nn.MSELoss()
    device = torch.device(args.device)
    start = int(round(time.time()))
    time_elapsed = 0
    print("| Testing...")
    print("| Loading the pre-trained model from ", args.pretrained_network)
    stats = torch.load(args.pretrained_network)
    network.load_state_dict(stats['net_param'])

    # Initialize the TEST dataset
    test = NuscenesDataset(args.testdata, args) 
    print("\n| Length of Testing dataset sequences is ", len(test))
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=4)

    f1_scores, ssim_scores, psnr_scores = [], [], []
    f1_scores_list, avg_f1_scores, std_dev_f1_scores = [], [], []
    ssim_scores_list, avg_ssim_scores, std_ssim_scores = [], [], []
    psnr_scores_list, avg_psnr_scores, std_psnr_scores = [], [], []
    total_loss, loss_perScene_list = [], []
    mean_loss_perScene, stdDev_loss_perScene = [], []

    for i in range(args.seq_len - args.input_len):
        f1_scores.append(0)
        ssim_scores.append(0)
        psnr_scores.append(0)
        avg_f1_scores.append(0)
        std_dev_f1_scores.append(0)
        avg_ssim_scores.append(0)
        std_ssim_scores.append(0)
        avg_psnr_scores.append(0)
        std_psnr_scores.append(0)
        f1_scores_list.append([])
        ssim_scores_list.append([])
        psnr_scores_list.append([])
        loss_perScene_list.append([])
        mean_loss_perScene.append(0)
        stdDev_loss_perScene.append(0)

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            
            seqTensor, bwTensor = batch["seqTensor"].to(device), batch["bwTensor"].to(device)
            # pass the sequence tensor to Network
            predictions = network(bwTensor, idx) 

            loss = MSE_criterion(predictions, bwTensor[:, 1:]) 
            total_loss.append(float(loss))

            output_len = args.seq_len - args.input_len
            img_out = predictions[:, -output_len:]

            testimgpath = os.path.join(args.savetestimages, str(idx))
            os.makedirs(testimgpath, exist_ok=True)

            for i in range(output_len):
                bwOriginal = bwTensor[:, i+args.input_len , :, :, :].permute(0,3,1,2) 
                gx =  img_out[:, i, :, :, :].permute(0,3,1,2) 

                # Accumulate total Static loss per scene
                loss_perScene_list[i].append(float(MSE_criterion(gx, bwOriginal))) 
                gxCombined = (gx>0.6).float()
                
                real_frm = np.uint8(bwOriginal.cpu().detach().numpy() * 255).ravel()
                pred_frm = np.uint8(gxCombined.cpu().detach().numpy() * 255).ravel()
                
                cv2.imwrite(f"{testimgpath}/gt_combined{i}.jpg", np.uint8(bwOriginal[0][0].cpu().detach().numpy()* 255) )
                cv2.imwrite(f"{testimgpath}/pd_combined{i}.jpg", np.uint8(gxCombined[0][0].cpu().detach().numpy() * 255)) 
                
                # Combine grouth truth and Semantic predicted images to SAVE Results
                gxSD = np.ones((gxCombined.shape[2], gxCombined.shape[3], 3), dtype = np.uint8)*255
                ori = 1-bwOriginal[0][0].cpu().detach().numpy()
                gxComb = 1-gxCombined[0][0].cpu().detach().numpy()

                gxSD[ori.astype(bool)]=[0,0,255] # ground Truth in RED 
                gxSD[gxComb.astype(bool)]=[255,0,0] # Semantic predictions in BLUE
                cv2.imwrite(f"{testimgpath}/gt_pd_combined{i}.jpg", gxSD )

                f1_scores[i] = f1_score(real_frm, pred_frm, average='micro') # F1 score calculation 
                f1_scores_list[i].append(f1_scores[i])
                ssim_scores[i] = structural_similarity(real_frm, pred_frm, win_size=11) # SSIM
                ssim_scores_list[i].append(ssim_scores[i])
                psnr_scores[i] = peak_signal_noise_ratio(real_frm, pred_frm)
                psnr_scores_list[i].append(psnr_scores[i])
            
            if idx % args.debug_itrDisplay_interval == 0 :
                print("\n----------------------------------------------------------")
                print('Current sequence no. ',idx)
                print('Total loss is ', float(loss))
                for i in range(output_len):
                    avg_f1_scores[i]= np.mean(f1_scores_list[i])
                    std_dev_f1_scores[i] = np.std(f1_scores_list[i])
                    avg_ssim_scores[i] = np.mean(ssim_scores_list[i])
                    std_ssim_scores[i] = np.std(ssim_scores_list[i])
                    avg_psnr_scores[i] = np.mean(psnr_scores_list[i])
                    std_psnr_scores[i] = np.std(psnr_scores_list[i])
                    mean_loss_perScene[i] = np.mean(loss_perScene_list[i])
                    stdDev_loss_perScene[i] = np.std(loss_perScene_list[i])

                avg_f1_scores = [round(num, 4) for num in avg_f1_scores]
                std_dev_f1_scores = [round(num, 5) for num in std_dev_f1_scores]
                print("\nAverage f1 scores are ", avg_f1_scores)
                print("Std deviation of f1 scores are ", std_dev_f1_scores)

                avg_ssim_scores = [round(num, 4) for num in avg_ssim_scores]
                std_ssim_scores = [round(num, 5) for num in std_ssim_scores]
                print("\nAverage ssim scores are ", avg_ssim_scores)
                print("Std deviation of ssim scores are ", std_ssim_scores)

                avg_psnr_scores = [round(num, 4) for num in avg_psnr_scores]
                std_psnr_scores = [round(num, 5) for num in std_psnr_scores]
                print("\nAverage psnr scores are ", avg_psnr_scores)
                print("Std deviation of psnr scores are ", std_psnr_scores)

                mean_loss_perScene = [round(num, 5) for num in mean_loss_perScene]
                stdDev_loss_perScene = [round(num, 5) for num in stdDev_loss_perScene]
                print("\nAverage loss per scene ", mean_loss_perScene)
                print("Std deviation of loss per scenes are ", stdDev_loss_perScene)

                print("\nMean of overall loss is ", np.mean(total_loss))
                print("Std deviation of overall loss is ", np.std(total_loss))

                f1_scores = [round(num, 3) for num in f1_scores]
                print("\nF1 scores for this sequence are ", f1_scores)

                time_elapsed = int(round(time.time())) - start
                print(datetime.datetime.now().strftime('\n%Y-%m-%d %H:%M:%S'))
                print("Batches done are ", idx)
                print("Total testing time has been ", time_elapsed, " seconds")

def test_StaticSemantic(args, network):
    MSE_criterion = nn.MSELoss()
    device = torch.device(args.device)
    start = int(round(time.time()))
    time_elapsed = 0
    print("\n| Testing the separate Static Semantic model")
    print("\n| Loading the pre-trained model from ", args.pretrained_network)
    stats = torch.load(args.pretrained_network)
    network.load_state_dict(stats['net_param'])
    network.eval()

    # Initialize the TEST dataset
    test = NuscenesDataset(args.testdata, args) 
    print("\n| Total length of Testing dataset sequences is ", len(test))
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=4)
    f1_scores, ssim_scores, psnr_scores = [], [], []
    f1_scores_list, avg_f1_scores, std_dev_f1_scores = [], [], []
    ssim_scores_list, avg_ssim_scores, std_ssim_scores = [], [], []
    psnr_scores_list, avg_psnr_scores, std_psnr_scores = [], [], []
    total_loss_static, total_loss_Semantic = [], []   
    lossStatic_perScene_list, lossSemantic_perScene_list = [], []
    mean_lossStatic_perScene, mean_lossSemantic_perScene = [], []
    stdDev_lossStatic_perScene, stdDev_lossSemantic_perScene = [], []

    for i in range(args.seq_len - args.input_len):
        f1_scores.append(0)
        ssim_scores.append(0)
        psnr_scores.append(0)
        avg_f1_scores.append(0)
        std_dev_f1_scores.append(0)
        avg_ssim_scores.append(0)
        std_ssim_scores.append(0)
        avg_psnr_scores.append(0)
        std_psnr_scores.append(0)
        f1_scores_list.append([])
        ssim_scores_list.append([])
        psnr_scores_list.append([])
        lossStatic_perScene_list.append([])
        lossSemantic_perScene_list.append([])
        mean_lossStatic_perScene.append(0)
        stdDev_lossStatic_perScene.append(0)
        mean_lossSemantic_perScene.append(0)
        stdDev_lossSemantic_perScene.append(0)

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            staticgridTensor, SemanticgridTensor, bwtensor =  batch["staticgridTensor"].to(device), batch["SemanticgridTensor"].to(device), batch["bwTensor"].to(device)
            # pass the sequence tensor to Network
            predictions_static, predictions_Semantic = network(staticgridTensor, SemanticgridTensor, idx)

            # Considering Static, Semantic and Total loss  
            loss_static = MSE_criterion(predictions_static, staticgridTensor[:, 1:]) 
            loss_Semantic = MSE_criterion(predictions_Semantic, SemanticgridTensor[:, 1:]) 
            loss = loss_static + args.k_loss * loss_Semantic
            total_loss_static.append(float(loss_static))
            total_loss_Semantic.append(float(loss_Semantic))

            output_len = args.seq_len - args.input_len
            imgoutStatic = predictions_static[:, -output_len:]
            imgoutSemantic = predictions_Semantic[:, -output_len:]
            testimgpath = os.path.join(args.savetestimages, str(idx))
            os.makedirs(testimgpath, exist_ok=True)

            for i in range(output_len):
                xStatic = staticgridTensor[:, i+args.input_len , :, :, :].permute(0,3,1,2) 
                gxStatic =  imgoutStatic[:, i, :, :, :].permute(0,3,1,2)
                # Accumulate total Static loss per scene
                lossStatic_perScene_list[i].append(float(MSE_criterion(gxStatic, xStatic))) 
                gxStatic = (gxStatic>0.6).float()
                #cv2.imwrite(f"{testimgpath}/pd_Static{i}.jpg", gxStatic[0][0].cpu().detach().numpy()*255 )

                xSemantic = SemanticgridTensor[:, i+args.input_len , :, :, :].permute(0,3,1,2) 
                gxSemantic =  imgoutSemantic[:, i, :, :, :].permute(0,3,1,2) 
                # Accumulate total Semantic loss per scene
                lossSemantic_perScene_list[i].append(float(MSE_criterion(gxSemantic, xSemantic)))
                gxSemantic = (gxSemantic>0.6).float()
                #cv2.imwrite(f"{testimgpath}/pd_Semantic{i}.jpg", gxSemantic[0][0].cpu().detach().numpy()*255 )

                bwOriginal = bwtensor[:, i+args.input_len , :, :, :].permute(0,3,1,2) 
                ori = 1-bwOriginal[0][0].cpu().detach().numpy()
                gxDyn = 1-gxSemantic[0][0].cpu().detach().numpy()
                gxSta = 1-gxStatic[0][0].cpu().detach().numpy()

                # Combine grouth truth and Semantic predicted images to SAVE Results
                gxSD = np.ones((gxStatic.shape[2], gxStatic.shape[3], 3), dtype = np.uint8)*255
                gxSD[ori.astype(bool)]=[0,0,255] # ground Truth in RED 
                gxSD[gxDyn.astype(bool)]=[255,0,0] # Semantic predictions in BLUE
                cv2.imwrite(f"{testimgpath}/gt_full_pd_Dyn{i}.jpg", gxSD )

                # Combine Static and Semantic predicted color images to SAVE Results
                gxSD = np.ones((gxStatic.shape[2], gxStatic.shape[3], 3), dtype = np.uint8)*255
                gxSD[gxSta.astype(bool)]=[255,0,0] # static predictions in BLUE
                #cv2.imwrite(f"{testimgpath}/pd_Static_color{i}.jpg", gxSD )

                # SAVE the combined image in COLOR
                gxSD = np.ones((gxStatic.shape[2], gxStatic.shape[3], 3), dtype = np.uint8)*255
                gxSD[gxDyn.astype(bool)]=[0,255,0] # Semantic predictions in GREEN
                #cv2.imwrite(f"{testimgpath}/pd_Dyn_color{i}.jpg", gxSD )
                
                gxSD[gxSta.astype(bool)]=[255,0,0] # static predictions in BLUE
                cv2.imwrite(f"{testimgpath}/pd_Static_Dyn_color{i}.jpg", gxSD )

                # Combine Static and Semantic predictions to SAVE Results
                gxStatDyn = (1-gxSemantic) + (1-gxStatic)
                gxStatDyn = (gxStatDyn<0.6).float()
                #cv2.imwrite(f"{testimgpath}/pd_StatDyn_gray{i}.jpg", gxStatDyn[0][0].cpu().detach().numpy()*255 )

                # F1 score for COMBINED static and Semantic predictions
                bwOriginal = np.uint8(bwOriginal[0][0].cpu().detach().numpy() * 255).ravel()
                gxStatDyn  = np.uint8(gxStatDyn[0][0].cpu().detach().numpy() * 255).ravel()
                f1_scores[i] = f1_score(bwOriginal, gxStatDyn, average='micro')
                f1_scores_list[i].append(f1_scores[i])
                ssim_scores[i] = structural_similarity(bwOriginal, gxStatDyn, win_size=11) # SSIM
                ssim_scores_list[i].append(ssim_scores[i])
                psnr_scores[i] = peak_signal_noise_ratio(bwOriginal, gxStatDyn)
                psnr_scores_list[i].append(psnr_scores[i])
            
            if idx % args.debug_itrDisplay_interval == 0 :
                print("\n----------------------------------------------------------")
                print('Current sequence no. ',idx)
                print('Total loss for static+Semantic is ', float(loss))
                for i in range(output_len):
                    avg_f1_scores[i]= np.mean(f1_scores_list[i])
                    std_dev_f1_scores[i] = np.std(f1_scores_list[i])
                    avg_ssim_scores[i] = np.mean(ssim_scores_list[i])
                    std_ssim_scores[i] = np.std(ssim_scores_list[i])
                    avg_psnr_scores[i] = np.mean(psnr_scores_list[i])
                    std_psnr_scores[i] = np.std(psnr_scores_list[i])
                    mean_lossStatic_perScene[i] = np.mean(lossStatic_perScene_list[i])
                    stdDev_lossStatic_perScene[i] = np.std(lossStatic_perScene_list[i])
                    mean_lossSemantic_perScene[i] = np.mean(lossSemantic_perScene_list[i])
                    stdDev_lossSemantic_perScene[i] = np.std(lossSemantic_perScene_list[i])

                avg_f1_scores = [round(num, 4) for num in avg_f1_scores]
                std_dev_f1_scores = [round(num, 5) for num in std_dev_f1_scores]
                print("\nAverage f1 scores are ", avg_f1_scores)
                print("Std deviation of f1 scores are ", std_dev_f1_scores)

                avg_ssim_scores = [round(num, 4) for num in avg_ssim_scores]
                std_ssim_scores = [round(num, 5) for num in std_ssim_scores]
                print("\nAverage ssim scores are ", avg_ssim_scores)
                print("Std deviation of ssim scores are ", std_ssim_scores)

                avg_psnr_scores = [round(num, 4) for num in avg_psnr_scores]
                std_psnr_scores = [round(num, 5) for num in std_psnr_scores]
                print("\nAverage psnr scores are ", avg_psnr_scores)
                print("Std deviation of psnr scores are ", std_psnr_scores)

                mean_lossStatic_perScene = [round(num, 5) for num in mean_lossStatic_perScene]
                stdDev_lossStatic_perScene = [round(num, 5) for num in stdDev_lossStatic_perScene]
                print("\nAverage Static loss per scene ", mean_lossStatic_perScene)
                print("Std deviation of Static loss per scenes are ", stdDev_lossStatic_perScene)

                mean_lossSemantic_perScene = [round(num, 5) for num in mean_lossSemantic_perScene]
                stdDev_lossSemantic_perScene = [round(num, 5) for num in stdDev_lossSemantic_perScene]
                print("\nAverage Semantic loss per scene ", mean_lossSemantic_perScene)
                print("Std deviation of Semantic loss per scenes are ", stdDev_lossSemantic_perScene)

                # Mean and STD of static and Semantic loss
                print("\nMean of overall Static loss is ", np.mean(total_loss_static))
                print("Std deviation of overall Static loss is ", np.std(total_loss_static))
                print("\nMean of overall Semantic loss is ", np.mean(total_loss_Semantic))
                print("Std deviation of overall Semantic loss is ", np.std(total_loss_Semantic))

                f1_scores = [round(num, 3) for num in f1_scores]
                print("\nF1 scores for this sequence are ", f1_scores)

                time_elapsed = int(round(time.time())) - start
                print(datetime.datetime.now().strftime('\n%Y-%m-%d %H:%M:%S'))
                print("Batches done are ", idx)
                print("Total testing time has been ", time_elapsed, " seconds")

def test_StaticFull(args, network):
    MSE_criterion = nn.MSELoss()
    device = torch.device(args.device)
    # Input: Static and full image scenes, Predictions: Static and Semantic scenes
    start = int(round(time.time()))
    time_elapsed = 0
    print("\n| Testing the separate Static and Full image scenes model with Static and Semantic as output")
    print("\n| Loading the pre-trained model from ", args.pretrained_network)
    stats = torch.load(args.pretrained_network)
    network.load_state_dict(stats['net_param'])
    network.eval()

    # Initialize the TEST dataset
    test = NuscenesDataset(args.testdata, args) 
    print("\n| Total length of Testing dataset sequences is ", len(test))
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=4)
    f1_scores, ssim, psnr = [], [], []
    f1_scores_list, avg_f1_scores, std_dev_f1_scores = [], [], []
    total_loss_static, total_loss_Semantic = [], []   
    lossStatic_perScene_list, lossSemantic_perScene_list = [], []
    mean_lossStatic_perScene, mean_lossSemantic_perScene = [], []
    stdDev_lossStatic_perScene, stdDev_lossSemantic_perScene = [], []

    for i in range(args.seq_len - args.input_len):
        f1_scores.append(0)
        avg_f1_scores.append(0)
        std_dev_f1_scores.append(0)
        f1_scores_list.append([])
        lossStatic_perScene_list.append([])
        lossSemantic_perScene_list.append([])
        mean_lossStatic_perScene.append(0)
        stdDev_lossStatic_perScene.append(0)
        mean_lossSemantic_perScene.append(0)
        stdDev_lossSemantic_perScene.append(0)

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            
            staticgridTensor, SemanticgridTensor, bwtensor =  batch["staticgridTensor"].to(device), batch["SemanticgridTensor"].to(device), batch["bwTensor"].to(device)
            # pass the sequence tensor to Network
            predictions_static, predictions_full = network(staticgridTensor, SemanticgridTensor, idx)

            # Considering Static, Semantic and Total loss  
            loss_static = MSE_criterion(predictions_static, staticgridTensor[:, 1:]) 
            loss_Semantic = MSE_criterion(predictions_full, SemanticgridTensor[:, 1:]) 
            loss = loss_static + args.k_loss * loss_Semantic
            total_loss_static.append(float(loss_static))
            total_loss_Semantic.append(float(loss_Semantic))

            output_len = args.seq_len - args.input_len
            imgoutStatic = predictions_static[:, -output_len:]
            imgoutSemantic = predictions_full[:, -output_len:]
            testimgpath = os.path.join(args.savetestimages, str(idx))
            os.makedirs(testimgpath, exist_ok=True)

            for i in range(output_len):
                xStatic = staticgridTensor[:, i+args.input_len , :, :, :].permute(0,3,1,2) 
                gxStatic =  imgoutStatic[:, i, :, :, :].permute(0,3,1,2)
                # Accumulate total Static loss per scene
                lossStatic_perScene_list[i].append(float(MSE_criterion(gxStatic, xStatic))) 
                gxStatic = (gxStatic>0.6).float()
                #cv2.imwrite(f"{testimgpath}/pd_Static{i}.jpg", gxStatic[0][0].cpu().detach().numpy()*255 )

                xSemantic = SemanticgridTensor[:, i+args.input_len , :, :, :].permute(0,3,1,2) 
                gxSemantic =  imgoutSemantic[:, i, :, :, :].permute(0,3,1,2) 
                # Accumulate total Semantic loss per scene
                lossSemantic_perScene_list[i].append(float(MSE_criterion(gxSemantic, xSemantic)))
                gxSemantic = (gxSemantic>0.6).float()
                #cv2.imwrite(f"{testimgpath}/pd_Semantic{i}.jpg", gxSemantic[0][0].cpu().detach().numpy()*255 )

                bwOriginal = bwtensor[:, i+args.input_len , :, :, :].permute(0,3,1,2) 
                ori = 1-bwOriginal[0][0].cpu().detach().numpy()
                gxDyn = 1-gxSemantic[0][0].cpu().detach().numpy()
                gxSta = 1-gxStatic[0][0].cpu().detach().numpy()

                # Combine grouth truth and Semantic predicted images to SAVE Results
                gxSD = np.ones((gxStatic.shape[2], gxStatic.shape[3], 3), dtype = np.uint8)*255
                gxSD[ori.astype(bool)]=[0,0,255] # ground Truth in RED 
                gxSD[gxDyn.astype(bool)]=[255,0,0] # Semantic predictions in BLUE
                cv2.imwrite(f"{testimgpath}/gt_full_pd_Dyn{i}.jpg", gxSD )

                # SAVE the combined image in COLOR
                gxSD = np.ones((gxStatic.shape[2], gxStatic.shape[3], 3), dtype = np.uint8)*255
                gxSD[gxDyn.astype(bool)]=[0,255,0] # Semantic predictions in GREEN
                #cv2.imwrite(f"{testimgpath}/pd_Dyn_color{i}.jpg", gxSD )
                
                gxSD[gxSta.astype(bool)]=[255,0,0] # static predictions in BLUE
                cv2.imwrite(f"{testimgpath}/pd_Static_Dyn_color{i}.jpg", gxSD )

                # Combine Static and Semantic predictions to SAVE Results
                gxStatDyn = (1-gxSemantic) + (1-gxStatic)
                gxStatDyn = (gxStatDyn<0.6).float()
                #cv2.imwrite(f"{testimgpath}/pd_StatDyn_gray{i}.jpg", gxStatDyn[0][0].cpu().detach().numpy()*255 )

                # F1 score for COMBINED static and Semantic predictions
                bwOriginal = np.uint8(bwOriginal[0][0].cpu().detach().numpy() * 255).ravel()
                gxStatDyn  = np.uint8(gxStatDyn[0][0].cpu().detach().numpy() * 255).ravel()
                f1_scores[i] = f1_score(bwOriginal, gxStatDyn, average='micro')
                ssim = structural_similarity(bwOriginal, gxStatDyn) # SSIM
                #print('\n SSIM is', ssim)
                psnr = peak_signal_noise_ratio(bwOriginal, gxStatDyn)
                print('\n PSNR is', psnr)
                f1_scores_list[i].append(f1_scores[i])
            
            if idx % args.debug_itrDisplay_interval == 0 :
                print("\n----------------------------------------------------------")
                print('Current sequence no. ',idx)
                print('Total loss for static+Semantic is ', float(loss))
                for i in range(output_len):
                    avg_f1_scores[i]= np.mean(f1_scores_list[i])
                    std_dev_f1_scores[i] = np.std(f1_scores_list[i])
                    mean_lossStatic_perScene[i] = np.mean(lossStatic_perScene_list[i])
                    stdDev_lossStatic_perScene[i] = np.std(lossStatic_perScene_list[i])
                    mean_lossSemantic_perScene[i] = np.mean(lossSemantic_perScene_list[i])
                    stdDev_lossSemantic_perScene[i] = np.std(lossSemantic_perScene_list[i])

                avg_f1_scores = [round(num, 4) for num in avg_f1_scores]
                std_dev_f1_scores = [round(num, 5) for num in std_dev_f1_scores]
                print("\nAverage f1 scores are ", avg_f1_scores)
                print("Std deviation of f1 scores are ", std_dev_f1_scores)

                mean_lossStatic_perScene = [round(num, 5) for num in mean_lossStatic_perScene]
                stdDev_lossStatic_perScene = [round(num, 5) for num in stdDev_lossStatic_perScene]
                print("\nAverage Static loss per scene ", mean_lossStatic_perScene)
                print("Std deviation of Static loss per scenes are ", stdDev_lossStatic_perScene)

                mean_lossSemantic_perScene = [round(num, 5) for num in mean_lossSemantic_perScene]
                stdDev_lossSemantic_perScene = [round(num, 5) for num in stdDev_lossSemantic_perScene]
                print("\nAverage Semantic loss per scene ", mean_lossSemantic_perScene)
                print("Std deviation of Semantic loss per scenes are ", stdDev_lossSemantic_perScene)

                # Mean and STD of static and Semantic loss
                print("\nMean of overall Static loss is ", np.mean(total_loss_static))
                print("Std deviation of overall Static loss is ", np.std(total_loss_static))
                print("\nMean of overall Semantic loss is ", np.mean(total_loss_Semantic))
                print("Std deviation of overall Semantic loss is ", np.std(total_loss_Semantic))

                f1_scores = [round(num, 3) for num in f1_scores]
                print("\nF1 scores for this sequence are ", f1_scores)

                time_elapsed = int(round(time.time())) - start
                print(datetime.datetime.now().strftime('\n%Y-%m-%d %H:%M:%S'))
                print("Batches done are ", idx)
                print("Total testing time has been ", time_elapsed, " seconds")
            

#   Testing: Standard RNN with separate Static and Full image scenes as input and predicting separate Static and Semantic objects 
def test_StandardLSTM_StaticFull(args, network):
    MSE_criterion = nn.MSELoss()
    device = torch.device(args.device)
    # Input: Static and full image scenes, Predictions: Static and Semantic scenes
    start = int(round(time.time()))
    time_elapsed = 0
    print("\n| Testing the model trained on Standard LSTM block ")
    print("\n| Testing the separate Static and Full image scenes model with Static and Semantic as output")
    print("\n| Loading the pre-trained model from ", args.pretrained_network)
    stats = torch.load(args.pretrained_network)
    network.load_state_dict(stats['net_param'])
    network.eval()

    # Initialize the TEST dataset
    test = NuscenesDataset(args.testdata, args) 
    print("\n| Total length of Testing dataset sequences is ", len(test))
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=4)
    f1_scores, ssim_scores, psnr_scores = [], [], []
    f1_scores_list, avg_f1_scores, std_dev_f1_scores = [], [], []
    ssim_scores_list, avg_ssim_scores, std_ssim_scores = [], [], []
    psnr_scores_list, avg_psnr_scores, std_psnr_scores = [], [], []
    total_loss_static, total_loss_Semantic = [], []   
    lossStatic_perScene_list, lossSemantic_perScene_list = [], []
    mean_lossStatic_perScene, mean_lossSemantic_perScene = [], []
    stdDev_lossStatic_perScene, stdDev_lossSemantic_perScene = [], []

    for i in range(args.seq_len - args.input_len):
        f1_scores.append(0)
        ssim_scores.append(0)
        psnr_scores.append(0)
        avg_f1_scores.append(0)
        std_dev_f1_scores.append(0)
        avg_ssim_scores.append(0)
        std_ssim_scores.append(0)
        avg_psnr_scores.append(0)
        std_psnr_scores.append(0)
        f1_scores_list.append([])
        ssim_scores_list.append([])
        psnr_scores_list.append([])
        lossStatic_perScene_list.append([])
        lossSemantic_perScene_list.append([])
        mean_lossStatic_perScene.append(0)
        stdDev_lossStatic_perScene.append(0)
        mean_lossSemantic_perScene.append(0)
        stdDev_lossSemantic_perScene.append(0)

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            
            staticgridTensor, SemanticgridTensor, bwtensor =  batch["staticgridTensor"].to(device), batch["SemanticgridTensor"].to(device), batch["bwTensor"].to(device)
            # pass the sequence tensor to Network
            predictions_static, predictions_full = network(staticgridTensor, SemanticgridTensor, idx)

            # Considering Static, Semantic and Total loss  
            loss_static = MSE_criterion(predictions_static, staticgridTensor[:, 1:]) 
            loss_Semantic = MSE_criterion(predictions_full, SemanticgridTensor[:, 1:]) 
            loss = loss_static + args.k_loss * loss_Semantic
            total_loss_static.append(float(loss_static))
            total_loss_Semantic.append(float(loss_Semantic))

            output_len = args.seq_len - args.input_len
            imgoutStatic = predictions_static[:, -output_len:]
            imgoutSemantic = predictions_full[:, -output_len:]
            testimgpath = os.path.join(args.savetestimages, str(idx))
            os.makedirs(testimgpath, exist_ok=True)

            for i in range(output_len):
                xStatic = staticgridTensor[:, i+args.input_len , :, :, :].permute(0,3,1,2) 
                gxStatic =  imgoutStatic[:, i, :, :, :].permute(0,3,1,2) 
                # Accumulate total Static loss per scene
                lossStatic_perScene_list[i].append(float(MSE_criterion(gxStatic, xStatic))) 
                gxStatic = (gxStatic>0.6).float()
                #cv2.imwrite(f"{testimgpath}/pd_Static{i}.jpg", gxStatic[0][0].cpu().detach().numpy()*255 )

                xSemantic = SemanticgridTensor[:, i+args.input_len , :, :, :].permute(0,3,1,2) 
                gxSemantic =  imgoutSemantic[:, i, :, :, :].permute(0,3,1,2) 
                # Accumulate total Semantic loss per scene
                lossSemantic_perScene_list[i].append(float(MSE_criterion(gxSemantic, xSemantic)))
                gxSemantic = (gxSemantic>0.6).float()
                #cv2.imwrite(f"{testimgpath}/pd_Semantic{i}.jpg", gxSemantic[0][0].cpu().detach().numpy()*255 )

                bwOriginal = bwtensor[:, i+args.input_len , :, :, :].permute(0,3,1,2) 
                ori = 1-bwOriginal[0][0].cpu().detach().numpy()
                gxDyn = 1-gxSemantic[0][0].cpu().detach().numpy()
                gxSta = 1-gxStatic[0][0].cpu().detach().numpy()

                # Combine grouth truth and Semantic predicted images to SAVE Results
                gxSD = np.ones((gxStatic.shape[2], gxStatic.shape[3], 3), dtype = np.uint8)*255
                gxSD[ori.astype(bool)]=[0,0,255] # ground Truth in RED 
                gxSD[gxDyn.astype(bool)]=[255,0,0] # Semantic predictions in BLUE
                #cv2.imwrite(f"{testimgpath}/gt_full_pd_Dyn{i}.jpg", gxSD )

                # SAVE the combined image in COLOR
                gxSD = np.ones((gxStatic.shape[2], gxStatic.shape[3], 3), dtype = np.uint8)*255
                gxSD[gxDyn.astype(bool)]=[0,255,0] # Semantic predictions in GREEN
                #cv2.imwrite(f"{testimgpath}/pd_Dyn_color{i}.jpg", gxSD )
                
                gxSD[gxSta.astype(bool)]=[255,0,0] # static predictions in BLUE
                #cv2.imwrite(f"{testimgpath}/pd_Static_Dyn_color{i}.jpg", gxSD )

                # Combine Static and Semantic predictions to SAVE Results
                gxStatDyn = (1-gxSemantic) + (1-gxStatic)
                gxStatDyn = (gxStatDyn<0.6).float()
                #cv2.imwrite(f"{testimgpath}/pd_StatDyn_gray{i}.jpg", gxStatDyn[0][0].cpu().detach().numpy()*255 )

                # F1 score for COMBINED static and Semantic predictions
                bwOriginal = np.uint8(bwOriginal[0][0].cpu().detach().numpy() * 255).ravel()
                gxStatDyn  = np.uint8(gxStatDyn[0][0].cpu().detach().numpy() * 255).ravel()
                f1_scores[i] = f1_score(bwOriginal, gxStatDyn, average='micro')
                f1_scores_list[i].append(f1_scores[i])
                ssim_scores[i] = structural_similarity(bwOriginal, gxStatDyn, win_size=11) # SSIM
                ssim_scores_list[i].append(ssim_scores[i])
                psnr_scores[i] = peak_signal_noise_ratio(bwOriginal, gxStatDyn)
                psnr_scores_list[i].append(psnr_scores[i])
            
            if idx % args.debug_itrDisplay_interval == 0 :
                print("\n----------------------------------------------------------")
                print('Current sequence no. ',idx)
                print('Total loss for static+Semantic is ', float(loss))
                for i in range(output_len):
                    avg_f1_scores[i]= np.mean(f1_scores_list[i])
                    std_dev_f1_scores[i] = np.std(f1_scores_list[i])
                    avg_ssim_scores[i] = np.mean(ssim_scores_list[i])
                    std_ssim_scores[i] = np.std(ssim_scores_list[i])
                    avg_psnr_scores[i] = np.mean(psnr_scores_list[i])
                    std_psnr_scores[i] = np.std(psnr_scores_list[i])
                    mean_lossStatic_perScene[i] = np.mean(lossStatic_perScene_list[i])
                    stdDev_lossStatic_perScene[i] = np.std(lossStatic_perScene_list[i])
                    mean_lossSemantic_perScene[i] = np.mean(lossSemantic_perScene_list[i])
                    stdDev_lossSemantic_perScene[i] = np.std(lossSemantic_perScene_list[i])

                avg_f1_scores = [round(num, 4) for num in avg_f1_scores]
                std_dev_f1_scores = [round(num, 5) for num in std_dev_f1_scores]
                print("\nAverage f1 scores are ", avg_f1_scores)
                print("Std deviation of f1 scores are ", std_dev_f1_scores)

                avg_ssim_scores = [round(num, 4) for num in avg_ssim_scores]
                std_ssim_scores = [round(num, 5) for num in std_ssim_scores]
                print("\nAverage ssim scores are ", avg_ssim_scores)
                print("Std deviation of ssim scores are ", std_ssim_scores)

                avg_psnr_scores = [round(num, 4) for num in avg_psnr_scores]
                std_psnr_scores = [round(num, 5) for num in std_psnr_scores]
                print("\nAverage psnr scores are ", avg_psnr_scores)
                print("Std deviation of psnr scores are ", std_psnr_scores)

                mean_lossStatic_perScene = [round(num, 5) for num in mean_lossStatic_perScene]
                stdDev_lossStatic_perScene = [round(num, 5) for num in stdDev_lossStatic_perScene]
                print("\nAverage Static loss per scene ", mean_lossStatic_perScene)
                print("Std deviation of Static loss per scenes are ", stdDev_lossStatic_perScene)

                mean_lossSemantic_perScene = [round(num, 5) for num in mean_lossSemantic_perScene]
                stdDev_lossSemantic_perScene = [round(num, 5) for num in stdDev_lossSemantic_perScene]
                print("\nAverage Semantic loss per scene ", mean_lossSemantic_perScene)
                print("Std deviation of Semantic loss per scenes are ", stdDev_lossSemantic_perScene)

                # Mean and STD of static and Semantic loss
                print("\nMean of overall Static loss is ", np.mean(total_loss_static))
                print("Std deviation of overall Static loss is ", np.std(total_loss_static))
                print("\nMean of overall Semantic loss is ", np.mean(total_loss_Semantic))
                print("Std deviation of overall Semantic loss is ", np.std(total_loss_Semantic))

                f1_scores = [round(num, 3) for num in f1_scores]
                print("\nF1 scores for this sequence are ", f1_scores)

                time_elapsed = int(round(time.time())) - start
                print(datetime.datetime.now().strftime('\n%Y-%m-%d %H:%M:%S'))
                print("Batches done are ", idx)
                print("Total testing time has been ", time_elapsed, " seconds")