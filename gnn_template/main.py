import fileinput
import gc
import os
import pathlib
import argparse
import torch
import torch.nn as nn
import numpy as np
import torch
import random
import json
import copy
import time
from scipy.special import softmax
import torch_geometric

import config
from config import args
import myDataset
import utils.draw as draw

if __name__ == '__main__':
    # Initialization is completed in config.py
    timeProgramStart = time.time()
    import train_test

    #Find device
    device = args.device
    
    # Initialize network, result log and optimizer
    net = copy.deepcopy(args.net)
    res = copy.deepcopy(args.reslog)
    args.optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    optimizer = args.optimizer
    
    args.reduce_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(args.optimizer, mode='min', factor=0.5, patience=8,
                 verbose=False, threshold=0.1, threshold_mode='rel',
                 cooldown=0, min_lr=1e-8, eps=1e-8)

    epoch_start = 1

    if args.pre_train != 0 and (not args.apply_only):
        net.load_state_dict(torch.load(args.pre_net, map_location=device))
        
        try:
            with open(args.pre_log,'r') as f:
                res = json.load(f)
            epoch_start = res['epochs'][-1]+1
        except:
            pass

    net.to(device)

    # Loss function 
    criterion = args.criterion

    # Record the smallest loss epoch
    bestLoss = 1e10
    bestEpoch = 0

    if args.apply_only == 0:
        # Loop epoch 
        for epoch in range(epoch_start, epoch_start + args.num_epochs):
            # Record test pred
            testpred = np.zeros(0)
            testOut = np.zeros(0)

            # Loop slices of data
            meanTrainLoss, meanTrainAcc = 0, 0
            meanLoss, meanAcc = 0, 0
            for data_slice_id in range(args.num_slices_train):
                # continue
                trainloader = myDataset.get_dataloader('train',data_slice_id, args.num_slices_train, args.data_size, args.batch_size)                
                if len(trainloader)==0:
                    continue
                # Log epoch and lr
                res['epochs'].append(epoch)
                res['lr'].append(optimizer.param_groups[0]['lr'])

                # Train
                res['train_slice'].append(data_slice_id)
                train_test.train_one_epoch(net, trainloader, criterion, optimizer, res)

                meanTrainLoss = meanTrainLoss + res['train_loss'][-1]
                meanTrainAcc = meanTrainAcc + res['train_acc'][-1]

                # Time usage
                if (data_slice_id+1) in [int(i/4*args.num_slices_train) for i in range(1,5)]:
                    print("Epoch %d/%d with lr %f, data_slice %d/%d training finished in %.2f min. Total time used: %.2f min." \
                            % (epoch, epoch_start+args.num_epochs-1, optimizer.param_groups[0]['lr'], data_slice_id+1, args.num_slices_train,\
                                (res['train_time'][-1])/60, (time.time() - timeProgramStart)/60), flush=True)
                
            # Test
            for data_slice_id in range(args.num_slices_test):
                res['test_slice'].append(data_slice_id)

                testloader = myDataset.get_dataloader('test',data_slice_id, args.num_slices_test, args.data_size, args.batch_size)
                ifcheckOutput = ( data_slice_id==args.num_slices_test-1 )
                pred_tmp, out_tmp = train_test.test_one_epoch(net, testloader, criterion, res, check_output = ifcheckOutput, save_new_graph_loc="./out/test/", save_new_graph_id=data_slice_id)

                if len(testpred) == 0:
                    testpred = pred_tmp
                    testOut = out_tmp
                else:
                    testpred = np.concatenate((testpred, pred_tmp))
                    testOut = np.concatenate((testOut, out_tmp))

                meanLoss = meanLoss + res['test_loss'][-1]
                meanAcc = meanAcc + res['test_acc'][-1]
                # Save result
                json_object = json.dumps(res, indent=4)
                with open(f"{args.logDir}/train-result.json", "w") as outfile:
                    outfile.write(json_object)

            meanTrainLoss = meanTrainLoss / args.num_slices_train
            meanTrainAcc = meanTrainAcc / args.num_slices_train
            meanLoss = meanLoss / args.num_slices_test
            meanAcc = meanAcc / args.num_slices_test
            args.reduce_schedule.step(meanLoss) 

            print(f"Test: epoch: {epoch}/{epoch_start+args.num_epochs-1}.")
            print(f"mean train loss: {meanTrainLoss}, mean train acc: {meanTrainAcc}")
            print(f"mean test loss: {meanLoss}, mean test acc: {meanAcc}")
            print("Total time used: %.2f min.\n"%((time.time() - timeProgramStart)/60))
            
            image_log_dir = args.logDir+'/images/'
            os.makedirs(image_log_dir, exist_ok=True)
            draw.draw_all(args.logDir, image_log_dir, False)
            # draw.draw_loss_acc(res, args.logDir)

            # Record the best epoch
            if meanLoss < bestLoss:
                bestLoss = meanLoss
                bestEpoch = epoch
                torch.save(net.state_dict(), f'{args.logDir}/net.pt')

            if True:
                # save pred given by network during test
                np.save(args.logDir+f'/predTest_GPU0.npy', arr=testpred)
                np.save(args.logDir+f'/outTest_GPU0.npy', arr=testOut)
                torch.save(net.state_dict(), f'{args.logDir}/net{epoch}.pt')

            print(f"Best epoch: {bestEpoch} with loss {bestLoss}")
            print("\n\n", flush=True)

    # Apply
    net = copy.deepcopy(args.net)
    if args.apply_net=="":
        net.load_state_dict(torch.load(f'{args.logDir}/net.pt', map_location=device))
    else:
        net.load_state_dict(torch.load(f'{args.apply_net}', map_location=device))

    net.to(device)
        
    applyRes = args.reslog.copy()
    pred = np.zeros(0)
    out = np.zeros(0)

    if len(args.apply_file_list)==0:        
        applyloader = myDataset.get_dataloader('apply', 0, args.num_slices_test, args.data_size, args.batch_size)
             
        pred, out = train_test.test_one_epoch(net, applyloader, criterion, applyRes)
        # save pred given by network during applying
        np.save(args.logDir+f'/predApply_GPU0.npy', arr=pred)
        np.save(args.logDir+f'/outApply_GPU0.npy', arr=out)

        os.makedirs(args.logDir+'/images', exist_ok=True)
        draw.draw_all(args.logDir, args.logDir+'/images/', True)
        print("\n\n")
        print("Apply finished.")
        print("Apply time %.2f min" % (sum(applyRes['test_time'])/60.))
        print("Apply loss: %.4f \t Apply acc: %.4f" % (sum(applyRes['test_loss'])/len(applyRes['test_loss']),
                                                        sum(applyRes['test_acc'])/len(applyRes['test_acc'])))        
        print("\nTotal time used: %.2f min.\n"%((time.time() - timeProgramStart)/60))
        
        image_log_dir = args.logDir+'/images/'
        os.makedirs(image_log_dir, exist_ok=True)
        draw.draw_all(args.logDir, image_log_dir, True)
    else:
        for iapply in range(len(args.apply_file_list)):
            infile_name = args.apply_file_list[iapply]
            print(f"\n {iapply+1}/{len(args.apply_file_list)}")
            print(infile_name)

            applyset = myDataset.MyDataset(infile_name)
            applyloader = torch_geometric.loader.DataLoader(applyset, batch_size=args.batch_size, shuffle=False)
            # applyloader = myDataset.get_dataloader('apply', iapply, args.num_slices_test, args.data_size, args.batch_size)
            pred, out = train_test.test_one_epoch(net, applyloader, criterion, applyRes)

            if len(out)>0:
                score = softmax(out[:,1:], axis=1)[:,1]
                save_file_name = os.path.splitext(infile_name)[0]
                save_file_name = save_file_name+f"_score{args.ifold}"+".npy"
                np.save(save_file_name, arr=score)
                print("Apply loss: %.4f \t Apply acc: %.4f" % (sum(applyRes['test_loss'])/len(applyRes['test_loss']),
                                                                sum(applyRes['test_acc'])/len(applyRes['test_acc'])))        
                print("Total time used: %.2f min.\n"%((time.time() - timeProgramStart)/60))
            else:
                print(f"Error: No data. Continue.", flush=True)

