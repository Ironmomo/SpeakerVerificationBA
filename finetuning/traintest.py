#  Original Author: Yuan Gong, 2021, Massachusetts Institute of Technology
#  Edited by: Andrin Fassbind, Fabian Bosshard, 2024, Zurich University of Applied Sciences

import sys
import os
import datetime
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
from utilities import *
import time
import torch
import numpy as np
import pickle
from torch.optim.lr_scheduler import StepLR

def train(audio_model, train_loader, test_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Now running on : ' + str(device))

    # initialize all of the statistics we want to keep track of
    batch_time = AverageMeter()
    per_sample_time = AverageMeter()
    data_time = AverageMeter()
    per_sample_data_time = AverageMeter()
    train_loss_meter = AverageMeter()
    test_loss_meter = AverageMeter()
    per_sample_dnn_time = AverageMeter()
    progress = []
    best_epoch, best_loss = 0, np.inf
    global_step, epoch = 0, 0
    start_time = time.time()
    exp_dir = args.exp_dir

    def _save_progress():
        progress.append([epoch, global_step, best_epoch, time.time() - start_time])
        with open("%s/progress.pkl" % exp_dir, "wb") as f:
            pickle.dump(progress, f)

    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)

    audio_model = audio_model.to(device)
    
    # save from-scratch models before the first epoch
    torch.save(audio_model.state_dict(), "%s/models/audio_model.%d.pth" % (exp_dir, global_step+1))
    
    # Set up the optimizer
    audio_trainables = [p for p in audio_model.parameters() if p.requires_grad]
    print('Total parameter number is : {:.9f} million'.format(sum(p.numel() for p in audio_model.parameters()) / 1e6))
    print('Total trainable parameter number is : {:.9f} million'.format(sum(p.numel() for p in audio_trainables) / 1e6))
    trainables = audio_trainables
    optimizer = torch.optim.Adam(trainables, args.lr, weight_decay=5e-7, betas=(0.95, 0.999))

    # LR scheduler
    scheduler = StepLR(optimizer=optimizer, step_size=10, gamma=0.05)
    epoch += 1

    # Loss function
    loss_fn = nn.CosineEmbeddingLoss()

    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")

    result = []
    audio_model.train()

    # training until break
    while epoch < args.n_epochs + 1:
        begin_time = time.time()
        end_time = time.time()
        audio_model.train()
        print(datetime.datetime.now())

        # batch loop
        for i, (audio_input, audio_input_two, label) in enumerate(train_loader):
            # measure data loading time
            B = audio_input.size(0) # batch size
            audio_input = audio_input.to(device, non_blocking=True)
            audio_input_two = audio_input_two.to(device)

            data_time.update(time.time() - end_time)
            per_sample_data_time.update((time.time() - end_time) / B)
            dnn_start_time = time.time()

            # first several steps for warm-up
            if global_step <= 1000 and global_step % 50 == 0:
                warm_lr = (global_step / 1000) * args.lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warm_lr
                print('warm-up learning rate is {:f}'.format(optimizer.param_groups[0]['lr']))

            # use cluster masking only when masking patches, not frames
            cluster = (args.num_mel_bins != args.fshape) 

            # calc embedding
            output_anchor = audio_model(audio_input, 'finetuning_avg', mask_patch=args.mask_patch, cluster=cluster)
            output_two = audio_model(audio_input_two, 'finetuning_avg', mask_patch=args.mask_patch, cluster=cluster)

            # calc loss
            target_pos = torch.ones(B).to(device)
            loss_pos = loss_fn(output_anchor, output_two, target_pos)
            
            # shuffle two
            perm = torch.randperm(B)
            output_two_shuffle = output_two[perm]
            output_two_shuffle = output_two_shuffle.to(device)
            
            label_shuffle = label[perm]
            # Check if tensors are equal element-wise
            equal_mask = torch.eq(label, label_shuffle)
        
            # Convert boolean mask to 1s and 0s
            equal_mask = equal_mask.float()

            # Sum along the second dimension to get a [B, 1] mask
            equal_mask = torch.sum(equal_mask, dim=1, keepdim=True)
            target_neg = torch.where(equal_mask == label.size(1), torch.ones_like(equal_mask), -torch.ones_like(equal_mask))
            target_neg = torch.squeeze(target_neg).to(device)
        
            
            loss_neg = loss_fn(output_anchor, output_two_shuffle, target_neg)
            
            loss_sum = loss_pos + loss_neg

            # Backpropagation
            optimizer.zero_grad()
            loss_sum.backward()
            optimizer.step()
            
            scheduler.step()

            # record loss
            train_loss_meter.update(loss_sum.item(), B)
            batch_time.update(time.time() - end_time)
            per_sample_time.update((time.time() - end_time) / B)
            per_sample_dnn_time.update((time.time() - dnn_start_time) / B)

            print_step = global_step % args.n_print_steps == 0
            early_print_step = epoch == 0 and global_step % (args.n_print_steps/10) == 0
            print_step = print_step or early_print_step

            if print_step and global_step != 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Per Sample Total Time {per_sample_time.avg:.5f}\t'
                  'Per Sample Data Time {per_sample_data_time.avg:.5f}\t'
                  'Per Sample DNN Time {per_sample_dnn_time.avg:.5f}\t'
                  'Train Loss {train_loss_meter.val:.4f}\t'
                  'Train Loss Avg {train_loss_meter.avg:.4f}\t'
                  .format(
                   epoch, i, len(train_loader), per_sample_time=per_sample_time, per_sample_data_time=per_sample_data_time,
                      per_sample_dnn_time=per_sample_dnn_time, train_loss_meter=train_loss_meter), flush=True)
                if np.isnan(train_loss_meter.avg):
                    print("training diverged...")
                    return

                result.append([train_loss_meter.avg, test_loss_meter.avg, optimizer.param_groups[0]['lr']])
                np.savetxt(exp_dir + '/result.csv', result, delimiter=',')
                
            global_step += 1

            # save the model every args.epoch_iter steps.
            epoch_iteration = args.epoch_iter
            if global_step % epoch_iteration == 0:
                
                equ_epoch = int(global_step/epoch_iteration) + 1 # => global_step = epoch_iteration * (equ_epoch - 1)

                # Evaluate
                test_loss = validate(audio_model=audio_model, val_loader=test_loader, args=args)
                
                test_loss_meter.update(test_loss)
                        
                # Check if best model
                if test_loss < best_loss:
                    best_loss = test_loss
                    best_epoch = epoch
                    torch.save(audio_model.state_dict(), "%s/models/best_audio_model.pth" % (exp_dir))

                torch.save(audio_model.state_dict(), "%s/models/audio_model.%d.pth" % (exp_dir, equ_epoch))
                if len(train_loader.dataset) > 2e5:
                    torch.save(optimizer.state_dict(), "%s/models/optim_state.pth" % (exp_dir)) # save optimizer state


                print('EVALUATION - Epoch: [{0}]\t'
                    'Test Loss val {test_loss:.4f}\t'
                  'Test Loss val {test_loss_meter.val:.4f}\t'
                  'Test Loss Avg {test_loss_meter.avg:.4f}\t'
                  .format(
                   epoch, test_loss= test_loss, test_loss_meter=test_loss_meter), flush=True)
                
                _save_progress()

                finish_time = time.time()
                print('# {:d}, step {:d}-{:d}, training time: {:.3f}'.format(equ_epoch, global_step-epoch_iteration, global_step, finish_time-begin_time))
                begin_time = time.time()

                batch_time.reset()
                per_sample_time.reset()
                data_time.reset()
                per_sample_data_time.reset()
                train_loss_meter.reset()
                per_sample_dnn_time.reset()

                # change the models back to train mode
                audio_model.train()
                print('---------------- evaluation finished ----------------')

            end_time = time.time()
            
        # Evaluate
        test_loss = validate(audio_model=audio_model, val_loader=test_loader, args=args)
        
        test_loss_meter.update(test_loss)
        
        print('EVALUATION - Epoch: [{0}]\t'
                  'Test Loss val {test_loss_meter.val:.4f}\t'
                  'Test Loss Avg {test_loss_meter.avg:.4f}\t'
                  .format(
                   epoch, test_loss_meter=test_loss_meter), flush=True)

        epoch += 1


def validate(audio_model, val_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_loss_meter = AverageMeter()
    
    # Loss function
    loss_fn = nn.CosineEmbeddingLoss()
    
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    # switch to evaluate mode
    audio_model.eval()
    
    with torch.no_grad():
        for i, (audio_input, audio_input_two, label) in enumerate(val_loader):
            B = audio_input.size(0) # batch size
            
            audio_input = audio_input.to(device)
            audio_input_two = audio_input_two.to(device)
            label = label.to(device)

            # use cluster masking only when masking patches, not frames
            cluster = (args.num_mel_bins != args.fshape)
            
            # calc embedding
            output_anchor = audio_model(audio_input, 'finetuning_avg', mask_patch=args.mask_patch, cluster=cluster)
            output_two = audio_model(audio_input_two, 'finetuning_avg', mask_patch=args.mask_patch, cluster=cluster)

            # calc loss
            target_pos = torch.ones(B).to(device)
            loss_pos = loss_fn(output_anchor, output_two, target_pos)
            # shuffle two
            perm = torch.randperm(B)
            output_two_shuffle = output_two[perm]
            output_two_shuffle = output_two_shuffle.to(device)
            
            label_shuffle = label[perm]
            # Check if tensors are equal element-wise
            equal_mask = torch.eq(label, label_shuffle)

            # Convert boolean mask to 1s and 0s
            equal_mask = equal_mask.float()

            # Sum along the second dimension to get a [B, 1] mask
            equal_mask = torch.sum(equal_mask, dim=1, keepdim=True)
            target_neg = torch.where(equal_mask == label.size(1), torch.ones_like(equal_mask), -torch.ones_like(equal_mask))
            target_neg = torch.squeeze(target_neg).to(device)
            
            loss_neg = loss_fn(output_anchor, output_two_shuffle, target_neg)
            
            loss_sum = loss_pos + loss_neg
            # Convert the list of losses to a tensor
            test_loss_meter.update(loss_sum.item())
            

    return test_loss_meter.avg