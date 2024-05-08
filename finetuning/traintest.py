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
    loss_meter = AverageMeter()
    per_sample_dnn_time = AverageMeter()
    train_acc_meter = AverageMeter()
    progress = []
    best_epoch, best_acc = 0, -np.inf
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
    loss_fn = nn.TripletMarginLoss()

    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")

    audio_model.train()

    # training until break
    while epoch < args.n_epochs + 1:
        begin_time = time.time()
        end_time = time.time()
        audio_model.train()
        print(datetime.datetime.now())

        # save from-scratch models before the first epoch
        torch.save(audio_model.state_dict(), "%s/models/audio_model.%d.pth" % (exp_dir, global_step+1))

        for i, (audio_input, audio_input_two, label) in enumerate(train_loader):
            # measure data loading time
            B = audio_input.size(0) # batch size
            audio_input = audio_input.to(device, non_blocking=True)

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
            # TODO: Change Task argument
            output_anchor = audio_model(audio_input, 'finetuning_avg', mask_patch=args.mask_patch, cluster=cluster)
            output_two = audio_model(audio_input_two, 'finetuning_avg', mask_patch=args.mask_patch, cluster=cluster)

            # calc loss
            losses = []
            
            for anchor_idx in range(len(output_anchor)):
                anchor = output_anchor[anchor_idx]
                for two_idx in range(len(output_two)):
                    if two_idx != anchor_idx:
                        loss = loss_fn(anchor, output_two[anchor_idx], output_two[two_idx])
                        losses.append(loss)
            
            # Convert the list of losses to a tensor
            loss_tensor = torch.stack(losses)        
                    
            loss_sum = loss_tensor.sum()

            optimizer.zero_grad()
            loss_sum.backward()
            optimizer.step()
            
            scheduler.step()

            # record loss
            #train_acc_meter.update(acc.detach().cpu().item())
            loss_meter.update(loss.item(), B)
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
                  'Train Loss {loss_meter.val:.4f}\t'.format(
                   epoch, i, len(train_loader), per_sample_time=per_sample_time, per_sample_data_time=per_sample_data_time,
                      per_sample_dnn_time=per_sample_dnn_time, loss_meter=loss_meter), flush=True)
                if np.isnan(loss_meter.avg):
                    print("training diverged...")
                    return

            global_step += 1

            # save the model every args.epoch_iter steps.
            epoch_iteration = args.epoch_iter
            if global_step % epoch_iteration == 0:
                
                equ_epoch = int(global_step/epoch_iteration) + 1 # => global_step = epoch_iteration * (equ_epoch - 1)

                # Evaluate
                # Check if best model
                #torch.save(audio_model.state_dict(), "%s/models/best_audio_model.pth" % (exp_dir))

                torch.save(audio_model.state_dict(), "%s/models/audio_model.%d.pth" % (exp_dir, equ_epoch))
                if len(train_loader.dataset) > 2e5:
                    torch.save(optimizer.state_dict(), "%s/models/optim_state.pth" % (exp_dir)) # save optimizer state

                #scheduler.step(acc_eval)

                #print('# {:d}, step {:d}-{:d}, lr: {:e}'.format(equ_epoch, global_step-epoch_iteration, global_step, optimizer.param_groups[0]['lr']))

                _save_progress()

                finish_time = time.time()
                print('# {:d}, step {:d}-{:d}, training time: {:.3f}'.format(equ_epoch, global_step-epoch_iteration, global_step, finish_time-begin_time))
                begin_time = time.time()

                train_acc_meter.reset()
                batch_time.reset()
                per_sample_time.reset()
                data_time.reset()
                per_sample_data_time.reset()
                loss_meter.reset()
                per_sample_dnn_time.reset()

                # change the models back to train mode
                audio_model.train()
                print('---------------- evaluation finished ----------------')

            end_time = time.time()

        epoch += 1


def validate(audio_model, val_loader, args, epoch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    # switch to evaluate mode
    audio_model.eval()

    A_accv = []
    A_loss1v = [] # InfoNCE
    A_loss2v = [] # MSE
    with torch.no_grad():
        for i, (audio_input, _) in enumerate(val_loader):
            audio_input = audio_input.to(device)

            # use cluster masking only when masking patches, not frames
            cluster = (args.num_mel_bins != args.fshape)

            # always use mask_patch=400 for evaluation, even the training mask patch number differs.
            acc, loss1v = audio_model(audio_input, 'pretrain_mpc', mask_patch=400, cluster=cluster)
            loss2v = audio_model(audio_input, 'pretrain_mpg', mask_patch=400, cluster=cluster)

            A_accv.append(torch.mean(acc).cpu())
            A_loss1v.append(torch.mean(loss1v).cpu())
            A_loss2v.append(torch.mean(loss2v).cpu())

        acc = np.mean(A_accv)
        loss1v = np.mean(A_loss1v)
        loss2v = np.mean(A_loss2v)

    return acc, loss1v, loss2v