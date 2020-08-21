import torch
from torch.nn import functional as F
from torch.optim import lr_scheduler

from os.path import join

from utils import topk_accuracy, ValueMeter
from warmup_scheduler import GradualWarmupScheduler


def log(mode, epoch, loss_meter, accuracy_meter, best_perf=None, green=False):
    if green:
        print('\033[92m', end="")

    print(
        f"[{mode}] Epoch: {epoch:0.2f}. "
        f"Loss: {loss_meter.value():.2f}. "
        f"Accuracy: {accuracy_meter.value():.2f}% ", end="")

    if best_perf:
        print(f"[best: {best_perf:0.2f}]%", end="")

    print('\033[0m')

def save_model(args, model, epoch, perf, best_perf, is_best=False):
    torch.save({'state_dict': model.state_dict(), 'epoch': epoch,
                'perf': perf, 'best_perf': best_perf}, join(args.save_path, args.exp_name + '.pth.tar'))
    if is_best:
        torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'perf': perf, 'best_perf': best_perf}, join(
            args.save_path, args.exp_name + '_best.pth.tar'))

def train(args, model, loaders, optimizer, epochs, start_epoch, start_best_perf, trainval_logger, writer):

    feature_criterion_sl1 = torch.nn.SmoothL1Loss().to(args.device)
    feature_criterion_sl2 = torch.nn.MSELoss().to(args.device)
    f_log_softmax = torch.nn.LogSoftmax()

    """Training/Validation code"""
    best_perf = start_best_perf  # to keep track of the best performing epoch
    for epoch in range(start_epoch, epochs):
        # define training and validation meters
        loss_meter = {'training': ValueMeter(), 'validation': ValueMeter()}
        annot_loss_meter = {'training': ValueMeter(), 'validation': ValueMeter()}
        feature_loss_meter = {'training': ValueMeter(), 'validation': ValueMeter()}
        verb_loss_meter = {'training': ValueMeter(), 'validation': ValueMeter()}
        noun_loss_meter = {'training': ValueMeter(), 'validation': ValueMeter()}
        accuracy_meter = {'training': ValueMeter(), 'validation': ValueMeter()}
        
        for mode in ['training', 'validation']:
            # enable gradients only if training
            with torch.set_grad_enabled(mode == 'training'):
                if mode == 'training':
                    model.train()
                else:
                    model.eval()

                for i, batch in enumerate(loaders[mode]):
                    
                    x = batch['past_features' if args.task == 'anticipation' else 'action_features']

                    if type(x) == list:
                        x = [xx.to(args.device) for xx in x]
                    else:
                        x = x.to(args.device)

                    label_temp = batch['label'].to(args.device)
                    verb_goal = label_temp[:, 0]
                    noun_goal = label_temp[:, 1]
                    y = label_temp[:, 2]
                    feature_labels = x.contiguous()

                    bs = y.shape[0]  # batch size
                    
                    preds, feature_preds, verb_preds, noun_preds = model(x)#, verb_embedding, noun_embedding)

                    ## anticipation loss
                    preds = preds[:, -args.S_ant:, :].contiguous()
                    linear_preds = preds.view(-1, preds.shape[-1])
                    linear_labels = y.view(-1, 1).expand(-1, preds.shape[1]).contiguous().view(-1)
                    ant_loss = F.cross_entropy(linear_preds, linear_labels)
                    
                    ## reinforce loss
                    verb_preds = verb_preds[:, -args.S_ant:, :].contiguous()
                    noun_preds = noun_preds[:, -args.S_ant:, :].contiguous()
                    linear_verb_preds = verb_preds.view(-1, verb_preds.shape[-1])
                    linear_verb_labels = verb_goal.view(-1, 1).expand(-1, verb_preds.shape[1]).contiguous().view(-1)
                    reinforce_verb_loss = F.cross_entropy(linear_verb_preds, linear_verb_labels)

                    linear_noun_preds = noun_preds.view(-1, noun_preds.shape[-1])
                    linear_noun_labels = noun_goal.view(-1, 1).expand(-1, noun_preds.shape[1]).contiguous().view(-1)
                    reinforce_noun_loss = F.cross_entropy(linear_noun_preds, linear_noun_labels)
        
                    ## revision loss
                    sample_features = batch['all_sample_features'].to(args.device)
                    sample_features = F.normalize(sample_features, p=2, dim=2)
        
                    feature_preds = feature_preds[:, -36:, :]
                    feature_preds = F.normalize(feature_preds, p=2, dim=2)
                    
                    x_norm =  F.normalize(x, p=2, dim=2)
                    
                    batch_size = feature_preds.size(0)
                    feature_size = feature_preds.size(2)
                    feature_loss = 0
                    global_indx = 0
                    
                    for indx in range(6, x.size(1), 1):
                        for temp_indx in range(indx, x_norm.size(1)):
                            pred_feature = feature_preds[:, global_indx, :]
                            global_indx += 1

                            gt_feature = x_norm[:, temp_indx, :]

                            negative_feature = sample_features[:, temp_indx-6, :]
                        
                            logit_pos = torch.bmm(pred_feature.view(batch_size, 1, feature_size), gt_feature.view(batch_size, feature_size, 1)).squeeze(2)
                            logit_neg = torch.bmm(pred_feature.unsqueeze(1), negative_feature.permute(0, 2, 1)).squeeze(1)
                            logit = torch.cat((logit_pos, logit_neg), 1)
                            labels = torch.zeros(batch_size).long().to(args.device)

                            feature_loss += F.cross_entropy(logit, labels)
                    
                    feature_loss = feature_loss / (global_indx + 1)
                    
                    ## total loss
                    loss = ant_loss + args.revision_weight * feature_loss + args.reinforce_verb_weight * reinforce_verb_loss + args.reinforce_noun_weight * reinforce_noun_loss

                    acc = topk_accuracy(preds[:, -4, :].detach().cpu().numpy(), y.detach().cpu().numpy(), (5,))[0]*100

                    # store the values in the meters to keep incremental averages
                    loss_meter[mode].add(loss.item(), bs)
                    annot_loss_meter[mode].add(ant_loss.item(), bs)
                    feature_loss_meter[mode].add(feature_loss.item(), bs)
                    verb_loss_meter[mode].add(reinforce_verb_loss.item(), bs)
                    noun_loss_meter[mode].add(reinforce_noun_loss.item(), bs)
                    accuracy_meter[mode].add(acc, bs)

                    writer.add_scalar( mode + '/total_loss_iter', loss_meter[mode].value(), i + 1 + len(loaders[mode]) * (epoch - 1))
                    writer.add_scalar( mode + '/annot_loss_iter', annot_loss_meter[mode].value(), i + 1 + len(loaders[mode]) * (epoch - 1))
                    writer.add_scalar( mode + '/feature_loss_iter', feature_loss_meter[mode].value(), i + 1 + len(loaders[mode]) * (epoch - 1))
                    writer.add_scalar( mode + '/verb_loss_iter', verb_loss_meter[mode].value(), i + 1 + len(loaders[mode]) * (epoch - 1))
                    writer.add_scalar( mode + '/noun_loss_iter', noun_loss_meter[mode].value(), i + 1 + len(loaders[mode]) * (epoch - 1))
                    writer.add_scalar( mode + '/accuracy_iter', accuracy_meter[mode].value(), i + 1 + len(loaders[mode]) * (epoch - 1))
                    
                    # if in training mode
                    if mode == 'training':
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_value)
                        optimizer.step()

                    # compute decimal epoch for logging
                    e = epoch + i/len(loaders[mode])

                    # log training during loop
                    if mode == 'training' and i != 0 and i % args.display_every == 0:
                        log(mode, e, loss_meter[mode], accuracy_meter[mode])

                # log at the end of each epoch
                log(mode, epoch+1, loss_meter[mode], accuracy_meter[mode], max(accuracy_meter[mode].value(), best_perf) if mode == 'validation' else None, green=True)

        trainval_logger.log({
            'epoch': epoch,
            'train_loss': loss_meter['training'].value(),
            'val_loss': loss_meter['validation'].value(),
            'train_accuracy': accuracy_meter['training'].value(),
            'val_accuracy': accuracy_meter['validation'].value(),
        })

        writer.add_scalar( mode + '/total_loss_epoch', loss_meter[mode].value(), epoch)
        writer.add_scalar( mode + '/annot_loss_epoch', annot_loss_meter[mode].value(), epoch)
        writer.add_scalar( mode + '/feature_loss_epoch', feature_loss_meter[mode].value(), epoch)
        writer.add_scalar( mode + '/verb_loss_epoch', verb_loss_meter[mode].value(), epoch)
        writer.add_scalar( mode + '/noun_loss_epoch', noun_loss_meter[mode].value(), epoch)
        writer.add_scalar( mode + '/accuracy_epoch', accuracy_meter[mode].value(), epoch)
        
        if best_perf < accuracy_meter['validation'].value():
            best_perf = accuracy_meter['validation'].value()
            is_best = True
        else:
            is_best = False

        # save checkpoint at the end of each train/val epoch
        save_model(args, model, epoch+1, accuracy_meter['validation'].value(), best_perf, is_best=is_best)
