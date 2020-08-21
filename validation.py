import torch
from utils import topk_accuracy, topk_accuracy_multiple_timesteps, get_marginal_indexes, marginalize, softmax,  topk_recall_multiple_timesteps
import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import join


def get_many_shot(args):
    """Get many shot verbs, nouns and actions for class-aware metrics (Mean Top-5 Recall)"""
    # read the list of many shot verbs
    many_shot_verbs = pd.read_csv(
        join(args.path_to_data, 'EPIC_many_shot_verbs.csv'))['verb_class'].values
    # read the list of many shot nouns
    many_shot_nouns = pd.read_csv(
        join(args.path_to_data, 'EPIC_many_shot_nouns.csv'))['noun_class'].values

    # read the list of actions
    actions = pd.read_csv(join(args.path_to_data, 'actions.csv'))
    # map actions to (verb, noun) pairs
    a_to_vn = {a[1]['id']: tuple(a[1][['verb', 'noun']].values)
               for a in actions.iterrows()}

    # create the list of many shot actions
    # an action is "many shot" if at least one
    # between the related verb and noun are many shot
    many_shot_actions = []
    for a, (v, n) in a_to_vn.items():
        if v in many_shot_verbs or n in many_shot_nouns:
            many_shot_actions.append(a)

    return many_shot_verbs, many_shot_nouns, many_shot_actions

def get_scores_early_recognition_fusion(args, models, loaders):
    verb_scores = 0
    noun_scores = 0
    action_scores = 0
    for model, loader in zip(models, loaders):
        outs = get_scores(args, model, loader)
        verb_scores += outs[0]
        noun_scores += outs[1]
        action_scores += outs[2]

    verb_scores /= len(models)
    noun_scores /= len(models)
    action_scores /= len(models)

    return [verb_scores, noun_scores, action_scores] + list(outs[3:])

def get_scores(args, model, loader):
    model.eval()
    predictions = []
    labels = []
    ids = []
    with torch.set_grad_enabled(False):
        for batch in tqdm(loader, 'Evaluating...', len(loader)):
            x = batch['past_features' if args.task ==
                      'anticipation' else 'action_features']
            if type(x) == list:
                x = [xx.to(args.device) for xx in x]
            else:
                x = x.to(args.device)

            y = batch['label'].numpy()
            ids.append(batch['id'].numpy())
            
            preds = model(x)
            preds = preds[0].cpu().numpy()[:, -args.S_ant:, :]

            predictions.append(preds)
            labels.append(y)

    action_scores = np.concatenate(predictions)
    labels = np.concatenate(labels)
    ids = np.concatenate(ids)

    actions = pd.read_csv(
        join(args.path_to_data, 'actions.csv'), index_col='id')

    vi = get_marginal_indexes(actions, 'verb')
    ni = get_marginal_indexes(actions, 'noun')

    action_probs = softmax(action_scores.reshape(-1, action_scores.shape[-1]))

    verb_scores = marginalize(action_probs, vi).reshape(
        action_scores.shape[0], action_scores.shape[1], -1)
    noun_scores = marginalize(action_probs, ni).reshape(
        action_scores.shape[0], action_scores.shape[1], -1)

    
    if labels.max()>0:
        return verb_scores, noun_scores, action_scores, labels[:, 0], labels[:, 1], labels[:, 2]
    else:
        return verb_scores, noun_scores, action_scores, ids

def validation(args, model, loader):
    verb_scores, noun_scores, action_scores, verb_labels, noun_labels, action_labels = get_scores(args, model, loader)

    verb_accuracies = topk_accuracy_multiple_timesteps(
        verb_scores, verb_labels)
    noun_accuracies = topk_accuracy_multiple_timesteps(
        noun_scores, noun_labels)
    action_accuracies = topk_accuracy_multiple_timesteps(
        action_scores, action_labels)

    many_shot_verbs, many_shot_nouns, many_shot_actions = get_many_shot(args)

    verb_recalls = topk_recall_multiple_timesteps(
        verb_scores, verb_labels, k=5, classes=many_shot_verbs)
    noun_recalls = topk_recall_multiple_timesteps(
        noun_scores, noun_labels, k=5, classes=many_shot_nouns)
    action_recalls = topk_recall_multiple_timesteps(
        action_scores, action_labels, k=5, classes=many_shot_actions)

    all_accuracies = np.concatenate(
        [verb_accuracies, noun_accuracies, action_accuracies, verb_recalls, noun_recalls, action_recalls])
    all_accuracies = all_accuracies[[0, 1, 6, 2, 3, 7, 4, 5, 8]]
    indices = [
        ('Verb', 'Top-1 Accuracy'),
        ('Verb', 'Top-5 Accuracy'),
        ('Verb', 'Mean Top-5 Recall'),
        ('Noun', 'Top-1 Accuracy'),
        ('Noun', 'Top-5 Accuracy'),
        ('Noun', 'Mean Top-5 Recall'),
        ('Action', 'Top-1 Accuracy'),
        ('Action', 'Top-5 Accuracy'),
        ('Action', 'Mean Top-5 Recall'),
    ]

    cc = np.linspace(args.alpha*args.S_ant, args.alpha, args.S_ant, dtype=str)
    scores = pd.DataFrame(all_accuracies*100, columns=cc, index=pd.MultiIndex.from_tuples(indices))
    print(scores)
