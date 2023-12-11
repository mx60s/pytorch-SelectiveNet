import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import click
from collections import OrderedDict

import torch
import torchvision
import numpy as np
import json

from external.dada.flag_holder import FlagHolder
from external.dada.metric import MetricDict
from external.dada.io import print_metric_dict
from external.dada.io import save_model
from external.dada.io import load_model
from external.dada.logger import Logger

from selectivenet.heareval import HearEvalNN
from selectivenet.model import SelectiveNet
from selectivenet.loss import SelectiveLoss
from selectivenet.data import EmbeddingsDataset
from selectivenet.evaluator import Evaluator

@click.command()
# model
@click.option('--dim_features', type=int, default=1024)
@click.option('--dropout_prob', type=float, default=0.1)
@click.option('-w', '--weight', type=str, required=True, help='model weight directory')
@click.option('--weight_prefix', type=str, required=True, help='prefix of each model weight')
# data
@click.option('-d', '--dataset', type=str, required=False)
@click.option('--dataroot', type=str, default='../data', help='path to dataset root')
@click.option('-j', '--num_workers', type=int, default=8)
@click.option('-N', '--batch_size', type=int, default=128)
@click.option('--normalize', is_flag=True, default=True)
# loss
@click.option('--coverage', type=float, required=True)
@click.option('--alpha', type=float, default=0.5, help='balancing parameter between selective_loss and ce_loss')

def main(**kwargs):
    test(**kwargs)

def test(**kwargs):
    FLAGS = FlagHolder()
    FLAGS.initialize(**kwargs)
    FLAGS.summary()

    seed = 1017
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU.

    # dataset
    test = EmbeddingsDataset(FLAGS.dataroot, fold_name='test')

    print(len(test))

    num_classes = 12
    hidden_dim = 1024

    accumulative_metrics = MetricDict()

    # test
    predictions = []
    selections = []

    test_loader = torch.utils.data.DataLoader(test, batch_size=FLAGS.batch_size, shuffle=False, num_workers=FLAGS.num_workers, pin_memory=True)

    # model
    features = HearEvalNN(input_dim=FLAGS.dim_features)
    model = SelectiveNet(features, hidden_dim, num_classes).cuda()
    model_path = os.path.join(FLAGS.weight, f'{FLAGS.weight_prefix}.pth')
    load_model(model, model_path)

    if torch.cuda.device_count() > 1: model = torch.nn.DataParallel(model)

    # loss
    base_loss = torch.nn.CrossEntropyLoss(reduction='none')
    SelectiveCELoss = SelectiveLoss(base_loss, coverage=FLAGS.coverage)

    # pre epoch
    test_metric_dict = MetricDict()

    # test
    model.eval()
    with torch.autograd.no_grad():
        for i, (x,t) in enumerate(test_loader):
            x = x.to('cuda', non_blocking=True)
            t = t.to('cuda', non_blocking=True)

            # forward
            out_class, out_select, out_aux = model(x)

            # compute selective loss
            loss_dict = OrderedDict()
            # loss dict includes, 'empirical_risk' / 'emprical_coverage' / 'penulty'
            selective_loss, loss_dict = SelectiveCELoss(out_class, out_select, t)
            selective_loss *= FLAGS.alpha
            loss_dict['selective_loss'] = selective_loss.detach().cpu().item()
            # compute standard cross entropy loss
            ce_loss = torch.nn.CrossEntropyLoss()(out_aux, t)
            ce_loss *= (1.0 - FLAGS.alpha)
            loss_dict['ce_loss'] = ce_loss.detach().cpu().item()
            # total loss
            loss = selective_loss + ce_loss
            loss_dict['loss'] = loss.detach().cpu().item()

            # evaluation
            selection_out = out_class.detach() 
            evaluator = Evaluator(selection_out, t.detach(), out_select.detach())
            loss_dict.update(evaluator())

            # collect predictions
            prediction_result = out_class.detach().argmax(dim=1)

            # collect selections
            t = t.detach()
            condition = (out_select >= 0.5)
            selection_result = torch.where(condition, torch.ones_like(out_select), torch.zeros_like(out_select)).view(-1)
            # a 1 is a non-rejection

            predictions.append(prediction_result.cpu().numpy())
            selections.append(selection_result.cpu().numpy())

            test_metric_dict.update(loss_dict)

    # post epoch
    print_metric_dict(None, None, test_metric_dict.avg, mode='test')
    accumulative_metrics.update(test_metric_dict.avg)

    # save predictions and selections for the fold
    np.save(f'{FLAGS.weight}/predictions.npy', np.concatenate(predictions))
    np.save(f'{FLAGS.weight}/selections.npy', np.concatenate(selections))

    with open(f"{FLAGS.weight}/test.json", "w") as outfile: 
        json.dump(accumulative_metrics.avg, outfile)


if __name__ == '__main__':
    main()
