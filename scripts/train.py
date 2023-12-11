import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import click
from collections import OrderedDict

import torch
import torchvision

from external.dada.flag_holder import FlagHolder
from external.dada.metric import MetricDict
from external.dada.io import print_metric_dict
from external.dada.io import save_model
from external.dada.logger import Logger

from selectivenet.heareval import HearEvalNN
from selectivenet.model import SelectiveNet
from selectivenet.loss import SelectiveLoss, OneHotToCrossEntropyLoss
from selectivenet.data import EmbeddingsDataset
from selectivenet.evaluator import Evaluator

# options
@click.command()
# model
@click.option('--dim_features', type=int, default=1024)
@click.option('--dropout_prob', type=float, default=0.1)
# data
@click.option('-d', '--dataset', type=str, required=False)
@click.option('--dataroot', type=str, default='../data', help='path to dataset root')
@click.option('-j', '--num_workers', type=int, default=8)
@click.option('-N', '--batch_size', type=int, default=128)
@click.option('--normalize', is_flag=True, default=True)
# optimization
@click.option('--num_epochs', type=int, default=100)
@click.option('--lr', type=float, default=0.00032, help='learning rate')
@click.option('--wd', type=float, default=5e-4, help='weight decay')
@click.option('--momentum', type=float, default=0.9)
# loss
@click.option('--coverage', type=float, required=True)
@click.option('--alpha', type=float, default=0.5, help='balancing parameter between selective_loss and ce_loss')
# logging
@click.option('-s', '--suffix', type=str, default='')
@click.option('-l', '--log_dir', type=str, required=True)

def main(**kwargs):
    train(**kwargs)

def train(**kwargs):
    FLAGS = FlagHolder()
    FLAGS.initialize(**kwargs)
    FLAGS.summary()
    FLAGS.dump(path=os.path.join(FLAGS.log_dir, 'flags{}.json'.format(FLAGS.suffix)))

    # dataset
    # dataset will be the name of the folder and dataroot just the embeds folder
    train = EmbeddingsDataset(FLAGS.dataroot, fold_name='train')
    valid = EmbeddingsDataset(FLAGS.dataroot, fold_name='valid')

    print(next(iter(train))[0].shape)

    num_classes = 12
    hidden_dim = 1024

    # model
    #features = vgg16_variant(dataset_builder.input_size, FLAGS.dropout_prob).cuda()
    features = HearEvalNN(input_dim=FLAGS.dim_features)
    model = SelectiveNet(features, hidden_dim, num_classes).cuda()
    if torch.cuda.device_count() > 1: model = torch.nn.DataParallel(model)

    # optimizer
    params = model.parameters() 
    optimizer = torch.optim.Adam(params, lr=FLAGS.lr)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
    # heareval doesn't use a scheduler

    # loss
    base_loss = torch.nn.CrossEntropyLoss()#(reduction='none')
    SelectiveCELoss = SelectiveLoss(base_loss, coverage=FLAGS.coverage)
    MyLoss = OneHotToCrossEntropyLoss()

    # logger
    train_logger = Logger(path=os.path.join(FLAGS.log_dir,'train_log{}.csv'.format(FLAGS.suffix)), mode='train')
    val_logger   = Logger(path=os.path.join(FLAGS.log_dir,'val_log{}.csv'.format(FLAGS.suffix)), mode='val')

    train_loader = torch.utils.data.DataLoader(train, batch_size=FLAGS.batch_size, shuffle=True, num_workers=os.cpu_count())
    val_loader = torch.utils.data.DataLoader(valid, batch_size=FLAGS.batch_size, shuffle=True, num_workers=os.cpu_count())

    for ep in range(FLAGS.num_epochs):
        # pre epoch
        train_metric_dict = MetricDict()
        val_metric_dict = MetricDict()

        # train
        for i, (x,t) in enumerate(train_loader):
            model.train()
            x = x.to('cuda', non_blocking=True)
            t = t.to('cuda', non_blocking=True)
            # forward
            #out_class, out_select, out_aux = model(x)
            out_aux = model.forward_logit(x)
            

            # compute selective loss
            loss_dict = OrderedDict()
            # loss dict includes, 'empirical_risk' / 'emprical_coverage' / 'penulty'
            #selective_loss, loss_dict = SelectiveCELoss(out_class, out_select, t)
            #selective_loss *= FLAGS.alpha
            #loss_dict['selective_loss'] = selective_loss.detach().cpu().item()
            # compute standard cross entropy loss
            ce_loss = base_loss(out_aux, t)
            #ce_loss *= (1.0 - FLAGS.alpha)
            loss_dict['ce_loss'] = ce_loss.detach().cpu().item()
            
            # total loss
            loss = ce_loss #selective_loss + ce_loss
            loss_dict['loss'] = loss.detach().cpu().item()

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_metric_dict.update(loss_dict)
        
        # validation
        with torch.autograd.no_grad():
            for i, (x,t) in enumerate(val_loader):
                model.eval()
                x = x.to('cuda', non_blocking=True)
                t = t.to('cuda', non_blocking=True)

                # forward
                #out_class, out_select, out_aux = model(x)
                out_aux = model.forward_logit(x)
                #pred = model.activation(out_aux, dim=1)
                # compute selective loss
                loss_dict = OrderedDict()
                # loss dict includes, 'empirical_risk' / 'emprical_coverage' / 'penulty'
                #selective_loss, loss_dict = SelectiveCELoss(out_class, out_select, t)
                #selective_loss *= FLAGS.alpha
                #loss_dict['selective_loss'] = selective_loss.detach().cpu().item()
                # compute standard cross entropy loss
                #ce_loss = torch.nn.CrossEntropyLoss()(out_aux, t)
                #ce_loss *= (1.0 - FLAGS.alpha)
                ce_loss = base_loss(out_aux, t)
                loss_dict['ce_loss'] = ce_loss.detach().cpu().item()
                
                # total loss
                loss = ce_loss #selective_loss + ce_loss
                loss_dict['loss'] = loss.detach().cpu().item()

                # evaluation
                #evaluator = Evaluator(out_class.detach(), t.detach(), out_select.detach())
                #loss_dict.update(evaluator())

                val_metric_dict.update(loss_dict)

        # post epoch
        # print_metric_dict(ep, FLAGS.num_epochs, train_metric_dict.avg, mode='train')
        print_metric_dict(ep, FLAGS.num_epochs, val_metric_dict.avg, mode='val')

        train_logger.log(train_metric_dict.avg, step=(ep+1))
        val_logger.log(val_metric_dict.avg, step=(ep+1))

        #scheduler.step()

    # post training
    save_model(model, path=os.path.join(FLAGS.log_dir, 'weight_final{}.pth'.format(FLAGS.suffix)))


if __name__ == '__main__':
    main()
