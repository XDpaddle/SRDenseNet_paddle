from x2paddle import torch2paddle
import argparse
import os
import copy
import paddle
from paddle import nn
import paddle.optimizer as optim
from x2paddle.torch2paddle import DataLoader
from tqdm import tqdm
from models import SRDenseNet
from datasets import TrainDataset
from datasets import EvalDataset
from utils import AverageMeter
from utils import calc_psnr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, required=True)
    parser.add_argument('--eval-file', type=str, required=True)
    parser.add_argument('--outputs-dir', type=str, required=True)
    parser.add_argument('--weights-file', type=str)
    parser.add_argument('--growth-rate', type=int, default=16)
    parser.add_argument('--num-blocks', type=int, default=8)
    parser.add_argument('--num-layers', type=int, default=8)
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--patch-size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-epochs', type=int, default=60)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))
    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    device=paddle.CUDAPlace(0)
    paddle.seed(args.seed)

    model = SRDenseNet(growth_rate=args.growth_rate, num_blocks=args.num_blocks, num_layers=args.num_layers)
    
    if args.weights_file is not None:
        state_dict = model.state_dict()
        for n, p in paddle.load(args.weights_file).items():
            if n in state_dict.keys():
                state_dict[n].copy_(p)
            else:
                raise KeyError(n)
            
    criterion = paddle.nn.MSELoss()            
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=args.lr)
    train_dataset = TrainDataset(args.train_file, 
                                 patch_size=args.patch_size, 
                                 scale=args.scale)
    
    train_dataloader = DataLoader(dataset=train_dataset,
                                   batch_size=args.batch_size, 
                                   shuffle=True, num_workers=args.num_workers,
                                  pin_memory=True
        )
    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0
    for epoch in range(args.num_epochs):
        lr = args.lr * 0.1 ** (epoch // int(args.num_epochs * 0.8))
            
        model.train()
        epoch_losses = AverageMeter()
        with tqdm(total=len(train_dataset) - len(train_dataset) % args.batch_size, ncols=80) as t:
            t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))
            
            for data in train_dataloader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                preds = model(inputs)

                loss = criterion(preds, labels)

                epoch_losses.update(loss.numpy()[0], len(inputs))
                optimizer.clear_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(inputs))

        paddle.save(model.state_dict(), os.path.join(args.outputs_dir,'epoch_{}.pdiparams'.format(epoch)))
        
        model.eval()
        epoch_psnr = AverageMeter()

        for data in eval_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            with paddle.no_grad():
                preds = model(inputs).clip(0.0, 1.0)

            preds = preds[:, :, args.scale:-args.scale, args.scale:-args.scale]
            labels = labels[:, :, args.scale:-args.scale, args.scale:-args.scale]

            epoch_psnr.update(calc_psnr(preds, labels), len(inputs))
        print('eval psnr: {:.2f}'.format(epoch_psnr.avg.numpy()[0]))

        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())

    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr.numpy()[0]))
    paddle.save(best_weights, os.path.join(args.outputs_dir, 'best.pdiparams'))
