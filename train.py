#!/usr/bin/env python

# python train_facade.py -g 0 -i ./facade/base --out result_facade --snapshot_interval 10000

from __future__ import print_function
import argparse
import os
from pathlib import Path

import chainer
from chainer import training
from chainer.training import extensions
from chainer import serializers

from net import Discriminator
from net import Encoder
from net import Decoder
from updater import FacadeUpdater

from dataset import FacadeDataset, MugFaceDataset
from visualizer import out_image
from tb_chainer import utils, SummaryWriter
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='chainer implementation of pix2pix')
    parser.add_argument('--batchsize', '-b', type=int, default=1,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=200,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--dataset', '-i', default='data/mug',
                        help='Directory of image files.')
    parser.add_argument('--lam1', type=int, default=100,
                        help='lambda1')
    parser.add_argument('--lam2', type=int, default=1,
                        help='lambda2')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--snapshot_interval', type=int, default=5,
                        help='Interval of snapshot (epoch)')
    parser.add_argument('--generate_interval', type=int, default=1,
                        help='Interval of generate samples (epoch)')
    parser.add_argument('--display_interval', type=int, default=1000,
                        help='Interval of displaying log to console (iter)')
    args = parser.parse_args()

    # Set up a neural network to train
    in_ch = 1
    out_ch = 3
    enc = Encoder(in_ch=in_ch)
    dec = Decoder(out_ch=out_ch)
    dis = Discriminator(in_ch=1, out_ch=3)
    
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        enc.to_gpu()  # Copy the model to the GPU
        dec.to_gpu()
        dis.to_gpu()

    # Setup an optimizer
    def make_optimizer(model, alpha=0.0002, beta1=0.5):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(0.00001), 'hook_dec')
        return optimizer

    opt_enc = make_optimizer(enc)
    opt_dec = make_optimizer(dec)
    opt_dis = make_optimizer(dis)

    train_d = MugFaceDataset(Path(args.dataset) / "train")
    test_d  = MugFaceDataset(Path(args.dataset) / "test")
    train_iter = chainer.iterators.SerialIterator(train_d, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test_d, args.batchsize)

    # tensorboard writer
    rpath = Path(args.out)
    tensorboard_path = rpath.parents[0] / 'runs' / rpath.name
    writer = SummaryWriter(str(tensorboard_path))

    # Set up a trainer
    updater = FacadeUpdater(
        models=(enc, dec, dis),
        iterator={
            'main': train_iter,
            'test': test_iter},
        optimizer={
            'enc': opt_enc, 'dec': opt_dec, 
            'dis': opt_dis},
        device=args.gpu,
        tensorboard=writer,
        lam1=args.lam1,
        lam2=args.lam2,
        )
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    snapshot_interval = (args.snapshot_interval, 'epoch')
    generate_interval = (args.generate_interval, 'epoch')
    display_interval = (args.display_interval, 'iteration')
    trainer.extend(extensions.snapshot(
        filename='snapshot_iter_{.updater.iteration}.npz'),
                   trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        enc, 'enc_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        dec, 'dec_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        dis, 'dis_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'enc/loss', 'dec/loss', 'dis/loss',
    ]), trigger=display_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(
        out_image(
            updater, enc, dec,
            5, 5, args.seed, args.out, writer),
            trigger=generate_interval)

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)
    
    # Print params
    params = {
        'host': os.uname()[1],
        'gpu': args.gpu,
        # 'task': task,
        'epoch' : args.epoch,
        'save_dir': args.out,
        'batchsize': args.batchsize,
        'input_channels': in_ch,
        'output_channels': out_ch,
        'lamda1': args.lam1,
        'lamda2': args.lam2,
        'patch_shape_discriminator':  (1,4,4),
        'generate_interval': "{} epochs".format(args.generate_interval),
        'snapshot_interval': "{} iters".format(args.snapshot_interval),
        'display_interval': "{} iters".format(args.display_interval),
        'tensorboard': str(tensorboard_path),
    }
    
    print('')
    for key, val in params.items():
        print('# {}: {}'.format(key, val))
    print('')

    # Run the training
    trainer.run()
    
    # Save the trained model
    chainer.serializers.save_npz(str(Path(args.out)/'enc_fianl.npz'), enc)
    chainer.serializers.save_npz(str(Path(args.out)/'dec_fianl.npz'), dec)
    chainer.serializers.save_npz(str(Path(args.out)/'dis_fianl.npz'), dis)



if __name__ == '__main__':
    main()
