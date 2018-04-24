import os

import numpy as np
from PIL import Image

import chainer
import chainer.cuda
from chainer import Variable
from chainer.dataset import concat_examples

def out_image(updater, enc, dec, rows, cols, seed, dst, tensorboard):
    @chainer.training.make_extension()
    def make_image(trainer):
        np.random.seed(seed)
        n_images = rows * cols
        xp = enc.xp
        
        w_in = 64
        w_out = 64
        in_ch = 1
        out_ch = 3
        
        in_all = np.zeros((n_images, in_ch, w_in, w_in)).astype("f")
        gt_all = np.zeros((n_images, out_ch, w_out, w_out)).astype("f")
        gen_all = np.zeros((n_images, out_ch, w_out, w_out)).astype("f")
        
        for it in range(n_images):
            x_in, t_out = concat_examples(updater.get_iterator('test').next())

            x_in = xp.asarray(x_in)
            t_out = xp.asarray(t_out)
            batchsize = len(x_in)
            
            z_c = np.asarray(enc.make_hidden(batchsize, 1))
            z_c = np.tile(z_c, (1, enc.dim_z))
            x_in_with_noise = Variable(enc.concat_noise(x_in, z_c))
            x_in = Variable(x_in)

            bottleneck = enc(x_in_with_noise)
            x_out = dec(bottleneck)
            
            if updater.device == -1:
                in_all[it,:] = x_in.data[0,:]
                gt_all[it,:] = t_out[0,:]
                gen_all[it,:] = x_out.data[0,:]
            else:
                in_all[it,:] = x_in.data.get()[0,:]
                gt_all[it,:] = t_out.get()[0,:]
                gen_all[it,:] = x_out.data.get()[0,:]
        
        def save_image(x, name, mode=None):
            _, C, H, W = x.shape
            x = x.reshape((rows, cols, C, H, W))
            x = x.transpose(0, 3, 1, 4, 2)
            if C==1:
                x = x.reshape((rows*H, cols*W))
            else:
                x = x.reshape((rows*H, cols*W, C))

            preview_dir = '{}/preview'.format(dst)
            preview_path = preview_dir +\
                '/image_{}_{:0>8}.png'.format(name, trainer.updater.iteration)
            if not os.path.exists(preview_dir):
                os.makedirs(preview_dir)
            Image.fromarray(x, mode=mode).convert('RGB').save(preview_path)

            # from here, conert x for logging to tensorboard
            if x.ndim == 3:
                x = x.transpose(2, 0, 1)
            else:
                x = np.tile(np.expand_dims(x, 0), (3,1,1))
            x = x / 255.
            tensorboard.add_image(name, x, updater.epoch)
        
        x = np.asarray(np.clip(gen_all * 128 + 128, 0.0, 255.0), dtype=np.uint8)
        save_image(x, "gen")
        
        x = np.asarray(np.clip(in_all * 128 + 128, 0.0, 255.0), dtype=np.uint8)
        save_image(x, "in")
        
        x = np.asarray(np.clip(gt_all * 128 + 128, 0.0, 255.0), dtype=np.uint8)
        save_image(x, "gt")
        
    return make_image
