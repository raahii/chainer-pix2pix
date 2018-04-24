import chainer
import chainer.functions as F
from chainer import Variable

from chainer import cuda
from chainer import function
from chainer.dataset import concat_examples

class FacadeUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.enc, self.dec, self.dis = kwargs.pop('models')
        self.tensorboard = kwargs.pop('tensorboard')
        self.lam1 = kwargs.pop('lam1')
        self.lam2 = kwargs.pop('lam2')
        super(FacadeUpdater, self).__init__(*args, **kwargs)

    def loss_enc(self, enc, x_out, t_out, y_out):
        batchsize,_,w,h = y_out.data.shape
        loss_rec = self.lam1*(F.mean_absolute_error(x_out, t_out))
        loss_adv = self.lam2*F.sum(F.softplus(-y_out)) / batchsize / w / h
        loss = loss_rec + loss_adv
        chainer.report({'loss': loss}, enc)
        if self.is_new_epoch:
            self.tensorboard.add_scalar('loss:encoder', loss.data, self.epoch)
        return loss
        
    def loss_dec(self, dec, x_out, t_out, y_out):
        batchsize,_,w,h = y_out.data.shape
        loss_rec = self.lam1*(F.mean_absolute_error(x_out, t_out))
        loss_adv = self.lam2*F.sum(F.softplus(-y_out)) / batchsize / w / h
        loss = loss_rec + loss_adv
        chainer.report({'loss': loss}, dec)
        if self.is_new_epoch:
            self.tensorboard.add_scalar('loss:decoder', loss.data, self.epoch)
        return loss
        
    def loss_dis(self, dis, y_in, y_out):
        batchsize,_,w,h = y_in.data.shape
        
        L1 = F.sum(F.softplus(-y_in)) / batchsize / w / h
        L2 = F.sum(F.softplus(y_out)) / batchsize / w / h
        loss = L1 + L2
        chainer.report({'loss': loss}, dis)
        if self.is_new_epoch:
            self.tensorboard.add_scalar('loss:discriminator', loss.data, self.epoch)
        return loss

    def update_core(self):
        enc_optimizer = self.get_optimizer('enc')
        dec_optimizer = self.get_optimizer('dec')
        dis_optimizer = self.get_optimizer('dis')
        
        enc, dec, dis = self.enc, self.dec, self.dis
        xp = enc.xp

        x_in, t_out = concat_examples(self.get_iterator('main').next())
        x_in = xp.asarray(x_in)
        t_out = xp.asarray(t_out)
        batchsize = len(x_in)

        x_in_with_noise = Variable(enc.concat_noise(x_in, xp=xp))
        bottleneck = enc(x_in_with_noise)
        x_out = dec(bottleneck)

        x_in = Variable(x_in)
        y_fake = dis(x_in, x_out)
        y_real = dis(x_in, t_out)

        enc_optimizer.update(self.loss_enc, enc, x_out, t_out, y_fake)
        for bottleneck_ in bottleneck:
            bottleneck_.unchain_backward()
        dec_optimizer.update(self.loss_dec, dec, x_out, t_out, y_fake)
        x_in.unchain_backward()
        x_out.unchain_backward()
        dis_optimizer.update(self.loss_dis, dis, y_real, y_fake)
