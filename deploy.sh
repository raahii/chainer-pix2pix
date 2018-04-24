#!/bin/bash
host=${1:-labo}
rsync -ua ~/study/chainer-pix2pix/ $host:~/study/chainer-pix2pix/
