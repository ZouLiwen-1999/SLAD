import os

cmd='CUDA_VISIBLE_DEVICES=4 python amae_train.py --logdir=output/ap1_rec1 --batch_size=1 --smartcache_dataset'

os.system(cmd)
