import os

cmd='CUDA_VISIBLE_DEVICES=0 python amae_train.py --logdir=output/ap1_rec1d5 --batch_size=1 --smartcache_dataset --resume=output/ap1_rec1d5/model_current_epoch.pt'

os.system(cmd)
