import os

cmd='CUDA_VISIBLE_DEVICES=3 python amae_train.py --logdir=output/ap0_rec1 --batch_size=1 --smartcache_dataset --resume=output/ap0_rec1/model_current_epoch.pt'

os.system(cmd)
