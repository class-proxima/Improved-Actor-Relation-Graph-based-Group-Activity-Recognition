import sys

from validate_net import validate_net
sys.path.append(".")
from train_net import *

cfg=Config('collective')
cfg.test_seqs=[9]
cfg.device_list="4,5,6,7"
cfg.training_stage=3
cfg.stage1_model_path='../result/STAGE1_MODEL.pth'  #PATH OF THE BASE MODEL
cfg.stage2_model_path='../result/STAGE2_MODEL.pth'  #PATH OF THE BASE MODEL
cfg.train_backbone=False
cfg.test_before_train=True
cfg.image_size=480, 720
cfg.out_size=57,87
cfg.num_boxes=13
cfg.num_actions=6
cfg.num_activities=5
cfg.num_frames=10
cfg.num_graph=4
cfg.tau_sqrt=True

cfg.batch_size=16
cfg.test_batch_size=8
cfg.train_learning_rate=1e-4
cfg.train_dropout_prob=0.2
cfg.weight_decay=1e-2
cfg.lr_plan={}
cfg.max_epoch=50

cfg.exp_note='Collective_stage3'
validate_net(cfg)