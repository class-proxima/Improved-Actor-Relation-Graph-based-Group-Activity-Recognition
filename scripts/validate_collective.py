import sys
sys.path.append(".")
from train_net import *
from validate_net import *

cfg=Config('collective')
cfg.test_seqs=[9,10]

cfg.stage1_model_path='result/[Collective_stage1_stage1]<2020-10-13_00-06-18>/stage1_epoch90_87.82%.pth'  #PATH OF THE BASE MODEL
cfg.stage2_model_path='result/[Collective_stage2_stage2]<2020-10-13_09-36-37>/stage2_epoch32_86.80%.pth'   #PATH OF THE BASE MODEL
cfg.device_list="1, 2"
cfg.training_stage=2

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
cfg.test_batch_size=1
cfg.train_learning_rate=1e-4
cfg.train_dropout_prob=0.2
cfg.weight_decay=1e-2
cfg.lr_plan={}
cfg.max_epoch=50

cfg.exp_note='Collective_validate'
validate_net(cfg)