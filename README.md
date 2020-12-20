
# Video Understanding based on Human Action and Group Activity Recognition

Source code for the following paper([arXiv link](https://arxiv.org/abs/2010.12968)):

        Video Understanding based on Human Action and Group Activity Recognition
        Zijian Kuang, Xinran Tie



## Dependencies

- Python `3.x`
- install requirements using `pip install -r requirements.txt`
- [RoIAlign for Pytorch](https://github.com/longcw/RoIAlign.pytorch) (Remeber, to rename folder "roi_align" to something else, such as "roi_align_1")
- Datasets: [Collective](http://vhosts.eecs.umich.edu/vision//activity-dataset.html) and [Augmented dataset](http://vhosts.eecs.umich.edu/vision//activity-dataset.html)



## Prepare Datasets

1. Download both [collective](http://vhosts.eecs.umich.edu/vision//ActivityDataset.zip) and [augmented dataset](http://vhosts.eecs.umich.edu/vision//ActivityDataset2.tar.gz) dataset file.
2. Unzip the dataset file into `data/collective`.
3. The folder structure should looks like: 

   ![1](https://github.com/kuangzijian/Group-Activity-Recognition/blob/master/read_me_pictures/folder_structure.png)



## Get Started

1. Stage1: Fine-tune the model on single frame without using GCN.

    ```shell    
    # collective dataset
    python scripts/train_collective_stage1.py
    ```

2. Stage2: Fix weights of the feature extraction part of network, and train the network with GCN.

    ```shell
    # collective dataset
    python scripts/train_collective_stage2.py
    ```

3. Test: Test the result using test video clips.
    ```shell
    # collective dataset
    python scripts/test_collective.py
    ```
    
4. You can specify the running arguments in the python files under `scripts/` directory. The meanings of arguments can be found in `config.py`
   
   Based on our expirements, we suggest to use either NCC or SAD to calculate the pair-wise acotrs' appearance similarty, by setting the self.appearance_calc = "NCC" or "SAD":
      
   ![2](https://github.com/kuangzijian/Group-Activity-Recognition/blob/master/read_me_pictures/appearance_calc.png)

   To speed up the training/testing speed, we would also suggest to set the self.backbone='mobilenet':
   
   ![3](https://github.com/kuangzijian/Group-Activity-Recognition/blob/master/read_me_pictures/back_bone.png)
   
## Experiment results
   
1. Our expirements proved that using MobileNet as backbone in feature extraction will improve the training speed by 35%.

2. Our expirements proved that using NCC or SAD to calculate the pair-wise acotrs' appearance similarty and draw the actor relation graph can improve the group activity prediction accuracy.

   ![3](https://github.com/kuangzijian/Group-Activity-Recognition/blob/master/read_me_pictures/experiments.png)
   
## Citation

```
@inproceedings{CVPR2019_ARG,
  title = {Learning Actor Relation Graphs for Group Activity Recognition},
  author = {Jianchao Wu and Limin Wang and Li Wang and Jie Guo and Gangshan Wu},
  booktitle = {CVPR},
  year = {2019},
}
```



