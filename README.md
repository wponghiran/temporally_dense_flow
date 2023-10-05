# [ICCV2023] Event-based Temporally Dense Optical Flow Estimation with Sequential Learning

This directory contains codes for preprocessing dataset, training models, and replicating results of the [Event-based Temporally Dense Optical Flow Estimation with Sequential Learning](https://openaccess.thecvf.com/content/ICCV2023/html/Ponghiran_Event-based_Temporally_Dense_Optical_Flow_Estimation_with_Sequential_Learning_ICCV_2023_paper.html) paper. Visualization is also included. Additional results to validate the proposed training method will be later available in this directory.

## Running instruction
Please follow the steps below to prepare the dataset and anaconda environment.
1. Create anaconda environment based on the yml file in the directory `environment.yml`
2. Prepare your dataset for training and testing by downloading all publicly available sequences from the DSEC dataset (see https://dsec.ifi.uzh.ch/dsec-datasets/download/). Note that the DSEC dataset omits the ground truths of the original test sequences from the public and the only way to use them is to submit model predictions to their evaluation system. We cannot use their evaluation system in our work as we want to make sure that all temporally dense flows generated by the models are reliable. As described in a paper, we generate extra intermediate flows between the existing ground truths using linear interpolation. Therefore, the original training set in the DSEC dataset is split into our training and testing set in an 80/20 ratio.
3. Pre-process the dataset in a deterministic manner using the following commands:
```bash
# Generate event counts that will be directly used and pre-processed for optical flow estimation
python preprocess_dsec_supervised.py --n-split 10 --dataset-dir <LOCATION_OF_DATASET> --save-dir <PATH_FOR_SAVING_PRE_PROCESSED_DATASET>

# Generate voxel representations to be used specifically by ERAFT for optical flow estimation
python preprocess_dsec_supervised_voxel.py --n-split 10 --dataset-dir <LOCATION_OF_DATASET> --save-dir <PATH_FOR_SAVING_PRE_PROCESSED_DATASET>
```
4. Train the baseline EV-FlowNet model for optical flow prediction from event representation. You can simply replace the argument to train other existing models like `Spike-FlowNet` and `Adaptive-FlowNet` as follows. Training is expected to take 1hr/epoch on a single NVIDIA A40 GPU. 
```bash
# Train EV-FlowNet for optical flow estimation at 10 Hz
CUDA_VISIBLE_DEVICES=<GPU_IDS_TO_BE_USED> python main_dsec.py --dataset-dir <PATH_TO_PRE_PROCESSED_DATASET> --arch NonSpikingEVFlowNet --n-epochs 10 --bs 16 --mode train  --save-dir <SAVE_PATH> --model-options \'num_res_blocks\':2 --lr 5e-4
# Replace NonSpikingEVFlowNet in --arch argument with SpikeFlowNet or AdaptiveFlowNet to achieve a similar training

# Train ERAFT model for optical flow estimation at 10 Hz
CUDA_VISIBLE_DEVICES=<GPU_IDS_TO_BE_USED> python main_dsec_voxel.py --dataset-dir <PATH_TO_PRE_PROCESSED_DATASET> --n-epochs 10 --bs 16 --mode train --save-dir <SAVE_PATH> --lr 1e-4 
```
5. Test the baseline model and other existing models for optical flow estimation from event representation using mode `test_w_reset`.
```bash
# Test EV-FlowNet for optical flow estimation at 10 Hz
CUDA_VISIBLE_DEVICES=<GPU_IDS_TO_BE_USED> python main_dsec.py --dataset-dir <PATH_TO_PRE_PROCESSED_DATASET> --arch NonSpikingEVFlowNet --n-epochs 10 --bs 16 --mode test_w_reset  --save-dir <SAVE_PATH> --model-options \'num_res_blocks\':2 --lr 5e-4 --model-path <SAVE_PATH>/dt1,tsplit10,NonSpikingEVFlowNet,adam,e10,bs16,lr5e-04,num_res_blocks-2/checkpoint_ep10.pt
# Replace NonSpikingEVFlowNet in --arch argument with SpikeFlowNet or AdaptiveFlowNet to achieve a similar testing

# Test ERAFT model for optical flow estimation at 10 Hz
CUDA_VISIBLE_DEVICES=<GPU_IDS_TO_BE_USED> python main_dsec_voxel.py --dataset-dir <PATH_TO_PRE_PROCESSED_DATASET> --n-epochs 10 --bs 16 --mode test_w_reset --save-dir <SAVE_PATH> --lr 1e-4 --model-path <SAVE_PATH>/dt1,tsplit10,ERAFT,adam,e10,bs16,lr1e-04/checkpoint_ep10.pt 
```
6. Train the proposed models for temporally dense optical flow estimation.
```bash
# Train EfficientSpike-FlowNet for optical flow estimation at 100 Hz
CUDA_VISIBLE_DEVICES=<GPU_IDS_TO_BE_USED> python main_dsec.py --dataset-dir <PATH_TO_PRE_PROCESSED_DATASET> --arch EfficientSpikeEVFlowNet --n-epochs 10 --bs 16 --mode train --save-dir <SAVE_PATH> --lr 5e-3 --n-split 10 --no-grad-ts 1  
# Replace 10 in the argument --n-split to train a model for less or more frequent optical flow estimation like 5 for 50Hz or 15 for 150 Hz

# Train LSTM-FlowNet for optical flow estimation at 100 Hz
CUDA_VISIBLE_DEVICES=<GPU_IDS_TO_BE_USED> python main_dsec.py --dataset-dir <PATH_TO_PRE_PROCESSED_DATASET> --arch LSTMEVFlowNet --n-epochs 10 --bs 8 --mode train --save-dir <SAVE_PATH> --no-grad-ts 1 --model-options \'num_res_blocks\':2 --lr 5e-4
```
7. Test the proposed models without network reset using mode `test_wo_reset`.
```bash
# Test EfficientSpike-FlowNet for optical flow estimation at 100 Hz
CUDA_VISIBLE_DEVICES=<GPU_IDS_TO_BE_USED> python main_dsec.py  --dataset-dir <PATH_TO_PRE_PROCESSED_DATASET> --arch EfficientSpikeEVFlowNet --n-epochs 10 --bs 16 --mode test_w_reset --save-dir <SAVE_PATH> --lr 5e-3 --n-split 10 --no-grad-ts 1 --model-path <SAVE_PATH>/dt1,tsplit10,EfficientSpikeEVFlowNet,adam,e1,bs16,lr5e-03,ng1/checkpoint_ep10.pt  
# Replace 10 in the argument --n-split to train a model for less or more frequent optical flow estimation like 5 for 50Hz or 15 for 150 Hz

# Test LSTM-FlowNet for optical flow estimation at 100 Hz
CUDA_VISIBLE_DEVICES=<GPU_IDS_TO_BE_USED> python main_dsec.py --dataset-dir <PATH_TO_PRE_PROCESSED_DATASET> --arch LSTMEVFlowNet --n-epochs 10 --bs 8 --mode test_wo_reset --save-dir <SAVE_PATH> --no-grad-ts 1 --model-options \'num_res_blocks\':2 --lr 5e-4 --model-path <SAVE_PATH>/dt1,tsplit10,LSTMEVFlowNet,adam,e10,bs8,lr5e-04,num_res_blocks-2,ng1/checkpoint_ep8.pt
```

## Additional results 
Additional results on top of the submitted ones to ICCV 2023 is plan to be included here. Stay tuned!

## Citation
If you use this code in your work, please cite the following [paper](https://openaccess.thecvf.com/content/ICCV2023/html/Ponghiran_Event-based_Temporally_Dense_Optical_Flow_Estimation_with_Sequential_Learning_ICCV_2023_paper.html):
```
@inproceedings{ponghiran2023event,
  title={Event-based Temporally Dense Optical Flow Estimation with Sequential Learning},
  author={Ponghiran, Wachirawit and Liyanagedera, Chamika Mihiranga and Roy, Kaushik},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={9827--9836},
  year={2023}
}
```

## Visualization
We have compiled a video illustrating the temporally dense optical flow estimation with EfficientSpike-FlowNet (see bottom right) along with a video illustrating the flow estimation from the baseline EV-FlowNet model (see bottom left) for a qualitative evaluation purpose.

Due to the way EV-FlowNet (and other existing models) was trained, it predicts optical flow at the same frequency as the original optical flow ground truth (10 Hz on the DSEC dataset) even if the event camera outputs events at a much faster rate. We use our proposed sequential training method to train EfficientSpike-FlowNet and achieve optical flow estimation at 100 Hz. The predictions from EfficientSpike-FlowNet are smoother than the predictions from EV-FlowNet as the predicted frame rate of the proposed method is 10x higher. EfficientSpike-FlowNet doesn't produce the same flow quality as EV-FlowNet (as discussed in the paper). However, it is expected to achieve a fast optical flow prediction with a fraction of energy than the baseline model due to its efficiency in processing events. This will be potentially useful for real-time application which requires a quick reaction time like a flying drone at a fast speed.
Video for demonstrating avaliable at this [link](https://youtube.com/shorts/nPwFKbhsCUI?feature=share). Generated GIF of the same visualization is included in [`./figure`](./figures)

![](./figures/supplementary_video.gif)


