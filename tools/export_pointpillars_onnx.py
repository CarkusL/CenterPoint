import argparse
import copy
import json
import os
import sys

try:
    import apex
except:
    print("No APEX!")
import numpy as np
import torch
from torch import nn
import yaml
from det3d import torchie
from det3d.datasets import build_dataloader, build_dataset
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.apis import (
    batch_processor,
    build_optimizer,
    get_root_logger,
    init_dist,
    set_random_seed,
    train_detector,
    example_to_device,
)
from det3d.torchie.trainer import load_checkpoint
import pickle 
import time 
from matplotlib import pyplot as plt 
from det3d.torchie.parallel import collate, collate_kitti
from torch.utils.data import DataLoader
import matplotlib.cm as cm
import subprocess
import cv2
from tools.demo_utils import visual 
from collections import defaultdict


class PointPillars(nn.Module):
    def __init__(self,model):
        super(PointPillars, self).__init__()
        self.model = model
    
    def forward(self, x):
        x = self.model.neck(x)
        preds = self.model.bbox_head(x)
        
        return preds



def main():
    cfg = Config.fromfile('configs/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_10sweep_demo_export_onnx.py')
    
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)

    dataset = build_dataset(cfg.data.val)

    data_loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=None,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_kitti,
        pin_memory=False,
    )

    checkpoint = load_checkpoint(model, './latest.pth', map_location="cpu")
    model.eval()
    
    model = model.cuda()

    gpu_device = torch.device("cuda")

    points_list = [] 
    gt_annos = [] 
    detections  = [] 
    
    data_iter = iter(data_loader)
    data_batch = next(data_iter)
    
    pp_model = PointPillars(model)

    points = data_batch['points'][:, 1:4].cpu().numpy()
    with torch.no_grad():

        example = example_to_device(data_batch, gpu_device, non_blocking=False)
        example["voxels"] = torch.zeros((example["voxels"].shape[0],example["voxels"].shape[1],10),dtype=torch.float32,device=gpu_device)
        
        
        example.pop("metadata")
        example.pop("points")

        example["shape"] = torch.tensor(example["shape"], dtype=torch.int32, device=gpu_device)
        model(example)
        torch.onnx.export(model.reader, (example["voxels"],example["num_voxels"],example["coordinates"]),"onnx_model/pfe.onnx",opset_version=11)
        
        rpn_input  = torch.zeros((1,64,512,512),dtype=torch.float32,device=gpu_device)
        torch.onnx.export(pp_model, rpn_input,"onnx_model/rpn.onnx",opset_version=11)



if __name__ == "__main__":
    main()
