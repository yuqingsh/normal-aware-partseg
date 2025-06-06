import os
import sys
import numpy as np
import torch

# 导入数据集类
from datasets.pnpnet import PointcloudPatchDataset
from datasets.shapenet_adafit import ShapeNet

# 简单测试函数 - 只加载数据集并获取一个样本
def test_pnpnet_load():
    project_root = "/share/home/u01015/ShenYuqing/normal-aware-partseg"
    print("=== 简单数据集加载测试 ===")
    
    # PCPNet数据集路径
    pnpnet_path = os.path.join(project_root, 'data/pnpnet/pnpnet')
    # ShapeNet数据集路径
    shapenet_path = os.path.join(project_root, 'data/shapenet/shapenetcore_partanno_segmentation_benchmark_v0_normal')
    
    print(f"PnpNet路径: {pnpnet_path}")
    print(f"ShapeNet路径: {shapenet_path}")
    
    # 检查数据集路径是否存在
    if not os.path.exists(pnpnet_path):
        print(f"警告: PnpNet数据路径不存在: {pnpnet_path}")
    
    if not os.path.exists(shapenet_path):
        print(f"警告: ShapeNet数据路径不存在: {shapenet_path}")
        return

    try:
        # Minimal parameters for instantiation, similar to typical AdaFit/PCPNet usage
        pnpnet_dataset = PointcloudPatchDataset(
            root=pnpnet_path,
            shape_list_filename='trainingset_no_noise.txt', # filename relative to root
            patch_radius=[0.05],          # Example value
            points_per_patch=500,       # Example value
            patch_features=('normal',), # Assuming normals are used
            use_pca=True,               # Common for PCPNet
            center='point',
            point_tuple=1,
            cache_capacity=1
        )
        print(f"PnpNetDataset instantiated.")
        print(f"  Number of shapes found: {len(pnpnet_dataset.shape_names)}")
        print(f"  Total patches: {len(pnpnet_dataset)}")
        
        if len(pnpnet_dataset) > 0:
            sample_idx = 0
            print(f"Fetching sample at index {sample_idx}...")
            sample_pnpnet = pnpnet_dataset[sample_idx]
            print(f"Sample from PnpNetDataset (index {sample_idx}):")
            print(f"  Type: {type(sample_pnpnet)}")
            print(f"  Length (number of elements in tuple): {len(sample_pnpnet)}")
            if len(sample_pnpnet) > 0 and hasattr(sample_pnpnet[0], 'shape'):
                print(f"  Points shape: {sample_pnpnet[0].shape}, dtype: {sample_pnpnet[0].dtype}")
            if len(sample_pnpnet) > 1 and hasattr(sample_pnpnet[1], 'shape'):
                print(f"  Features (e.g., normals) shape: {sample_pnpnet[1].shape}, dtype: {sample_pnpnet[1].dtype}")
            # print(f"  Full sample data (first element, points): {sample_pnpnet[0]}") # Uncomment to see data
        else:
            print("PnpNetDataset is empty or failed to load shapes. Cannot fetch a sample.")
            print(f"Please check data at {pnpnet_path}")

    except Exception as e:
        print(f"Error during PnpNetDataset test: {e}")
        import traceback
        traceback.print_exc()

def test_shapenet_adafit_load():    
    print("=== 简单数据集加载测试 ===")
    project_root = "/share/home/u01015/ShenYuqing/normal-aware-partseg"
    shapenet_data_root_relative = 'data/shapenet/shapenetcore_partanno_segmentation_benchmark_v0_normal'
    shapenet_data_root_abs = os.path.join(project_root, shapenet_data_root_relative)
    
    print(f"ShapeNet data root: {shapenet_data_root_abs}")

    try:
        # Parameters for instantiation based on your shapenet_adafit.py
        shapenet_dataset = ShapeNet(
            root=shapenet_data_root_abs,
            split='train',              # 'train', 'val', or 'test'
            class_choice=None,          # Load all classes, or e.g. ['Airplane']
            patch_radius=[0.05],          # Example value
            points_per_patch=500,       # Example value
            patch_features=('normal',), # Assuming normals are used
            use_pca=True,               # Consistent with AdaFit's needs
            center='point',
            point_tuple=1,
            cache_capacity=1,
            normal_channel_shapenet=True # True if .txt files include normals
        )
        print(f"ShapeNetAdaFitDataset instantiated.")
        print(f"  Number of shapes found: {len(shapenet_dataset.shape_names)}")
        print(f"  Total patches: {len(shapenet_dataset)}")

        if len(shapenet_dataset) > 0:
            sample_idx = 0
            print(f"Fetching sample at index {sample_idx}...")
            sample_shapenet = shapenet_dataset[sample_idx]
            print(f"Sample from ShapeNetAdaFitDataset (index {sample_idx}):")
            print(f"  Type: {type(sample_shapenet)}")
            print(f"  Length (number of elements in tuple): {len(sample_shapenet)}")
            if len(sample_shapenet) > 0 and hasattr(sample_shapenet[0], 'shape'):
                print(f"  Points shape: {sample_shapenet[0].shape}, dtype: {sample_shapenet[0].dtype}")
            if len(sample_shapenet) > 1 and hasattr(sample_shapenet[1], 'shape'):
                print(f"  Features (e.g., normals) shape: {sample_shapenet[1].shape}, dtype: {sample_shapenet[1].dtype}")
            # print(f"  Full sample data (first element, points): {sample_shapenet[0]}") # Uncomment to see data
        else:
            print("ShapeNetAdaFitDataset is empty or failed to load shapes. Cannot fetch a sample.")
            print(f"Please check data at {shapenet_data_root_abs}, the 'train' split, and category subdirectories/split files.")

    except Exception as e:
        print(f"Error during ShapeNetAdaFitDataset test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    # test_pnpnet_load()
    test_shapenet_adafit_load()