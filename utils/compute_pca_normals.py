import os
import numpy as np
from tqdm import tqdm
import argparse
from sklearn.neighbors import NearestNeighbors
from datasets.shapenet import PartNormalDataset

def compute_pca_normals(points, k=20):
    """
    使用PCA方法计算点云法线
    Args:
        points: Nx3 点云坐标
        k: 用于计算局部协方差矩阵的邻居点数量
    Returns:
        normals: Nx3 法线向量
    """
    # 构建KNN
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(points)
    distances, indices = nbrs.kneighbors(points)

    # 移除自身点
    indices = indices[:, 1:]

    normals = np.zeros_like(points)
    for i in range(len(points)):
        # 获取局部点集
        local_points = points[indices[i]]
        # 计算局部协方差矩阵
        cov = np.cov(local_points.T)
        # 计算特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # 最小特征值对应的特征向量即为法线方向
        normals[i] = eigenvectors[:, 0]

    # 确保法线方向一致（指向外部）
    normals = orient_normals(points, normals)
    return normals

def orient_normals(points, normals):
    """
    统一法线方向，使其指向点云外部
    """
    # 计算点云中心
    center = np.mean(points, axis=0)
    # 计算从中心到每个点的向量
    vectors = points - center
    # 计算点积
    dot_products = np.sum(vectors * normals, axis=1)
    # 如果点积为负，则翻转法线方向
    normals[dot_products < 0] *= -1
    return normals

def compute_rmse(pca_normals, gt_normals):
    """
    计算PCA法线与真实法线之间的RMSE
    Args:
        pca_normals: Nx3 PCA计算的法线
        gt_normals: Nx3 真实法线
    Returns:
        rmse: 均方根误差
    """
    # 确保法线方向一致（通过点积判断）
    dot_products = np.sum(pca_normals * gt_normals, axis=1)
    pca_normals[dot_products < 0] *= -1

    # 计算RMSE
    squared_diff = np.sum((pca_normals - gt_normals) ** 2, axis=1)
    rmse = np.sqrt(np.mean(squared_diff))
    return rmse

def process_dataset(dataset_path, output_path, k=20):
    """
    处理整个数据集，计算PCA法线并保存
    Args:
        dataset_path: 原始数据集路径
        output_path: 输出路径
        k: KNN的邻居数量
    """
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)

    # 加载数据集
    dataset = PartNormalDataset(root=dataset_path, npoints=2048, split='trainval', normal_channel=True)

    # 用于存储所有RMSE值
    all_rmses = []

    # 遍历所有点云文件
    for item, file_path in tqdm(dataset.datapath, desc="Processing point clouds"):
        # 创建输出文件路径
        rel_path = os.path.relpath(file_path, dataset_path)
        output_file = os.path.join(output_path, rel_path)

        # 读取原始点云数据
        data = np.loadtxt(file_path).astype(np.float32)
        points = data[:, 0:3]
        gt_normals = data[:, 3:6]  # 真实法线
        seg = data[:, -1].astype(np.int32)

        # 如果输出文件已存在，读取PCA法线
        if os.path.exists(output_file):
            # print(f"Reading existing file: {output_file}")
            output_data = np.loadtxt(output_file).astype(np.float32)
            pca_normals = output_data[:, 3:6]
        else:
            # 计算PCA法线
            pca_normals = compute_pca_normals(points, k=k)

            # 组合数据：xyz + pca_normals + segmentation
            combined_data = np.hstack([points, pca_normals, seg.reshape(-1, 1)])

            # 创建输出目录
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            # 保存数据
            np.savetxt(output_file, combined_data, fmt='%.6f')

        # 计算RMSE
        rmse = compute_rmse(pca_normals, gt_normals)
        all_rmses.append(rmse)
        # print(f"RMSE for {rel_path}: {rmse:.6f}")

    # 计算并打印整体RMSE统计信息
    if all_rmses:
        mean_rmse = np.mean(all_rmses)
        std_rmse = np.std(all_rmses)
        print(f"\nRMSE Statistics:")
        print(f"Mean RMSE: {mean_rmse:.6f}")
        print(f"Std RMSE: {std_rmse:.6f}")
        print(f"Min RMSE: {min(all_rmses):.6f}")
        print(f"Max RMSE: {max(all_rmses):.6f}")

        # 保存RMSE统计信息到文件
        stats_file = os.path.join(output_path, 'rmse_stats.txt')
        with open(stats_file, 'w') as f:
            f.write(f"Mean RMSE: {mean_rmse:.6f}\n")
            f.write(f"Std RMSE: {std_rmse:.6f}\n")
            f.write(f"Min RMSE: {min(all_rmses):.6f}\n")
            f.write(f"Max RMSE: {max(all_rmses):.6f}\n")
            f.write(f"Total processed files: {len(all_rmses)}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True,
                      help='Path to the original ShapeNet dataset')
    parser.add_argument('--output_path', type=str, required=True,
                      help='Path to save the processed dataset with PCA normals')
    parser.add_argument('--k', type=int, default=20,
                      help='Number of neighbors for PCA normal computation')
    args = parser.parse_args()

    process_dataset(args.dataset_path, args.output_path, args.k)

if __name__ == '__main__':
    main()