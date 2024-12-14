#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import numpy as np
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

# def render_set(model_path, source_path, name, iteration, views, gaussians, pipeline, background, args):
#     render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
#     gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
#     render_npy_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_npy")
#     gts_npy_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_npy")

#     makedirs(render_npy_path, exist_ok=True)
#     makedirs(gts_npy_path, exist_ok=True)
#     makedirs(render_path, exist_ok=True)
#     makedirs(gts_path, exist_ok=True)
#     from network import SemanticPredictor    
#     for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
#         semantic_nn = SemanticPredictor(input_dim=3, camera_dim=12, output_dim=3, hidden_dim=64, num_layers=3, degree=1).cuda()
#         semantic_nn_ckpt = torch.load('/n/netscratch/pfister_lab/Everyone/yingwei/final_proj/2952X-Final-Project/dataset/patio/output/patio/semantic_nn_30000_lvl3.pth')
#         semantic_nn.load_state_dict(semantic_nn_ckpt)
#         semantic_nn.eval()

#         camera_extrinsics = torch.cat((torch.tensor(view.R).flatten(), torch.tensor(view.T).flatten()), dim=0).cuda()
#         pred_language_feature = semantic_nn(gaussians.get_xyz, camera_extrinsics.cuda())

#         output = render(view, gaussians, pipeline, background, args, pred_language_feature=pred_language_feature)

#         if not args.include_feature:
#             rendering = output["render"]
#         else:
#             rendering = output["language_feature_image"]
            
#         if not args.include_feature:
#             gt = view.original_image[0:3, :, :]
            
#         else:
#             gt, mask = view.get_language_feature(os.path.join(source_path, args.language_features_name), feature_level=3)

#         np.save(os.path.join(render_npy_path, '{0:05d}'.format(idx) + ".npy"),rendering.permute(1,2,0).cpu().numpy())
#         np.save(os.path.join(gts_npy_path, '{0:05d}'.format(idx) + ".npy"),gt.permute(1,2,0).cpu().numpy())
#         torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
#         torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

from sklearn.metrics.pairwise import cosine_similarity
def render_set(
    model_path, source_path, name, iteration, views, gaussians, pipeline, background, args
):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_after")
    makedirs(render_path, exist_ok=True)

    # 初始化语义预测器并加载权重
    from network import SemanticPredictor
    semantic_nn = SemanticPredictor(
        input_dim=3, camera_dim=12, output_dim=3, hidden_dim=64, num_layers=3, degree=1
    ).cuda()
    semantic_nn_ckpt = torch.load(
        '/n/netscratch/pfister_lab/Everyone/yingwei/final_proj/2952X-Final-Project/dataset/patio/output/patio/semantic_nn_30000_lvl3.pth'
    )
    semantic_nn.load_state_dict(semantic_nn_ckpt)
    semantic_nn.eval()

    N = gaussians.get_xyz.shape[0]  # 总高斯球数
    mask = torch.ones(N, device='cuda', dtype=torch.float32)  # 初始化 mask，默认全可见

    # 初始化完整的 pred_features 存储张量
    all_features = torch.zeros((N, len(views), 3), device='cuda')  # Shape: (N, num_views, output_dim)


    # 遍历每个视角生成 pred_features
    for view_idx, view in enumerate(tqdm(views[:20], desc="Processing Views")):
        # 当前视角的相机外参
        camera_extrinsics = torch.cat(
            (torch.tensor(view.R).flatten(), torch.tensor(view.T).flatten()), dim=0
        ).cuda()  # Shape: (12,)

        # 计算所有高斯球的 pred_features
        pred_features = semantic_nn(gaussians.get_xyz, camera_extrinsics)  # Shape: (N, output_dim)
        all_features[:, view_idx, :] = pred_features  # 存储当前视角的特征

    sample_size = min(1000, N)  # 每次采样的高斯球数
    for _ in range(100):  # 重复采样次数
        # 随机采样高斯球
        sampled_indices = np.random.choice(N, sample_size, replace=False)

        # 提取采样的高斯球特征
        sampled_features = all_features[sampled_indices, :, :]  # Shape: (sample_size, num_views, output_dim)

        # 比较不同视角下的特征
        for i in range(sample_size):
            features = sampled_features[i].cpu().detach().numpy()  # Shape: (num_views, output_dim)
            if features.shape[0] < 2:  # 如果视角不足两次，则跳过该高斯球
                continue
            # 计算不同视角间的相似性
            similarity_matrix = cosine_similarity(features)  # Shape: (num_views, num_views)
            avg_similarity = np.mean(similarity_matrix[np.triu_indices(len(views), k=1)])  # 取非对角线部分的平均值
            # print("avg_similarity", avg_similarity)
            # 更新 mask
            if avg_similarity < 0.7:  # 阈值可调
                mask[sampled_indices[i]] = 0.0


    # 在 render 中传入 mask
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        semantic_nn = SemanticPredictor(input_dim=3, camera_dim=12, output_dim=3, hidden_dim=64, num_layers=3, degree=1).cuda()
        semantic_nn_ckpt = torch.load('/n/netscratch/pfister_lab/Everyone/yingwei/final_proj/2952X-Final-Project/dataset/patio/output/patio/semantic_nn_30000_lvl3.pth')
        semantic_nn.load_state_dict(semantic_nn_ckpt)
        semantic_nn.eval()

        camera_extrinsics = torch.cat((torch.tensor(view.R).flatten(), torch.tensor(view.T).flatten()), dim=0).cuda()
        pred_language_feature = semantic_nn(gaussians.get_xyz, camera_extrinsics.cuda())

        output = render(view, gaussians, pipeline, background, args, pred_language_feature=pred_language_feature, mask=mask)

        rendering = output["render"] if not args.include_feature else output["render"] 
        # rendering = output["render"] if not args.include_feature else output["language_feature_image"]

       # np.save(os.path.join(render_path, '{0:05d}'.format(idx) + ".npy"),rendering.permute(1,2,0).cpu().numpy())
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))



def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, args):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, shuffle=False)
        checkpoint = os.path.join(args.model_path, 'chkpnt30000.pth')
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, args, mode='test')
        
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, dataset.source_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, args)

        if not skip_test:
             render_set(dataset.model_path, dataset.source_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, args)

if __name__ == "__main__":
    # Set up command line argument parser
    
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--include_feature", action="store_true")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args)
