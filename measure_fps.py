import time
import torch
import numpy as np
import argparse

from utils import dist_util
from utils.fixseed import fixseed
from data_loaders.dataloader_amass import DataloaderAMASS
from data_loaders.motion_representation import (
    REPR_LIST, REPR_DIM_DICT, recover_from_repr_smpl, get_repr_smplx, 
    rot6d_to_rotmat, rotation_matrix_to_angle_axis
)
import smplx
from model.posenet import PoseNet
from model.trajnet import TrajNet
from diffusion import gaussian_diffusion_posenet, gaussian_diffusion_trajnet
from diffusion.respace import SpacedDiffusionPoseNet, SpacedDiffusionTrajNet
from utils.model_util import create_gaussian_diffusion

def create_models_and_diffusion(args, test_pose_dataset, test_traj_dataset):
    print("Creating models and diffusion...")
    
    # 1. PoseNet
    model_posenet = PoseNet(dataset=test_pose_dataset, body_feat_dim=test_pose_dataset.body_feat_dim,
                            latent_dim=512, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1, activation="gelu",
                            body_model_path=args.body_model_path,
                            device=dist_util.dev(),
                            traj_feat_dim=test_pose_dataset.traj_feat_dim,
                            ).to(dist_util.dev())

    print('[INFO] loaded PoseNet checkpoint path:', args.model_path_posenet)
    weights = torch.load(args.model_path_posenet, map_location=lambda storage, loc: storage)
    model_posenet.load_state_dict(weights)
    model_posenet.eval()

    diffusion_posenet_eval = create_gaussian_diffusion(args, gd=gaussian_diffusion_posenet,
                                                       return_class=SpacedDiffusionPoseNet,
                                                       num_diffusion_timesteps=args.diffusion_steps_posenet,
                                                       timestep_respacing=args.timestep_respacing_eval,
                                                       device=dist_util.dev())

    # 2. TrajNet
    model_trajnet = TrajNet(time_dim=32, mid_dim=512,
                    cond_dim=test_traj_dataset.traj_feat_dim,
                    traj_feat_dim=test_traj_dataset.traj_feat_dim,
                    trajcontrol=False,
                    device=dist_util.dev(),
                    dataset=test_traj_dataset,
                    repr_abs_only=args.repr_abs_only,
                    ).to(dist_util.dev())

    print('[INFO] loaded TrajNet checkpoint path:', args.model_path_trajnet)
    weights = torch.load(args.model_path_trajnet, map_location=lambda storage, loc: storage)
    model_trajnet.load_state_dict(weights)
    model_trajnet.eval()

    # 3. TrajNet Control (if iterative refinement is used)
    model_trajnet_control = None
    if args.sample_iter > 1:
        model_trajnet_control = TrajNet(time_dim=32, mid_dim=512,
                                cond_dim=test_traj_dataset.traj_feat_dim,
                                traj_feat_dim=test_traj_dataset.traj_feat_dim,
                                trajcontrol=True,
                                device=dist_util.dev(),
                                dataset=test_traj_dataset,
                                repr_abs_only=args.repr_abs_only,
                                ).to(dist_util.dev())

        print('[INFO] loaded TrajNet TrajControl checkpoint path:', args.model_path_trajnet_control)
        weights = torch.load(args.model_path_trajnet_control, map_location=lambda storage, loc: storage)
        model_trajnet_control.load_state_dict(weights)
        model_trajnet_control.eval()

    diffusion_trajnet_eval = create_gaussian_diffusion(args, gd=gaussian_diffusion_trajnet,
                                                       return_class=SpacedDiffusionTrajNet,
                                                       num_diffusion_timesteps=args.diffusion_steps_trajnet,
                                                       timestep_respacing=args.timestep_respacing_eval,
                                                       device=dist_util.dev())

    return model_posenet, model_trajnet, model_trajnet_control, diffusion_posenet_eval, diffusion_trajnet_eval

def run_inference_on_batch(test_batch_pose, test_batch_traj, models, args, dataset_traj, smplx_neutral):
    model_posenet, model_trajnet, model_trajnet_control, diffusion_posenet, diffusion_trajnet_eval = models
    
    # We clone and move to device in full_inference
    batch_size = test_batch_traj['cond'].shape[0]
    clip_len = test_batch_traj['cond'].shape[1]
    
    val_output_traj = None
    val_output_pose = None

    for iter_idx in range(args.sample_iter):
        # 1. TrajNet
        traj_feat_dim = dataset_traj.traj_feat_dim
        pose_feat_dim = dataset_traj.pose_feat_dim
        shape_traj = [batch_size, clip_len, traj_feat_dim]
        
        if iter_idx == 0:
            _, val_output_traj = diffusion_trajnet_eval.eval_losses(
                model=model_trajnet, batch=test_batch_traj, shape=shape_traj, progress=False,
                clip_denoised=False, timestep_respacing=args.timestep_respacing_eval,
                cond_fn_with_grad=args.cond_fn_with_grad, compute_loss=False, smplx_model=smplx_neutral
            )
        else:
            test_batch_traj['control_cond'] = torch.zeros([batch_size, clip_len, pose_feat_dim]).to(dist_util.dev())
            test_batch_traj['control_cond'][:, 0:-1] = val_output_pose[:, :, 0].permute(0, 2, 1)[:, :, -pose_feat_dim:]
            test_batch_traj['control_cond'][:, -1] = test_batch_traj['control_cond'][:, -2].clone()
            
            _, val_output_traj = diffusion_trajnet_eval.eval_losses(
                model=model_trajnet_control, batch=test_batch_traj, shape=shape_traj, progress=False,
                clip_denoised=False, timestep_respacing=args.timestep_respacing_eval,
                cond_fn_with_grad=args.cond_fn_with_grad, compute_loss=False, smplx_model=smplx_neutral
            )
            
        # 2. Canonicalization / Recovery (Slow Step)
        motion_repr_clean_root_rec = test_batch_traj['motion_repr_clean'].clone()
        if not args.repr_abs_only:
             motion_repr_clean_root_rec = torch.cat([val_output_traj, test_batch_traj['motion_repr_clean'][:, :, traj_feat_dim:]], dim=-1)
        else:
            motion_repr_clean_root_rec[..., 0] = val_output_traj[..., 0]
            motion_repr_clean_root_rec[..., 2:4] = val_output_traj[..., 1:3]
            motion_repr_clean_root_rec[..., 6] = val_output_traj[..., 3]
            motion_repr_clean_root_rec[..., 7:13] = val_output_traj[..., 4:10]
            motion_repr_clean_root_rec[..., 16:19] = val_output_traj[..., 10:13]
        
        motion_repr_cpu = motion_repr_clean_root_rec.detach().cpu().numpy()
        motion_repr_cpu = motion_repr_cpu * dataset_traj.Std + dataset_traj.Mean
        
        cur_total_dim = 0
        repr_dict_root_rec = {}
        for repr_name in REPR_LIST:
            repr_dict_root_rec[repr_name] = torch.from_numpy(motion_repr_cpu[..., cur_total_dim:(cur_total_dim + REPR_DIM_DICT[repr_name])]).to(dist_util.dev())
            cur_total_dim += REPR_DIM_DICT[repr_name]
        
        rec_ric_data_rec = recover_from_repr_smpl(repr_dict_root_rec, recover_mode='smplx_params', smplx_model=smplx_neutral, return_verts=False)
        rec_ric_data_rec_numpy = rec_ric_data_rec.detach().cpu().numpy()
        
        traj_rec_full = []
        for seq_i in range(batch_size):
            global_orient_mat = rot6d_to_rotmat(repr_dict_root_rec['smplx_rot_6d'][seq_i])
            global_orient_aa = rotation_matrix_to_angle_axis(global_orient_mat)
            body_pose_mat = rot6d_to_rotmat(repr_dict_root_rec['smplx_body_pose_6d'][seq_i].reshape(-1, 6))
            body_pose_aa = rotation_matrix_to_angle_axis(body_pose_mat).reshape(-1, 21, 3)
            smplx_params = {
                'transl': repr_dict_root_rec['smplx_trans'][seq_i].detach().cpu().numpy(),
                'global_orient': global_orient_aa.detach().cpu().numpy(),
                'body_pose': body_pose_aa.reshape(-1, 63).detach().cpu().numpy(),
                'betas': repr_dict_root_rec['smplx_betas'][seq_i].detach().cpu().numpy(),
            }
            repr_dict = get_repr_smplx(positions=rec_ric_data_rec_numpy[seq_i], smplx_params_dict=smplx_params, feet_vel_thre=5e-5)
            new_motion_repr = np.concatenate([repr_dict[key] for key in REPR_LIST], axis=-1)
            new_motion_repr = (new_motion_repr - dataset_traj.Mean) / dataset_traj.Std
            traj_rec_full.append(new_motion_repr[:, 0:22])
            
        traj_rec_full_tensor = torch.tensor(np.asarray(traj_rec_full)).to(dist_util.dev())

        # 3. PoseNet
        if iter_idx == 0:
            test_batch_pose['motion_repr_noisy'] = test_batch_pose['motion_repr_noisy'][:, 0:-1]
            test_batch_pose['motion_repr_clean'] = test_batch_pose['motion_repr_clean'][:, 0:-1]
            
        test_batch_pose['cond'] = test_batch_pose['motion_repr_noisy'].clone()
        test_batch_pose['cond'][:, :, 0:22] = traj_rec_full_tensor
        
        test_batch_pose['cond'] = torch.permute(test_batch_pose['cond'], (0, 2, 1)).unsqueeze(-2)
        
        # Only permute clean repr once
        if iter_idx == 0:
             test_batch_pose['motion_repr_clean'] = torch.permute(test_batch_pose['motion_repr_clean'], (0, 2, 1)).unsqueeze(-2)

        shape_pose = list(test_batch_pose['motion_repr_clean'].shape)
        
        # print(f"DEBUG: iter {iter_idx} shape_pose: {shape_pose}")
        # print(f"DEBUG: iter {iter_idx} cond shape: {test_batch_pose['cond'].shape}")
        
        _, val_output_pose = diffusion_posenet.eval_losses(
            model=model_posenet, batch=test_batch_pose, shape=shape_pose, progress=False,
            clip_denoised=False, timestep_respacing=args.timestep_respacing_eval,
            cond_fn_with_grad=args.cond_fn_with_grad, early_stop=args.early_stop,
            compute_loss=False, smplx_model=smplx_neutral
        )

    return val_output_pose

def measure_fps():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--num_trials', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=2)
    
    # RoHM standard arguments
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--diffusion_steps_posenet", default=1000, type=int)
    parser.add_argument("--diffusion_steps_trajnet", default=100, type=int)
    parser.add_argument("--noise_schedule", default='cosine', choices=['linear', 'cosine'], type=str)
    parser.add_argument("--timestep_respacing_eval", default='', type=str)
    parser.add_argument("--sigma_small", default='True', type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--body_model_path', type=str, default='body_models/smplx_model')
    parser.add_argument('--dataset_root', type=str, default='/mnt/hdd/diffusion_mocap_datasets/AMASS_smplx_preprocessed')
    parser.add_argument("--clip_len", default=145, type=int)
    parser.add_argument('--repr_abs_only', default='True', type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--model_path_trajnet', type=str, default='')
    parser.add_argument('--model_path_trajnet_control', type=str, default='')
    parser.add_argument('--model_path_posenet', type=str, default='')
    
    parser.add_argument('--input_noise', default='True', type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument("--noise_std_smplx_global_rot", default=3, type=float)
    parser.add_argument("--noise_std_smplx_body_rot", default=3, type=float)
    parser.add_argument("--noise_std_smplx_trans", default=0.03, type=float)
    parser.add_argument("--noise_std_smplx_betas", default=0.1, type=float)
    parser.add_argument('--load_noise', default='True', type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument("--load_noise_level", default=3, type=int)
    parser.add_argument("--spacing", default=1, type=int)

    parser.add_argument('--cond_fn_with_grad', default='True', type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument("--sample_iter", default=2, type=int)
    parser.add_argument("--early_stop", default='False', type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument("--mask_scheme", default='full', type=str, choices=['lower', 'upper', 'full'])
    parser.add_argument("--logdir", default='logs', type=str)

    args = parser.parse_args()
    
    import yaml
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    for k, v in cfg.items():
        if hasattr(args, k):
            setattr(args, k, v)
        else:
            setattr(args, k, v)
    
    dist_util.setup_dist()
    fixseed(args.seed)

    # Derive logdir from model path if not explicitly provided
    if args.logdir == 'logs' and args.model_path_posenet:
        args.logdir = '/'.join(args.model_path_posenet.split('/')[:-1])
    elif args.logdir == 'logs' and args.model_path_trajnet:
        args.logdir = '/'.join(args.model_path_trajnet.split('/')[:-1])

    # Load pre-computed body noise if requested (mimicking test_amass_full.py)
    loaded_smplx_noise_dict = None
    if args.load_noise:
        import pickle
        noise_pkl_path = f'data/eval_noise_smplx/smplx_noise_level_{args.load_noise_level}.pkl'
        try:
            with open(noise_pkl_path, 'rb') as f:
                loaded_smplx_noise_dict = pickle.load(f)
        except FileNotFoundError:
            print(f"Warning: Noise file {noise_pkl_path} not found. Proceeding without loaded noise.")

    print("Loading datasets...")
    # Pose dataset for PoseNet (always repr_abs_only=False)
    test_pose_dataset = DataloaderAMASS(
        preprocessed_amass_root=args.dataset_root,
        body_model_path=args.body_model_path,
        amass_datasets=['HumanEva'],
        split='test',
        task='pose',
        spacing=args.spacing,
        repr_abs_only=False, # PoseNet always expects full trajectory repr
        input_noise=args.input_noise,
        noise_std_smplx_global_rot=args.noise_std_smplx_global_rot,
        noise_std_smplx_body_rot=args.noise_std_smplx_body_rot,
        noise_std_smplx_trans=args.noise_std_smplx_trans,
        noise_std_smplx_betas=args.noise_std_smplx_betas,
        load_noise=args.load_noise,
        loaded_smplx_noise_dict=loaded_smplx_noise_dict,
        clip_len=args.clip_len,
        logdir=args.logdir,
        device=dist_util.dev()
    )

    # Traj dataset for TrajNet (uses args.repr_abs_only)
    test_traj_dataset = DataloaderAMASS(
        preprocessed_amass_root=args.dataset_root,
        body_model_path=args.body_model_path,
        amass_datasets=['HumanEva'],
        split='test',
        task='traj',
        spacing=args.spacing,
        repr_abs_only=args.repr_abs_only,
        input_noise=args.input_noise,
        noise_std_smplx_global_rot=args.noise_std_smplx_global_rot,
        noise_std_smplx_body_rot=args.noise_std_smplx_body_rot,
        noise_std_smplx_trans=args.noise_std_smplx_trans,
        noise_std_smplx_betas=args.noise_std_smplx_betas,
        load_noise=args.load_noise,
        loaded_smplx_noise_dict=loaded_smplx_noise_dict,
        clip_len=args.clip_len,
        logdir=args.logdir,
        device=dist_util.dev()
    )

    smplx_neutral = smplx.create(model_path=args.body_model_path, model_type="smplx",
                                gender='neutral', flat_hand_mean=True, use_pca=False).to(dist_util.dev())

    models_all = create_models_and_diffusion(args, test_pose_dataset, test_traj_dataset)

    # Use traj dataset for the batch as it matches the TrajNet input better
    # But we need both to ensure field consistency, though in simplified mode one is enough
    dataloader_traj = torch.utils.data.DataLoader(test_traj_dataset, batch_size=1, shuffle=False)
    batch_traj = next(iter(dataloader_traj))
    
    dataloader_pose = torch.utils.data.DataLoader(test_pose_dataset, batch_size=1, shuffle=False)
    batch_pose = next(iter(dataloader_pose))
    
    def full_inference():
        local_batch_traj = {k: v.to(dist_util.dev()) if isinstance(v, torch.Tensor) else v for k, v in batch_traj.items()}
        local_batch_pose = {k: v.to(dist_util.dev()) if isinstance(v, torch.Tensor) else v for k, v in batch_pose.items()}
        
        # Merge them or just pass both if needed. run_inference_on_batch needs to handle them.
        # Actually, let's just pass batch_pose and ensure it has what TrajNet needs.
        # test_amass_full.py uses both. I will update run_inference_on_batch to take both.
        out = run_inference_on_batch(local_batch_pose, local_batch_traj, models_all, args, test_traj_dataset, smplx_neutral)
        return out

    print(f"\nBenchmarking on {dist_util.dev()}")
    print(f"Clip Length: {args.clip_len} frames")
    
    for _ in range(args.warmup):
        _ = full_inference()
    
    torch.cuda.synchronize()
    times = []
    for i in range(args.num_trials):
        start_time = time.time()
        _ = full_inference()
        torch.cuda.synchronize()
        times.append(time.time() - start_time)
        print(f"Trial {i+1}: {times[-1]:.3f} s")
    
    avg_time = np.mean(times)
    fps = args.clip_len / avg_time
    print(f"\nInference FPS: {fps:.2f} frames/s")

if __name__ == "__main__":
    measure_fps()
