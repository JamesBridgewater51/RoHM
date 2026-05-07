import random
import configargparse
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils import dist_util
from tensorboardX import SummaryWriter
from train.training_loop_posenet import TrainLoopPoseNet
from data_loaders.dataloader_amass import DataloaderAMASS
from model.posenet import PoseNet
from diffusion import gaussian_diffusion_posenet
from diffusion.respace import SpacedDiffusionPoseNet
from utils.model_util import create_gaussian_diffusion
from utils.other_utils import *

arg_formatter = configargparse.ArgumentDefaultsHelpFormatter
cfg_parser = configargparse.YAMLConfigFileParser
description = 'RoHM code'
group = configargparse.ArgParser(formatter_class=arg_formatter,
                                      config_file_parser_class=cfg_parser,
                                      description=description,
                                      prog='')
group.add_argument('--config', is_config_file=True, default='', help='config file path')
group.add_argument("--device", default=0, type=int, help="Device id to use.")
# group.add_argument("--seed", default=0, type=int, help="For fixing random seed.")

######################## diffusion setups
group.add_argument("--diffusion_steps", default=1000, type=int, help='diffusion time steps')
group.add_argument("--noise_schedule", default='cosine', choices=['linear', 'cosine'], type=str, help="Noise schedule type")
group.add_argument("--timestep_respacing_eval", default='', type=str)  # if use ddim, set to 'ddimN', where N denotes ddim sampling steps
group.add_argument("--sigma_small", default='True', type=lambda x: x.lower() in ['true', '1'], help="Use smaller sigma values.")

######################## path to AMASS and body model
group.add_argument('--body_model_path', type=str, default='body_models/smplx_model', help='path to smplx model')
group.add_argument('--dataset_root', type=str, default='/mnt/hdd/diffusion_mocap_datasets/AMASS_smplx_preprocessed', help='path to datas')

####################### model setups
group.add_argument('--task', default='pose', type=str, choices=['traj', 'pose'])
group.add_argument("--clip_len", default=145, type=int, help="sequence length for each clip")
### load pretrained checkpoints
group.add_argument('--load_pretrained_model', default='False', type=lambda x: x.lower() in ['true', '1'], help='if load pretrained checkpoint')
group.add_argument('--pretrained_model_path', type=str, default='', help='')

######################## input noise scaling setups
group.add_argument('--input_noise', default='True', type=lambda x: x.lower() in ['true', '1'], help='if add nosie to input conditions')
group.add_argument("--noise_std_smplx_global_rot", default=3, type=float, help="noise ratio for smplx global orientation (unit: degree)")
group.add_argument("--noise_std_smplx_body_rot", default=2, type=float, help="noise ratio for smplx body pose (unit: degree)")
group.add_argument("--noise_std_smplx_trans", default=0.01, type=float, help="noise ratio for smplx global translation (unit: m)")
group.add_argument("--noise_std_smplx_betas", default=0.2, type=float, help="noise ratio for smplx shape param")

######################## loss weight setups
group.add_argument("--weight_loss_rec_repr_full_body", default=1.0, type=float)
group.add_argument("--weight_loss_repr_foot_contact_mse", default=1.0, type=float)
group.add_argument("--weight_loss_joint_pos_global", default=100.0, type=float)
group.add_argument("--weight_loss_joint_vel_global", default=1000.0, type=float)  # 1/1e1/1e2
group.add_argument("--weight_loss_joint_smooth", default=0.0, type=float)
group.add_argument("--start_skating_loss_epoch", default=1000, type=int, help="")
group.add_argument("--weight_loss_foot_skating", default=0.0, type=float)  # 0.1

####################### training setups
group.add_argument("--batch_size", default=32, type=int, help="Batch size during training.")
group.add_argument('--debug', default='False', type=lambda x: x.lower() in ['true', '1'], help='')
group.add_argument("--start_prox_mask_epoch", default=500, type=int, help="which epoch to start to apply prox masks")
group.add_argument("--mask_scheme", default='lower', type=str,
                   choices=['lower', 'lower+upper', 'lower+full', 'lower+upper+full',
                            'random_joints_max15', 'omniposer_three_settings'])
group.add_argument('--use_setting_encoders', default='False', type=lambda x: x.lower() in ['true', '1'],
                   help='use three setting-specific encoders before the shared backbone')
group.add_argument('--num_settings', default=3, type=int, help='number of unified baseline settings')
group.add_argument("--save_dir", default='runs', type=str, help="Path to save checkpoints and results.")
group.add_argument("--lr", default=1e-4, type=float, help="Learning rate.")
group.add_argument("--weight_decay", default=0.0, type=float, help="Optimizer weight decay.")
group.add_argument("--log_interval", default=25000, type=int)
group.add_argument("--save_interval", default=25000, type=int)
group.add_argument("--num_steps", default=1000000_000, type=int)

args = group.parse_args()

def main(args, writer, logdir, logger):
    dist_util.setup_dist(args.device)
    print("creating data loader...")
    amass_train_datasets = ['HumanEva', 'HDM05', 'MoSh', 'Transitions',
                            'ACCAD', 'BMLhandball', 'BMLmovi', 'BMLrub', 'CMU',
                            'DFaust', 'Eyes_Japan_Dataset', 'PosePrior',
                            'SSM', 'GRAB', 'SOMA']
    amass_test_datasets = ['TCDHands', 'TotalCapture', 'SFU']
    if args.debug:
        amass_train_datasets = ['HumanEva']
        amass_test_datasets = ['TCDHands']
    train_dataset = DataloaderAMASS(preprocessed_amass_root=args.dataset_root, split='train',
                                    amass_datasets=amass_train_datasets,
                                    body_model_path=args.body_model_path,
                                    repr_abs_only=False,
                                    input_noise=args.input_noise,
                                    noise_std_smplx_global_rot=args.noise_std_smplx_global_rot,
                                    noise_std_smplx_body_rot=args.noise_std_smplx_body_rot,
                                    noise_std_smplx_trans=args.noise_std_smplx_trans,
                                    noise_std_smplx_betas=args.noise_std_smplx_betas,
                                    task=args.task,
                                    clip_len=args.clip_len,
                                    logdir=logdir,
                                    device=dist_util.dev())
    # [DDP Core] DistributedSampler shards samples per rank; DataLoader shuffle must be disabled when sampler is set.
    train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=False) if dist_util.is_distributed() else None
    # [DDP Core] Keep CLI batch_size as global batch size; split it per rank to avoid silent DDP OOM.
    per_rank_batch_size = max(1, args.batch_size // dist_util.get_world_size())
    if dist_util.is_main_process() and per_rank_batch_size * dist_util.get_world_size() != args.batch_size:
        print('[DDP Core] batch_size={} is not divisible by world_size={}; using per-rank batch_size={} (effective global batch_size={}).'.format(
            args.batch_size, dist_util.get_world_size(), per_rank_batch_size, per_rank_batch_size * dist_util.get_world_size()))
    train_dataloader = DataLoader(train_dataset, batch_size=per_rank_batch_size,
                                  shuffle=(train_sampler is None), sampler=train_sampler,
                                  num_workers=8, drop_last=False)
    # [DDP Core] Rank 0 writes normalization stats; all ranks wait before test split reads them.
    dist_util.barrier()

    test_dataset = DataloaderAMASS(preprocessed_amass_root=args.dataset_root, split='test', spacing=2,
                                   amass_datasets=amass_test_datasets,
                                   body_model_path=args.body_model_path,
                                   repr_abs_only=False,
                                   input_noise=args.input_noise,
                                   noise_std_smplx_global_rot=args.noise_std_smplx_global_rot,
                                   noise_std_smplx_body_rot=args.noise_std_smplx_body_rot,
                                   noise_std_smplx_trans=args.noise_std_smplx_trans,
                                   noise_std_smplx_betas=args.noise_std_smplx_betas,
                                   task=args.task,
                                   clip_len=args.clip_len,
                                   logdir=logdir,
                                   device=dist_util.dev())
    test_sampler = DistributedSampler(test_dataset, shuffle=False, drop_last=False) if dist_util.is_distributed() else None
    test_dataloader = DataLoader(test_dataset, batch_size=per_rank_batch_size,
                                 shuffle=False, sampler=test_sampler,
                                 num_workers=4, drop_last=False)


    print("creating model and diffusion...")
    model = PoseNet(dataset=train_dataset, body_feat_dim=train_dataset.body_feat_dim,
                    latent_dim=512, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1, activation="gelu",
                    body_model_path=args.body_model_path,
                    device=dist_util.dev(),
                    traj_feat_dim=train_dataset.traj_feat_dim,
                    weight_loss_rec_repr_full_body=args.weight_loss_rec_repr_full_body,
                    weight_loss_repr_foot_contact_mse=args.weight_loss_repr_foot_contact_mse,
                    weight_loss_joint_pos_global=args.weight_loss_joint_pos_global,
                    weight_loss_joint_vel_global=args.weight_loss_joint_vel_global,
                    weight_loss_joint_smooth=args.weight_loss_joint_smooth,
                    weight_loss_foot_skating=args.weight_loss_foot_skating,
                    start_skating_loss_epoch=args.start_skating_loss_epoch,
                    use_setting_encoders=args.use_setting_encoders,
                    num_settings=args.num_settings,
                    ).to(dist_util.dev())
    if args.load_pretrained_model:
        weights = torch.load(args.pretrained_model_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(weights, strict=not args.use_setting_encoders)
        print('loaded checkpoint from {}'.format(args.pretrained_model_path))

    # [DDP Core] Wrap only after checkpoint loading so saved keys stay single-rank compatible.
    # Setting encoders are conditionally activated per sample, so their branches may be unused on a rank.
    model = dist_util.wrap_model_ddp(model, find_unused_parameters=args.use_setting_encoders)

    diffusion_train = create_gaussian_diffusion(args, gd=gaussian_diffusion_posenet,
                                                return_class=SpacedDiffusionPoseNet,
                                                num_diffusion_timesteps=args.diffusion_steps,
                                                timestep_respacing=args.timestep_respacing_eval,
                                                device=dist_util.dev())
    diffusion_eval = create_gaussian_diffusion(args, gd=gaussian_diffusion_posenet,
                                                return_class=SpacedDiffusionPoseNet,
                                                num_diffusion_timesteps=args.diffusion_steps,
                                                timestep_respacing=args.timestep_respacing_eval,
                                               device=dist_util.dev())

    print("Training...")
    TrainLoopPoseNet(args, writer=writer, model=model,
                     diffusion_train=diffusion_train, diffusion_eval=diffusion_eval,
                     timestep_respacing_eval=args.timestep_respacing_eval,
                     train_dataloader=train_dataloader, test_dataloader=test_dataloader,
                     logdir=logdir, logger=logger,
                     start_prox_mask_epoch=args.start_prox_mask_epoch, mask_scheme=args.mask_scheme,
                     input_noise=args.input_noise, device=dist_util.dev(),
                     ).run_loop()


if __name__ == "__main__":
    # [DDP Core] Initialize before run directory creation so every rank shares one logdir.
    dist_util.setup_dist(args.device)
    run_id = random.randint(1, 100000) if dist_util.is_main_process() else None
    run_id = dist_util.broadcast_object(run_id)
    logdir = os.path.join(args.save_dir, str(run_id))  # create new path

    writer = SummaryWriter(log_dir=logdir) if dist_util.is_main_process() else None
    logger = get_logger(logdir) if dist_util.is_main_process() else None
    if dist_util.is_main_process():
        print('RUNDIR: {}'.format(logdir))
        sys.stdout.flush()
        logger.info('Let the games begin')  # write in log file
        save_config(logdir, args)
    dist_util.barrier()

    try:
        main(args, writer, logdir, logger)
    finally:
        if writer is not None:
            writer.close()
        dist_util.cleanup_dist()
