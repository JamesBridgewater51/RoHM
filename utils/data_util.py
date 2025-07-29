import torch as th
import numpy as np


def mask_batch_cond(batch, task, args, mask_joint_ids: np.ndarray, num_joints: int, traj_feat_dim: int):
    bs, clip_len, *_ = batch['motion_repr_clean'].shape
    device, dtype = batch['motion_repr_clean'].device, batch['motion_repr_clean'].dtype
    cond_mask = th.ones(bs, clip_len, num_joints, 1 if args.repr_abs_only else 2, device=device, dtype=th.bool)

    # Apply masking to the batch['cond'] tensor, using an offset for the pose task
    offset = traj_feat_dim if task == 'pose' else 0
    # handle `local_positions` masking.
    batch['cond'][:, :, offset+mask_joint_ids*3] = 0.
    cond_mask[:, :, mask_joint_ids, 0] = 0
    if not args.repr_abs_only:
        # handle `local_vel` masking.
        batch['cond'][:, :, offset+num_joints*3+mask_joint_ids*3] = 0.
        cond_mask[:, :, mask_joint_ids, 1] = 0
    batch['cond_mask'] = cond_mask
    return batch