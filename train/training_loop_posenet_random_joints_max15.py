# Copyright (c) Meta Platforms, Inc. and affiliates.
# Part of this code is based on https://github.com/GuyTevet/motion-diffusion-model

import blobfile as bf
import random
from torch.optim import AdamW
import smplx

from utils import dist_util
from diffusion.fp16_util import MixedPrecisionTrainer
from tqdm import tqdm
from diffusion.resample import create_named_schedule_sampler
from utils.other_utils import *

class TrainLoopPoseNetRandomJointsMax15:
    def __init__(self, args, writer, model, diffusion_train, diffusion_eval, timestep_respacing_eval, input_noise,
                 train_dataloader, test_dataloader, logdir, logger, start_prox_mask_epoch, mask_scheme, device='cpu'):
        self.args = args
        self.writer = writer
        self.model = model
        self.diffusion_train = diffusion_train
        self.diffusion_eval = diffusion_eval
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.use_fp16 = False  # deprecating this option
        self.fp16_scale_growth = 1e-3  # deprecating this option
        self.weight_decay = args.weight_decay
        self.timestep_respacing_eval = timestep_respacing_eval
        self.input_noise = input_noise
        self.start_prox_mask_epoch = start_prox_mask_epoch
        self.mask_scheme = mask_scheme

        self.smplx_neutral = smplx.create(model_path=args.body_model_path, model_type="smplx",
                                          gender='neutral', flat_hand_mean=True, use_pca=False).to(device)

        self.step = 0
        self.global_batch = self.batch_size
        self.num_steps = args.num_steps
        self.num_epochs = self.num_steps // len(self.train_dataloader) + 1
        self.sync_cuda = torch.cuda.is_available()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,  # MDM
            use_fp16=self.use_fp16,  # False
            fp16_scale_growth=self.fp16_scale_growth,
        )
        self.save_dir = logdir
        self.logger = logger
        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )

        self.device = torch.device("cpu")
        if torch.cuda.is_available() and dist_util.dev() != 'cpu':
            self.device = torch.device(dist_util.dev())

        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion_train)


    def run_loop(self):
        ######################################## load prox masks
        print('[INFO] skipping PROX joint masks loading since using random_joints_max15...')
        prox_mask_list = []
        print('[INFO] prox masks loaded, get {} prox mask clips in total.'.format(len(prox_mask_list)))

        ######################################## start training
        for epoch in range(self.num_epochs):
            self.model.train()
            traj_feat_dim = self.train_dataloader.dataset.traj_feat_dim
            for batch in tqdm(self.train_dataloader):
                for key in batch.keys():
                    batch[key] = batch[key].to(self.device)
                if not self.input_noise:
                    batch['cond'] = batch['motion_repr_clean'].clone()  # [bs, clip_len, body_feat_dim]
                else:
                    batch['cond'] = batch['motion_repr_noisy'].clone()
                bs, clip_len = batch['motion_repr_clean'].shape[0], batch['motion_repr_clean'].shape[1]

                ####################### add mask, with some schedules
                # random_joints_max15
                mask_joint_n = random.randint(1, 15)
                mask_joint_id = torch.empty(bs, mask_joint_n, dtype=torch.long)
                for i in range(bs):
                    mask_joint_id[i] = torch.randperm(22)[:mask_joint_n]
                for i in range(bs):
                    for k in range(3):
                        batch['cond'][i, :, traj_feat_dim + mask_joint_id[i] * 3 + k] = 0.
                    for k in range(3):
                        batch['cond'][i, :, traj_feat_dim + 22 * 3 + mask_joint_id[i] * 3 + k] = 0.
                    for m_id in mask_joint_id[i]:
                        if m_id > 0:
                            for k in range(6):
                                batch['cond'][i, :, traj_feat_dim + 22 * 3 + 22 * 3 + (m_id - 1) * 6 + k] = 0.
                    if 7 in mask_joint_id[i] or 10 in mask_joint_id[i]:  # left foot
                        batch['cond'][i, :, -4:-2] = 0.
                    if 8 in mask_joint_id[i] or 11 in mask_joint_id[i]:  # right foot
                        batch['cond'][i, :, -2:] = 0.
                if self.input_noise:
                    batch['cond'][:, :, -4:] = 0.

                batch['motion_repr_clean'] = torch.permute(batch['motion_repr_clean'], (0, 2, 1)).unsqueeze(-2)  # [bs, body_feat_dim, 1, clip_len]
                batch['cond'] = torch.permute(batch['cond'], (0, 2, 1)).unsqueeze(-2)

                train_losses = self.run_step(batch)

                if self.step % self.log_interval == 0 and self.step > 0:
                    for key in train_losses.keys():
                        self.writer.add_scalar('train/{}'.format(key), train_losses[key].item(), self.step)
                        print_str = '[Step {:d}/ Epoch {:d}] [train]  {}: {:.10f}'. format(self.step, epoch, key, train_losses[key].item())
                        self.logger.info(print_str)
                        print(print_str)

                if self.step % self.log_interval == 0 and self.step > 0:
                    self.model.eval()
                    for test_step, test_batch in tqdm(enumerate(self.test_dataloader)):
                        for key in test_batch.keys():
                            test_batch[key] = test_batch[key].to(self.device)
                        if not self.input_noise:
                            test_batch['cond'] = test_batch['motion_repr_clean'].clone()  # [bs, clip_len, 263]
                        else:
                            test_batch['cond'] = test_batch['motion_repr_noisy'].clone()
                        bs, clip_len = test_batch['motion_repr_clean'].shape[0], test_batch['motion_repr_clean'].shape[1]

                        ####################### add mask, mask 1-15 joints randomly
                        mask_joint_n = random.randint(1, 15)
                        mask_joint_id = torch.empty(bs, mask_joint_n, dtype=torch.long)
                        for i in range(bs):
                            mask_joint_id[i] = torch.randperm(22)[:mask_joint_n]
                        for i in range(bs):
                            for k in range(3):
                                test_batch['cond'][i, :, traj_feat_dim + mask_joint_id[i] * 3 + k] = 0.
                            for k in range(3):
                                test_batch['cond'][i, :, traj_feat_dim + 22 * 3 + mask_joint_id[i] * 3 + k] = 0.
                            for m_id in mask_joint_id[i]:
                                if m_id > 0:
                                    for k in range(6):
                                        test_batch['cond'][i, :, traj_feat_dim + 22 * 3 + 22 * 3 + (m_id - 1) * 6 + k] = 0.
                            if 7 in mask_joint_id[i] or 10 in mask_joint_id[i]:  # left foot
                                test_batch['cond'][i, :, -4:-2] = 0.
                            if 8 in mask_joint_id[i] or 11 in mask_joint_id[i]:  # right foot
                                test_batch['cond'][i, :, -2:] = 0.
                        if self.input_noise:
                            test_batch['cond'][:, :, -4:] = 0.

                        test_batch['motion_repr_clean'] = torch.permute(test_batch['motion_repr_clean'], (0, 2, 1)).unsqueeze(-2)  # [bs, body_feat_dim, 1, clip_len]
                        test_batch['cond'] = torch.permute(test_batch['cond'], (0, 2, 1)).unsqueeze(-2)
                        shape = list(test_batch['motion_repr_clean'].shape)
                        eval_losses, val_output = self.diffusion_eval.eval_losses(model=self.model, batch=test_batch,
                                                                                  shape=shape, progress=False,
                                                                                  clip_denoised=False, cur_epoch=epoch,
                                                                                  timestep_respacing=self.timestep_respacing_eval,
                                                                                  smplx_model=self.smplx_neutral)
                        for key in eval_losses.keys():
                            if test_step == 0:
                                eval_losses[key] = eval_losses[key].detach().clone()
                            if test_step > 0:
                                eval_losses[key] += eval_losses[key].detach().clone()

                    for key in eval_losses.keys():
                        eval_losses[key] = eval_losses[key] / (test_step + 1)
                        self.writer.add_scalar('eval/{}'.format(key), eval_losses[key].item(), self.step)
                        print_str = '[Step {:d}/ Epoch {:d}] [test]  {}: {:.10f}'.format(self.step, epoch, key, eval_losses[key].item())
                        self.logger.info(print_str)
                        print(print_str)

                    self.model.train()

                if self.step % self.save_interval == 0 and self.step > 0:
                    self.save()

                self.step += 1


    def run_step(self, batch):
        losses = self.forward_backward(batch)
        self.mp_trainer.optimize(self.opt)
        return losses


    def forward_backward(self, batch):
        self.mp_trainer.zero_grad()
        t, weights = self.schedule_sampler.sample(batch['motion_repr_clean'].shape[0], dist_util.dev())
        losses, model_output = self.diffusion_train.training_losses(model=self.model, batch=batch, t=t, noise=None, smplx_model=self.smplx_neutral)
        loss = (losses["loss"] * weights).mean()
        self.mp_trainer.backward(loss)
        return losses


    def ckpt_file_name(self):
        return f"model{(self.step):09d}.pt"


    def save(self):
        def save_checkpoint(params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            filename = self.ckpt_file_name()
            with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
                torch.save(state_dict, f)
            self.logger.info('[*] model saved\n')
        save_checkpoint(self.mp_trainer.master_params)

