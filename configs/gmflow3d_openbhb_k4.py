# GMFlow 3D - OpenBHB Brain MRI Generation conditioned on Age
# K=4 Gaussians, 64x64x64 volume, patch_size=4
# Designed for RTX 3090/4090 (24GB VRAM)
name = 'gmflow3d_openbhb_k4'

model = dict(
    type='Diffusion3DAge',
    diffusion=dict(
        type='GMFlow3D',
        denoising=dict(
            type='GMDiTTransformer3DModel',
            num_gaussians=4,
            logstd_inner_dim=1024,
            gm_num_logstd_layers=2,
            age_dropout_prob=0.1,
            num_attention_heads=12,
            attention_head_dim=64,  # inner_dim = 12 * 64 = 768
            in_channels=1,
            num_layers=12,
            sample_size=64,
            patch_size=4,
            torch_dtype='float32',
            checkpointing=True),
        flow_loss=dict(
            type='GMFlowNLLLoss3D',
            log_cfgs=dict(type='quartile', prefix_name='loss_trans', total_timesteps=1000),
            data_info=dict(
                pred_means='means',
                target='x_t_low',
                pred_logstds='logstds',
                pred_logweights='logweights'),
            weight_scale=2.0),
        num_timesteps=1000,
        timestep_sampler=dict(type='ContinuousTimeStepSampler', shift=1.0, logit_normal_enable=True),
        denoising_mean_mode='U'),
    diffusion_use_ema=True,
    autocast_dtype='bfloat16')

save_interval = 1000
must_save_interval = 20000
eval_interval = 10000
work_dir = 'work_dirs/' + name

train_cfg = dict(
    trans_ratio=0.5,
    prob_age=0.9,  # 10% unconditional for CFG
    diffusion_grad_clip=10.0,
    diffusion_grad_clip_begin_iter=1000,
)
test_cfg = dict(
    volume_size=(1, 64, 64, 64),
    sampler='FlowEulerODE',
    output_mode='mean',
    num_timesteps=16,
    num_substeps=4,
)

optimizer = {
    'diffusion': dict(
        type='AdamW8bit', lr=1e-4, betas=(0.9, 0.95), weight_decay=0.05,
        paramwise_cfg=dict(custom_keys={
            'bias': dict(decay_mult=0.0),
        })
    ),
}

gradient_accumulation_steps = 4  # effective batch size = 2 * 4 = 8

data = dict(
    workers_per_gpu=4,
    train=dict(
        type='OpenBHB',
        data_root='data/openbhb/train/quasiraw_3d',
        metadata_path='data/openbhb/train/quasiraw_3d/metadata.tsv',
        target_shape=(64, 64, 64),
        use_cache=True,
        cache_dir='data/openbhb/train_cache_64'),
    train_dataloader=dict(samples_per_gpu=2),  # 24GB VRAM
    val=dict(
        type='OpenBHB',
        test_mode=True,
        num_test_volumes=16),
    val_dataloader=dict(samples_per_gpu=2),
    test_dataloader=dict(samples_per_gpu=2),
    persistent_workers=True,
    prefetch_factor=4)

lr_config = dict(
    policy='fixed',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001)
checkpoint_config = dict(
    interval=save_interval,
    must_save_interval=must_save_interval,
    by_epoch=False,
    max_keep_ckpts=1,
    out_dir='checkpoints/')

step = 16
substep = 4
guidance_scale = 0.04

evaluation = [
    dict(
        type='GenerativeEvalHook',
        data='val',
        prefix=f'gmode_g{guidance_scale:.2f}_step{step}',
        sample_kwargs=dict(
            test_cfg_override=dict(
                output_mode='mean',
                sampler='FlowEulerODE',
                guidance_scale=guidance_scale,
                num_timesteps=step,
                num_substeps=substep,
            )),
        interval=eval_interval,
        feed_batch_size=2,
        viz_step=16,
        viz_dir='viz/' + name + f'/gmode_g{guidance_scale:.2f}_step{step}',
        save_best_ckpt=False)]

total_iters = 100000
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])

custom_hooks = [
    dict(
        type='ExponentialMovingAverageHookMod',
        module_keys=('diffusion_ema', ),
        interp_mode='lerp',
        interval=1,
        start_iter=0,
        momentum_policy='rampup',
        momentum_cfg=dict(
            ema_kimg=30000, ema_rampup=0.05, batch_size=8, eps=1e-8),
        priority='VERY_HIGH'),
]

runner = dict(
    type='DynamicIterBasedRunnerMod',
    is_dynamic_ddp=False,
    pass_training_status=True,
    ckpt_trainable_only=True,
    ckpt_fp16_ema=True)
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = f'checkpoints/{name}/latest.pth'
workflow = [('train', save_interval)]
use_ddp_wrapper = True
find_unused_parameters = False
cudnn_benchmark = True
opencv_num_threads = 0
mp_start_method = 'fork'
