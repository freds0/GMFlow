• I audited the repository as it exists now. I did not modify code.

  External references used: GMFlow paper (https://arxiv.org/abs/2504.05304), FlowLet paper (https://arxiv.org/abs/2601.05212), official FlowLet implementation
  (https://github.com/sisinflab/FlowLet).

  Executive Verdict

  The current code is a partial and promising 3D adaptation, but it is not yet technically reliable for age-conditioned 3D MRI synthesis at FlowLet quality. The strongest
  parts are the new 3D Haar transform, 3D patch embedding, 3D Gaussian-mixture output head, randomized transition ratio, and order-2 sampler. The largest blockers are
  inference mismatch, weak/fragile validation, global-attention memory scaling, incomplete FlowLet compliance, and insufficient neuroimaging preprocessing guarantees.

  The current implementation can train on 64^3 sample data, but it is not ready for full OpenBHB-scale synthesis or medically meaningful validation without fixes.

  ———

  ## 1. Architecture Audit

  Correctly converted to 3D

  The main 3D model path uses volumetric tensors:

  - PatchEmbed3D uses nn.Conv3d and accepts [B, C, D, H, W]: lib/models/architecture/gmflow3d.py:33
  - The model reconstructs Gaussian mixture means as [B, K, C, D, H, W]: lib/models/architecture/gmflow3d.py:136
  - The current config correctly sets in_channels=8, out_channels=8, sample_size=32 for wavelet-space training: configs/gmflow3d_openbhb_k4.py:13

  I did not find accidental Conv2d usage inside the active 3D GMFlow architecture. The remaining transformer blocks are sequence-based, not explicitly 2D convolutional.

  Major architectural limitations

  The 3D model is still a global DiT-style transformer, not a FlowLet-style 3D U-Net. The code flattens the 3D volume into patch tokens and applies full global attention:
  lib/models/architecture/gmflow3d.py:278

  This is technically valid for [B, C, D, H, W], but it has poor scaling. At 64^3 voxel input with one-level DWT and patch_size=2, the transformer sees 4096 tokens. At
  128^3, it sees 32768 tokens. Full attention then becomes prohibitive.

  The unpatchify path assumes cubic volumes:

  - It uses one scalar npd = sample_size // patch_size: lib/models/architecture/gmflow3d.py:415
  - It reshapes with npd, npd, npd: lib/models/architecture/gmflow3d.py:418

  This prevents native support for OpenBHB/FlowLet-like non-cubic sizes such as 91x109x91, 121x145x121, or raw 182x218x182.

  ———

  ## 2. Flow Matching / GMFlow Audit

  Implemented GMFlow-like components

  The 3D path follows the GMFlow idea of predicting dynamic Gaussian-mixture parameters:

  - Means, log-weights, and log-stds are produced by the denoiser: lib/models/architecture/gmflow3d.py:188
  - The 3D loss evaluates a Gaussian-mixture NLL: lib/models/losses/diffusion_loss.py:40
  - Reverse transition is implemented with Gaussian algebra: lib/models/diffusions/gmflow3d.py:209
  - Order-2 GM-ODE sampling was ported to 3D: lib/models/diffusions/gmflow3d.py:290

  This is broadly consistent with GMFlow’s probabilistic vector-field formulation.

  Important discrepancies

  The implementation is not a clean mathematical reproduction of the GMFlow paper.

  The code clamps and sanitizes predicted Gaussian parameters:

  - NaN/Inf replacement and log-std clamp: lib/models/diffusions/gmflow3d.py:88

  This prevents crashes but changes the learned distribution. It can hide instability rather than solve it.

  The weighted wavelet loss multiplies channel log-likelihoods before the mixture logsumexp: lib/models/losses/diffusion_loss.py:59

  That is not simply a “weighted average NLL.” It changes the joint likelihood geometry of each Gaussian component. A safer formulation would compute per-band NLL terms
  explicitly and then reduce:

  loss = sum(w_c * nll_c for c in channels) / sum(w_c)

  The 3D forward transition has no numerical clamp inside the square root: lib/models/diffusions/gmflow3d.py:131

  Small floating-point negatives can produce NaNs.

  ———

  ## 3. MRI Data Pipeline Audit

  The current dataset class is simple and not neuroimaging-complete.

  It loads .npy volumes, not NIfTI files: lib/datasets/openbhb.py:68

  It performs:

  - per-volume z-score
  - clipping to [-3, 3]
  - scaling to [-1, 1]
  - trilinear resizing to 64x64x64

  Relevant code: lib/datasets/openbhb.py:100

  Missing compared with FlowLet/OpenBHB-grade preprocessing:

  - no affine/orientation handling
  - no voxel spacing validation
  - no NIfTI metadata preservation
  - no N4 bias correction
  - no registration to template
  - no skull stripping
  - no site/harmonization handling
  - no preservation of anatomical aspect ratio beyond trilinear resize

  FlowLet’s official pipeline uses ANTs/FSL-style preprocessing including N4 correction, registration to MNI, skull stripping, resampling, and z-score normalization.

  Actual inspected sample data

  At:

  /home/fred/Projetos/Einstein/OpenBHB_Dataset/openbhb_train_sample/

  I found 10 raw .npy volumes under the configured training root, all shaped:

  (182, 218, 182)

  Example intensity ranges before normalization:

  100053248969: min -363.44, max 2179.02, mean 265.90, std 507.15
  101404752059: min -25.06, max 339.93, mean 41.09, std 77.71
  103116472269: min -85.60, max 1142.74, mean 96.77, std 192.19

  Metadata contains 3227 rows, but the local sample only has 10 matching raw files. Training on this path is therefore a 10-subject smoke test, not meaningful generative
  training.

  There is also a cache path mismatch: prepare_data.sh writes to train/train_cache_64, while the config expects train/quasiraw_3d/train_cache_64: configs/
  gmflow3d_openbhb_k4.py:87

  ———

  ## 4. Age Conditioning Audit

  Age enters the model through Fourier features:

  - AgeEmbedding builds [age, age^2, sin, cos]: lib/models/architecture/gmflow3d.py:62
  - Time and age embeddings are summed: lib/models/architecture/gmflow3d.py:113
  - The combined embedding is passed into transformer blocks: lib/models/architecture/gmflow3d.py:390

  So age is not dead. It can influence synthesis.

  However, it is weak compared with FlowLet. FlowLet uses age conditioning through FiLM and spatial cross-attention. This repository uses a global conditioning vector
  through DiT/AdaLN-style transformer conditioning. There is no explicit multi-scale FiLM modulation and no spatial cross-attention.

  There are also two dropout mechanisms:

  - Diffusion3DAge.train_step replaces age with negative age for classifier-free guidance: lib/models/diffusion_3d_age.py:55
  - _GMDiTTransformer3DModel.forward also randomly drops age inside the denoiser: lib/models/architecture/gmflow3d.py:385

  This can make conditioning inconsistent across layers and batches. I would keep only one CFG dropout mechanism.

  Age normalization is hardcoded to [6, 86]: lib/datasets/openbhb.py:149

  The inspected metadata has max age 86.198494, so some values exceed the assumed normalized range.

  ———

  ## 5. FlowLet Compliance Audit

  Implemented

  The code now has one-level invertible 3D Haar DWT:

  - DWT: lib/models/diffusions/wavelet.py:6
  - IDWT: lib/models/diffusions/wavelet.py:46
  - Training converts voxel MRI to wavelet space when use_wavelet=True: lib/models/diffusions/gmflow3d.py:413
  - Sampling applies IDWT at the end: lib/models/diffusions/gmflow3d.py:577

  This is the most important FlowLet-inspired feature, and it is implemented in the active config.

  Missing or incomplete

  The code does not reproduce FlowLet’s architecture or full training formulation:

  - no multiscale wavelet pyramid
  - no FlowLet 3D U-Net backbone
  - no FiLM conditioning blocks
  - no spatial cross-attention conditioning
  - no explicit LLL/detail loss decomposition
  - no selectable Rectified/CFM/VP/trigonometric flow formulations
  - no native support for FlowLet’s non-cubic MRI shape
  - no FlowLet-grade preprocessing pipeline

  So this is “GMFlow in one-level Haar space,” not a FlowLet implementation.

  ———

  ## 6. Training Pipeline Audit

  Works structurally

  The training pipeline supports:

  - MMCV runner
  - EMA hook
  - AMP/autocast
  - checkpoint saving
  - resume path
  - gradient clipping
  - gradient accumulation

  Relevant files:

  - Training wrapper: train.py:61
  - Model training step: lib/models/diffusion_3d_age.py:55
  - Resume handling: lib/apis/train.py:184
  - Checkpoint hook: lib/runner/hooks/checkpoint.py:29

  Problems

  Validation does not use real validation MRIs. The config uses:

  val=dict(type='OpenBHB', test_mode=True, num_test_volumes=16)

  at configs/gmflow3d_openbhb_k4.py:89

  In test_mode, the dataset returns only synthetic random ages, not real volumes: lib/datasets/openbhb.py:125

  Therefore validation only generates samples. It does not measure reconstruction, age consistency, anatomical fidelity, or held-out quality.

  The evaluation hook defaults to empty metrics: lib/core/evaluation/eval_hooks.py:99

  So current validation can produce visual samples but cannot prove training quality.

  train.sh exposes BATCH_SIZE, LEARNING_RATE, and EPOCHS, but those values are not propagated into the Python config unless the config reads environment variables. It
  currently does not. This means CLI overrides are misleading.

  ———

  ## 7. Inference Pipeline Audit

  This is one of the most severe issues.

  tools/inference.py does not build the model from the training config. Instead, it manually creates a direct 1-channel voxel model:

  - in_channels=1: tools/inference.py:225
  - sample_size=volume_size: tools/inference.py:225
  - Manual GMFlow3D.__new__ construction: tools/inference.py:253

  This is incompatible with the current wavelet-trained config, which expects:

  in_channels=8
  out_channels=8
  sample_size=32
  use_wavelet=True

  The inference script also expects checkpoint keys named model or ema: tools/inference.py:273

  MMCV checkpoints usually store state_dict, optimizer, scheduler, and metadata. So inference checkpoint loading is likely incompatible.

  NIfTI export uses an identity affine: tools/inference.py:381

  That loses anatomical spacing/orientation and is not medically meaningful.

  ———

  ## 8. Memory and Scaling Analysis

  Current config:

  voxel input:      1 x 64 x 64 x 64
  wavelet input:   8 x 32 x 32 x 32
  patch size:      2
  tokens:          16 x 16 x 16 = 4096
  hidden dim:      768
  layers:          12
  heads:           12

  This is feasible but heavy.

  Approximate attention memory:

   Input Volume    Wavelet Size           Tokens                                  Attention Cost    Feasibility
  ━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   64^3                    32^3             4096    ~201M attention values/layer across 12 heads    feasible with checkpointing
  ──────────────  ──────────────  ───────────────  ──────────────────────────────────────────────  ──────────────────────────────────
   128^3                   64^3            32768                   ~12.9B attention values/layer    not feasible
  ──────────────  ──────────────  ───────────────  ──────────────────────────────────────────────  ──────────────────────────────────
   160x192x160         80x96x80          ~76,800                                        enormous    impossible with global attention
  ──────────────  ──────────────  ───────────────  ──────────────────────────────────────────────  ──────────────────────────────────
   182x218x182        91x109x91    non-cubic/odd                        unsupported + impossible    impossible

  Hardware estimate:

   GPU              64^3                     128^3                Full OpenBHB
  ━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━
   RTX 4090 24GB    possible, small batch    no                   no
  ───────────────  ───────────────────────  ───────────────────  ──────────────
   L4 24GB          possible batch 1         no                   no
  ───────────────  ───────────────────────  ───────────────────  ──────────────
   A100 40GB        comfortable for 64^3     no                   no
  ───────────────  ───────────────────────  ───────────────────  ──────────────
   H100 80GB        comfortable for 64^3     still unrealistic    no

  The bottleneck is global attention, not only channel count.

  ———

  # Critical Issues

   Severity    Issue
  ━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Critical    tools/inference.py is incompatible with wavelet-trained checkpoints and builds a 1-channel voxel model instead of the 8-channel wavelet model.
  ──────────  ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   Critical    Validation uses synthetic test_mode=True ages only, not real held-out MRIs, so quality is not measurable.
  ──────────  ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   Critical    Global full attention prevents scaling beyond 64^3; full OpenBHB dimensions are unsupported.
  ──────────  ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   Critical    Architecture assumes cubic shapes and cannot natively handle 91x109x91, 121x145x121, or 182x218x182.
  ──────────  ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   High        Data preprocessing lacks affine/orientation/spacing/skull-stripping/site harmonization guarantees.
  ──────────  ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   High        Age conditioning is weaker than FlowLet: no FiLM, no spatial cross-attention, no multiscale conditioning.
  ──────────  ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   High        Weighted wavelet NLL is not a simple weighted average; weights are inside the mixture likelihood.
  ──────────  ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   High        One-level Haar wavelet is implemented, but FlowLet’s broader multiscale inductive bias is not.
  ──────────  ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   Medium      train.sh CLI flags for batch size, LR, and epochs do not actually alter the config.
  ──────────  ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   Medium      Cache path mismatch prevents prepared cache from being used consistently.
  ──────────  ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   Medium      NaN sanitization stabilizes execution but can hide broken sampling dynamics.
  ──────────  ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   Medium      Resume works partially, but non-DDP dataloader resume is explicitly limited.
  ──────────  ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   Low         TensorBoard image logging exists, but current validation samples are not medically grounded.

  ———

  ## Final Scores

  ### GMFlow Compliance

  6.5 / 10

  The Gaussian-mixture parameterization and transition machinery are present, but numerical sanitization, weighted NLL behavior, and 3D-specific deviations mean it is not a
  clean reproduction.

  ### FlowLet Compliance

  3.5 / 10

  The one-level 3D Haar domain is implemented, but the model lacks FlowLet’s U-Net, FiLM, spatial cross-attention, multiscale design, and preprocessing assumptions.

  ### MRI Readiness

  4 / 10

  Can process .npy MRI-like volumes, but loses metadata, spatial geometry, and high-resolution anatomy.

  ### OpenBHB Compatibility

  4 / 10

  The sample path is readable, but only 10 raw volumes are present locally and preprocessing is minimal.

  ### Age Conditioning Quality

  5 / 10

  Age reaches the network and is used during training/sampling, but conditioning is global and weaker than FlowLet.

  ### Production Readiness

  2.5 / 10

  Training can run as a research prototype, but inference, validation, scaling, and preprocessing are not production-ready.

  ———

  ## Required Code Modifications Before Serious Training

  1. Replace or fix tools/inference.py so it loads the exact training config, builds GMFlow3D(use_wavelet=True), uses 8 wavelet channels, and restores OpenMMLab state_dict
     checkpoints correctly.

  2. Replace synthetic validation with a real held-out OpenBHB validation dataset. Log real metrics: BAP age consistency, SSIM/PSNR where applicable, tissue statistics, and
     3D visual slices.

  3. Replace global full attention with a FlowLet-like 3D U-Net, windowed attention, axial attention, or hierarchical transformer. Current global attention will not scale.
  4. Add non-cubic shape support through explicit (D, H, W) sample sizes, patch-grid tracking, padding, and inverse cropping.
  5. Align preprocessing with FlowLet/OpenBHB assumptions: orientation, affine, spacing, skull stripping or confirmed skull-stripped inputs, template registration, and
     reproducible normalization.

  6. Rework age conditioning into multiscale FiLM/AdaGN or cross-attention blocks.
  7. Split wavelet losses into explicit LLL and detail-band losses instead of weighting log-likelihoods inside the mixture reduction.
  8. Fix train.sh so CLI overrides actually propagate into config or generate config overrides.
  9. Align prepare_data.sh output cache path with configs/gmflow3d_openbhb_k4.py.
  10. Add subband diagnostics: per-band variance, per-band NLL, per-band sample energy, and NaN/Inf counters before sanitization.

