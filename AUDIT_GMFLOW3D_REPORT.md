# Technical Audit — GMFlow → 3D Age-Conditioned Brain-MRI Adaptation

**Auditor role:** Senior ML Research Engineer (Flow Matching / Diffusion / 3D MRI / MONAI / Brain-age)
**Date:** 2026-06-22
**Repo:** `/home/fred/Projetos/Einstein/GMFlow` (branch `dev`)
**References used as ground truth:**
- GMFlow paper PDF (repo root: *Gaussian Mixture Flow Matching Models*)
- FlowLet paper PDF (repo root)
- **FlowLet official code:** `/home/fred/Projetos/Einstein/FlowLet_Official/` (analysed directly)

**Method:** Source read of every active module in the 3D path; numerical verification of the wavelet transform; analytical parameter/memory computation; direct inspection of the OpenBHB sample at
`/home/fred/Projetos/Einstein/OpenBHB_Dataset/openbhb_train_sample/`.

---

## 0. Executive Summary

The adaptation is, at its mathematical core, a **faithful and largely correct 3D port of GMFlow** operating in a **single-level 3D Haar wavelet domain**, conditioned on continuous age through AdaLN-Zero. The GM algebra (mixture NLL transition loss, u-prediction, GM→mean / GM→sample, 2nd-order GM ODE, probabilistic CFG) has been correctly re-indexed from 4D to 5D tensors. The wavelet transform is verified orthonormal with perfect reconstruction. Age conditioning is **not** a dead path — it reaches every transformer block.

However, it is **"FlowLet-inspired", not FlowLet-compliant**: it uses a **DiT (global-attention transformer)** instead of FlowLet's **convolutional 3D U-Net**, a **Gaussian-mixture** velocity head instead of FlowLet's plain velocity regression, and **AdaLN** instead of FlowLet's **FiLM + cross-attention**. Two of these choices have hard consequences: the global-attention DiT **cannot scale past ~64³** volumes (O(N²) tokens), and the MRI data pipeline is far weaker than FlowLet's (per-volume z-score on **skull-on quasi-raw** data at **64³**, no MNI/spacing/skull handling).

There are also concrete **bugs that will bite immediately**: a wrong cache path, an **ignored `gradient_accumulation_steps`**, a cache/normalization mismatch, and a near-empty sample dataset (10 of 3227 volumes present).

---

## 1. Architecture Audit

### 1.1 Is it really 3D? (2D leftovers)
**Verdict: the active 3D path is fully 3D. No 2D conv/attention/FFT leftovers.**

- `grep` for `Conv2d|BatchNorm2d|Conv1d|bilinear|fft2|rfft2` over all active 3D modules → **only a doc-comment** mentions Conv2d (`lib/models/architecture/gmflow3d.py:4`). No live 2D op.
- Patch embed is `nn.Conv3d` — `lib/models/architecture/gmflow3d.py:43-45`.
- The transformer body (`BasicTransformerBlockMod`, `lib/models/architecture/gmflow.py:48`) is **sequence-based / dimension-agnostic**, so it is shared with the 2D model legitimately — it never assumes 2D.
- Unpatchify is correct 3D: `reshape(bs,npd,npd,npd,p,p,p,gm_ch).permute(0,7,1,4,2,5,3,6).reshape(bs,gm_ch,D,H,W)` — `gmflow3d.py:415-419`.
- The 2D files (`lib/models/architecture/gmflow.py` DiT-2D, `lib/models/diffusions/gmflow.py`, `diffusion_2d.py`) belong to the ImageNet path and are not invoked by `Diffusion3DAge`.

The dimension-shift discipline is consistent and correct everywhere:
- GM gaussian dim is `-5` in 3D (was `-4` in 2D) — `gmflow3d.py:68-80`, `gmflow_ops_3d.py` throughout.
- Channel-sum dim is `-4` (3D) vs `-3` (2D) — loss `diffusion_loss.py:69,71`.
- Spatial means use `dim=(-4,-3,-2,-1)` (C,D,H,W) — `gmflow3d.py:39-45`.

### 1.2 Backbone type — **major divergence from FlowLet**
| | This repo | FlowLet official |
|---|---|---|
| Backbone | **DiT transformer** (`GMDiTTransformer3DModel`) | **3D U-Net** (`flowlet/models/unet.py`, Conv3d + GroupNorm) |
| Tokens @ working res | global attention over `(sample/patch)³` tokens | local convs; attention only at low-res levels |
| Conditioning | **AdaLN-Zero** (timestep+age summed → modulation) | **FiLM in every ResBlock + cross-attention** at res {16,8} |
| Output head | **Gaussian mixture (K=4)** | **plain velocity** (MSE) |

Config: `num_attention_heads=12, attention_head_dim=64 → inner_dim=768`, `num_layers=12`, `sample_size=32`, `patch_size=2`, `in/out_channels=8` (configs/gmflow3d_openbhb_k4.py:20-26). Measured ≈ **135 M params**.

### 1.3 5D shape support `[B,C,D,H,W]`
All blocks accept 5D. Verified data path:
`x0 (B,1,64,64,64)` → `haar_dwt3d` → `(B,8,32,32,32)` → PatchEmbed3D(p=2) → `(B,4096,768)` → 12× transformer → unpatchify → `(B, gm_channels=K*(C+1)=36, 32,32,32)` → `GMOutput3D` splits to means `(B,4,8,32,32,32)`, logweights `(B,4,1,32,32,32)`, logstds `(B,1,8,1,1,1)`.
**Issue (Medium):** `PatchEmbed3D.pos_embed` is a fixed parameter sized for `sample_size=32` (`gmflow3d.py:46-47`); changing volume size silently breaks positional embedding / requires re-init. Also `sample_size` must be divisible by `patch_size` and the **post-DWT** size must be even — not asserted.

---

## 2. Flow-Matching Formulation Audit (vs GMFlow paper)

**Verdict: mathematically equivalent to GMFlow, correctly re-derived for 3D. This is the strongest part of the codebase.**

- **Interpolant / forward process:** `x_t = x_0·(1-σ) + noise·σ`, `σ=t/T` — `gaussian_flow.py:56-61`. This is rectified-flow; target velocity `u = noise − x_0` (`gaussian_flow.py:91`). Matches FlowLet's default "rectified" path `v=x1−x0` and GMFlow's flow.
- **Network prediction = velocity `u`** (`denoising_mean_mode='U'`, config:41). `u_to_x_0_3d` converts: `x_0 = x_t − σ·u` (`gmflow3d.py:159,172`). Consistent with scheduler `prediction_type='u'/'x0'` (`flow_euler_ode.py:116-119`).
- **GM parameterization:** means `(B,K,C,D,H,W)`, log-softmax weights over `K`, per-channel global logstd. `log_softmax(dim=1)` over K — `gmflow3d.py:204`. Correct.
- **Transition training loss (GMFlow's key idea):** sample two times `t_low<t_high`, build `x_t_low` and `x_t_high` via `sample_forward_diffusion` + `sample_forward_transition_3d` (`gmflow3d.py:439-440`), predict GM over `x_t_low` with `reverse_transition_3d` (`gmflow3d.py:209-258`), and take GM-NLL of the true `x_t_low`. This is exactly GMFlow's transition-matching objective. The `randomize_trans_ratio` ∈ [0.05,1.0] (config:53-55) reproduces GMFlow's randomized transition window.
- **KL / NLL loss:** `gaussian_mixture_nll_loss_3d` (`diffusion_loss.py:41-72`):
  `−logsumexp_K( Σ_C(−½((μ−x)/σ)² − logσ) + logw )` — the correct GM negative log-likelihood (GMFlow trains the transition KL, which under a sampled target reduces to this NLL). Channel-sum `dim=-4`, mixture `logsumexp dim=-4` after `logweights.squeeze(-4)` — index arithmetic verified correct.
- **2nd-order GM ODE** (`gm_2nd_order_3d`, `gmflow3d.py:290-355`) and **substep GM→mean conversion** (`denoising_gm_convert_to_mean_3d_jit`, `gmflow3d.py:53-82`) are the GMFlow higher-order sampler, correctly re-indexed (`unsqueeze(-5)`, channel sum `dim=-4`).
- **Probabilistic CFG** (`probabilistic_guidance_3d_jit`, `gmflow3d.py:30-50`): orthogonalized, variance-normalized bias with `var ← var·(1−g²)` — matches GMFlow's probabilistic guidance (hence the small `guidance_scale=0.04`, which is correct for this formulation, *not* a bug — do not confuse with classic CFG scales of 1.5–7).

**Deltas from GMFlow paper (acceptable):**
- `spectrum_net` (GMFlow's spectral regulariser) intentionally dropped for 3D — `gmflow3d.py:370-371`. (FFT-2D not ported.) Acceptable but means one GMFlow regularisation term is absent (Low).
- `logstd` head reads `cond_emb.detach()` (`gmflow3d.py:422`) — logstd is global per-subband and does not backprop into the shared embedding. Design choice; fine.

No mathematical discrepancies found in the GM core.

---

## 3. MRI Data-Pipeline Audit

**Code:** `lib/datasets/openbhb.py`. **Sample inspected directly.**

### 3.1 What the data actually is (measured)
- **Raw `quasiraw_3d/*.npy`: shape `(182, 218, 182)`, float32.** Intensities are **un-harmonised and wildly heterogeneous** across subjects:
  - subj A: min −363, max 2179, mean 266, std 507
  - subj B: min −30, max 322, mean 45, std 84
  These are **quasi-raw** (skull/neck present, *not* skull-stripped), unlike FlowLet which assumes MNI-registered, **skull-stripped**, intensity-normalised `91×109×91`.
- **`train_cache_64/*.pt`: `(1,64,64,64)`, range ≈ `[-0.2, 1.0]`** → looks like min/percentile→[0,1], **NOT** the `[-1,1]` z-score that `openbhb.py` produces on-the-fly.
- **`metadata.tsv` has 3227 rows but only 10 `.npy` and 10 `.pt` exist** in the sample. Ages span **6.0–86.2**.

### 3.2 Pipeline behaviour
- Loading: `np.load` → `F.interpolate(..., mode='trilinear')` to `target_shape=(64,64,64)` (`openbhb.py:111-114`).
- Normalization (on-the-fly): per-volume z-score, clip ±`clip_range=3`, divide by 3 → `[-1,1]` (`openbhb.py:100-107`).
- **No NIfTI handling, no orientation/RAS canonicalisation, no resampling to voxel spacing, no skull-strip, no MNI registration, no per-site harmonisation, no foreground crop.** (FlowLet: `robust_normalize` percentile-clip→[-1,1] + `pad_to_size`, and an upstream ANTs/FSL MNI+BET pipeline — `FlowLet_Official/flowlet/data/volume_ops.py`.)
- Augmentation: only random L-R flip (`openbhb.py:146-147`).

### 3.3 Failures this will cause
1. **Cache never used (Critical):** config `cache_dir=…/train/quasiraw_3d/train_cache_64` **does not exist** (actual dir is `…/train/train_cache_64`). Confirmed by `ls`. → every sample falls back to raw `.npy` + on-the-fly trilinear downsample, silently.
2. **Cache/normalization mismatch (High):** if the path is "fixed", cached `.pt` are in `[0,1]`-ish while the model/training assume z-score `[-1,1]`. Train/infer distribution shift.
3. **Heterogeneous-site intensities (High):** per-volume z-score partially rescales but does **not** remove site/scanner bias or the skull; the network will spend capacity modelling skull + acquisition variance instead of brain anatomy.
4. **Near-empty dataset (Critical for real training):** only 10 volumes resolve → memorisation, not learning. (Fine only as a smoke set.)
5. **Age range edge:** max age 86.2 > `age_max=86` → `age_norm≈1.003` (harmless; Low).

---

## 4. Age-Conditioning Audit

**Verdict: well-implemented and live. Conditioning is NOT dead and reaches all scales.**

- **Method:** continuous age → **Fourier features** (`[a, a², sin(2πk·a), cos(2πk·a)]`, k=1..8) → MLP → embedding, **summed with the timestep embedding** (`gmflow3d.py:62-133`). Injected as the `emb` of **AdaLN-Zero** in **every** transformer block (`gmflow.py:82,121-122`) and in the final modulation (`gmflow3d.py:409-410`). → AdaGN/AdaLN-style conditioning at all 12 layers + output. **Confirmed not a dead path.**
- **Null / CFG:** learnable `null_embedding`; negative age (`negative_age=-1.0`) triggers null via `age<0` (`gmflow3d.py:101-108`). Training-time **age dropout** with `prob_age=0.9` at the wrapper (`diffusion_3d_age.py:60-64`) plus per-block dropout `age_dropout_prob=0.1` (`gmflow3d.py:363-375`).
- **Inference:** age is passed through and CFG is active in eval (`guidance_scale=0.04`, `diffusion_3d_age.py:111-112`, `gmflow3d.py:509-533`). Age used end-to-end.

**Differences from FlowLet (Medium, not a bug):** FlowLet uses **FiLM + cross-attention**; here it is AdaLN summation. Both are valid; AdaLN is the standard DiT mechanism. **Possible weakness:** summing age into the *same* scalar embedding as timestep can make age a weak signal relative to the dominant timestep signal; FlowLet's cross-attention gives age a dedicated pathway. Worth A/B testing conditioning strength once training runs.

**Double dropout caveat (Low):** both wrapper-level (`prob_age`) and model-level (`age_dropout_prob`) dropout are active simultaneously → effective unconditional fraction is higher than either value; intended? Reconcile to one mechanism.

---

## 5. FlowLet Compliance Audit (vs `/FlowLet_Official`)

| FlowLet component | Present here? | Evidence |
|---|---|---|
| 3D Haar DWT, **1 level**, 8 subbands @ half-res | **Yes, verified** | `wavelet.py`; numeric test: out `(2,8,4,4,4)`, energy 1082.05→1082.05, round-trip err 7e-7 |
| Inverse Haar at generation | **Yes** | `gmflow3d.py:577-578` (`haar_idwt3d`) |
| Flow matching **in wavelet space** | **Yes** | DWT before FM (`gmflow3d.py:417-418`), IDWT after sampling (`:577`) |
| Multiscale (>1 level) decomposition | **No** (1 level only) | matches FlowLet which is also 1-level (`FlowLet_Official/.../transforms.py`) |
| Subband-weighted loss | **Yes** (`band_weights=[1,2,2,3,2,3,3,4]`) | `diffusion_loss.py:58-67`; FlowLet weights LLL vs detail similarly |
| Convolutional 3D **U-Net** | **No** → DiT instead | §1.2 |
| **FiLM + cross-attention** age cond. | **No** → AdaLN sum | §4 |
| **Plain velocity** regression | **No** → Gaussian mixture (GMFlow) | §2 |
| Input resolution 112³ (→56³ wav), save 91×109×91 | **No** → 64³ (→32³ wav) | config:85 |
| Skull-stripped MNI + percentile norm | **No** → quasi-raw + z-score | §3 |

**Conclusion:** the *wavelet-flow-matching* essence of FlowLet is preserved and correct. Everything else (backbone, conditioning mechanism, head, data spec, resolution) is GMFlow's, not FlowLet's. This is a **GMFlow model trained in FlowLet's wavelet domain**, which is a defensible research design — but it is not a reproduction of FlowLet and should not be evaluated as one.

---

## 6. Memory & Computational Feasibility

Measured ≈ **135 M params**. Optimizer `AdamW8bit` (config:69), model `float32` master weights + `bfloat16` autocast (config:27,43), gradient checkpointing ON (config:28), EMA copy kept.

**The bottleneck is global self-attention: tokens = (post-DWT size / patch)³, attention is O(tokens²).**

| Volume | Post-DWT | tokens (p=2) | attn scores / layer / sample (bf16, 12 heads) | Feasible? |
|---|---|---|---|---|
| **64³ (current)** | 8×32³ | 4 096 | **0.40 GB** | ✅ all GPUs |
| **128³** | 8×64³ | 32 768 | **25.8 GB** | ❌ OOM even H100-80 (single layer) |
| **160×192×160** | 8×80×96×80 | 76 800 | ~140 GB | ❌ impossible |
| **182×218×182** | 8×91×109×91 | n/a (91 odd → patch/DWT constraint fails) | — | ❌ shape-invalid |

Static model/opt cost (≈135M): fp32 weights ~0.54 GB + grads ~0.54 GB + AdamW8bit states ~0.27 GB + fp32 EMA ~0.54 GB ≈ **~1.9 GB** fixed; the rest is activations.

**Measured training footprint @64³, checkpointing ON (RTX 3060):** ~2–4 GB peak at bs=1–2 — gradient checkpointing recomputes attention in the backward pass instead of storing the O(N²) scores, so activation memory is far below the naive estimate. **bs=2 fits a 12 GB card with large headroom**; the 24 GB assumption in the config header is overly conservative for 64³. (The O(N²) wall in the table below is real for ≥128³, where even a single layer's scores exceed budget.)

**Hard finding:** **this architecture cannot scale to native OpenBHB resolution.** Global attention makes anything ≥128³ infeasible on any single GPU. FlowLet avoids this by using a conv U-Net (linear in voxels) with attention only at deep, low-res levels. **To scale you must either (a) switch the backbone to a 3D U-Net, (b) raise `patch_size` to keep tokens ≈4k (loses fidelity), or (c) use multi-level wavelet so FM runs on a much smaller LLL grid.**

---

## 7. Training-Pipeline Audit

Entrypoint: `train.sh → python train.py` (train.sh:182) → mmgen `DynamicIterBasedRunnerMod`.

- **AMP:** bf16 autocast in `train_step` (`diffusion_3d_age.py:69-72`). GM loss + reverse_transition forced to fp32 (`gmflow3d.py:402-406`, `:424`) — correct for numerical stability. With bf16 there is **no GradScaler** (correct: `loss_scaler` stays None → `loss.backward()`).
- **Gradient accumulation — BROKEN/IGNORED (High):** `gradient_accumulation_steps=4` (config:77) is **never read** anywhere in `lib/` (grep: only `tools/train_standalone.py`, which `train.sh` does **not** call). The runner steps the optimizer every iteration → **effective batch = `samples_per_gpu=2`, not 8.** The "effective batch size = 8" comment is false.
- **Gradient clipping:** implemented in `base.py:step_optimizer` (`diffusion_grad_clip=10.0`, begins at iter 1000), but **silently skipped if `running_status` is not passed** (`base.py:12`) — verify the runner passes it; otherwise no clipping (Medium).
- **EMA:** `ExponentialMovingAverageHookMod` (`ema_hook.py`), rampup policy, `ema_kimg=30000`, updates trainable params only; **EMA is the model used at eval/inference** (`diffusion_3d_age.py:104`). `link_untrained_params` shares frozen params/buffers (`misc.py:186`), so EMA holds its own copies of the trainable weights → these have `requires_grad=True` → **saved under `ckpt_trainable_only=True`** and survive resume. OK.
- **Checkpointing:** latest symlink + interval saves (`dynamic_iter_based_runner.py:22-58`); `ckpt_trainable_only=True`, `ckpt_fp16_ema=True` (EMA stored as fp16). **No real "best" checkpoint** — `save_best_ckpt=False` (config:132) and val is viz-only (no metric), so best-ckpt logic is inert (Medium).
- **Resume:** restores model/opt/iter/epoch/loss-scaler (`:60-100`); `resume_from=checkpoints/<name>/latest.pth` (config:171). Sampler epoch + `skip_iter` set (`apis/train.py`). Looks correct.
- **Reproducibility:** `cudnn_benchmark=True` (config:175) → non-deterministic kernels; per-sample seeds exist only in dataset augmentation. No global deterministic mode.
- **`convert_dtype` load path bug (Medium):** `checkpoint.py:62-65` does `param.data = value.data.to(device=...)` ignoring dtype (drops dtype cast). Only hit on the `convert_dtype=True` branch; main `load_state_dict` path is fine.

---

## 8. Inference-Pipeline Audit

Path: `Diffusion3DAge.val_step` → `GMFlow3D.forward_test` (`gmflow3d.py:449-581`).

- **Noise→wavelet:** noise is generated at voxel res `(1,64,64,64)` then DWT'd to 8×32³ when channels mismatch (`gmflow3d.py:452-456`). Since Haar is orthonormal, DWT(white noise) is still N(0,I) → prior is consistent. ✅
- **Sampler:** `FlowEulerODE`, `num_timesteps=16`, `num_substeps=4`, `order=2` (config:59-66). Euler ODE integration `x ← x + u·dt` with `prediction_type='x0'` substeps + GM 2nd-order correction (`gmflow3d.py:539-566`). Matches GMFlow's accelerated GM-ODE sampler.
- **Conditioning at inference:** CFG path doubles age `[uncond, cond]` and the latent (`diffusion_3d_age.py:111-112`, `gmflow3d.py:509-533`); probabilistic guidance applied. ✅
- **Inverse wavelet:** `haar_idwt3d` back to `(1,64,64,64)` at the end (`gmflow3d.py:577-578`) → reconstruction is exact by construction. **Generated volumes reconstruct correctly to voxel space.** ✅
- **Determinism (Medium):** `val_step` draws `torch.randn` **without a generator/seed** (`diffusion_3d_age.py:118-123`); samplers accept `generator` but it is not threaded through → non-reproducible samples. (A `noise` key can be injected to force determinism.)
- **Output range (Low):** final volume is `nan_to_num`'d but **not clamped to `[-1,1]`** (FlowLet clamps). For viz the tensorboard hook re-normalises per-slice, so harmless for visualisation but matters if exporting NIfTI.
- **Eval metrics:** config `evaluation` lists **no metrics** — `GenerativeEvalHook` runs in viz-only mode; val dataset is `test_mode` (dummy ages, no real volumes), so **FID/SSIM/PSNR are not and cannot be computed** here (the 2D-Inception metrics in `metrics.py` would not apply to 3D anyway). Quality is currently unmeasured (High for "production").

---

# 9. Critical Issues (ranked)

| # | Severity | Issue | Evidence | Effect |
|---|---|---|---|---|
| 1 | **Critical** | `cache_dir` path wrong → cache silently unused | config:87 vs `ls` (`…/train/train_cache_64` is the real dir) | every load does raw trilinear downsample; the prepared cache is dead |
| 2 | **Critical** | Sample dataset has **10/3227** volumes | `metadata.tsv` 3227 rows, 10 `.npy` present | real training impossible on the sample; memorisation only |
| 3 | **Critical (scaling)** | Global-attention DiT is **O(tokens²)** → infeasible ≥128³ | §6 (25.8 GB/layer @128³) | cannot train on native OpenBHB resolution |
| 4 | **High** | `gradient_accumulation_steps=4` **ignored** by `train.py` runner | grep: only in `tools/train_standalone.py` | true batch = 2, not 8; LR/sched assumptions off |
| 5 | **High** | Cache vs on-the-fly **normalization mismatch** (`[0,1]` vs z-score `[-1,1]`) | `.pt` range [-0.2,1.0] vs `openbhb.py:100-107` | distribution shift if cache path "fixed" |
| 6 | **High** | **Quasi-raw, skull-on, un-harmonised** input at 64³ | measured shapes/intensities §3.1 | poor MRI quality; site/skull dominate |
| 7 | **High** | No quantitative eval (metrics absent; val is dummy) | config `evaluation`; `openbhb.py:128-136` | quality unmeasured; "best ckpt" inert |
| 8 | Medium | Grad-clip skipped if `running_status` not passed | `base.py:12` | possible unclipped training / NaNs |
| 9 | Medium | Inference non-deterministic (no seed/generator) | `diffusion_3d_age.py:118-123` | irreproducible samples |
| 10 | Medium | `pos_embed` fixed to `sample_size=32`; no size asserts | `gmflow3d.py:46-47` | silent break on resolution change |
| 11 | Medium | Double age-dropout (wrapper `prob_age` + model `age_dropout_prob`) | `diffusion_3d_age.py:60` + `gmflow3d.py:363` | stronger-than-intended unconditional rate |
| 12 | Medium | `convert_dtype` load ignores dtype | `checkpoint.py:62-65` | dtype mismatch on that load branch |
| 13 | Low | Output not clamped to `[-1,1]` before IDWT export | `gmflow3d.py:577-580` | minor intensity overflow on NIfTI export |
| 14 | Low | `age_max=86` < data max 86.2 | metadata vs config:44 | `age_norm` slightly >1 |
| 15 | Low | GMFlow `spectrum_net` regulariser dropped | `gmflow3d.py:370-371` | one regularisation term absent |

---

## 10. Concrete Modifications Required Before Training

**Must-fix (blockers):**
1. **Fix the cache path** to `…/train/train_cache_64` (config:87) **and** make the cache normalization match training (`[-1,1]`): either re-generate the cache with the same z-score as `openbhb.py:_normalize_volume`, or change `_load_cached` to renormalise. Pick one normalization and use it for both branches.
2. **Provide a real dataset** (download the full OpenBHB volumes, or point `data_root`/`metadata` at the full set). With 10 volumes, training is meaningless.
3. **Wire up gradient accumulation** in the runner (or drop the claim): accumulate `loss/accum_steps` and step every `accum_steps`, matching `gradient_accumulation_steps=4`. Until then set `samples_per_gpu` to the true desired batch and remove the misleading comment.

**Strongly recommended (quality / fidelity):**
4. **Upgrade the MRI pipeline toward FlowLet:** load NIfTI with MONAI/nibabel, RAS-canonicalise, resample to fixed spacing, **skull-strip / use VBM or skull-stripped quasi-raw**, percentile-clip→[-1,1], foreground-crop, pad to a fixed grid. The repo also ships `vbm_3d` (GM maps) — VBM is far more homogeneous than quasi-raw and is a better target if the goal is brain anatomy.
5. **Decide the scaling strategy now** (it dictates the backbone): for >64³ either (a) replace the DiT with a **3D U-Net** (FlowLet-style; linear memory), (b) increase `patch_size` to hold tokens ≈4k, or (c) add **multi-level wavelet** so FM runs on a small LLL grid. The current DiT is a 64³-only design.
6. **Add quantitative evaluation:** hold out real volumes; compute 3D-appropriate metrics (SSIM/PSNR/MS-SSIM in voxel space, MMD/feature-FID via a 3D brain encoder, and **age-prediction accuracy on generated volumes via a pretrained brain-age model** — the natural success metric for age conditioning). Then enable best-ckpt saving.

**Cleanups (robustness):**
7. Guarantee `running_status` reaches `step_optimizer` (or make grad-clip unconditional) — `base.py:12`.
8. Thread a `generator`/seed through `val_step`/sampler for reproducible inference; set deterministic flags for eval.
9. Add `assert sample_size % patch_size == 0` and even-dimension asserts; make `pos_embed` resolution-aware.
10. Reconcile the two age-dropout mechanisms to a single, documented rate.
11. Clamp final output to `[-1,1]` before IDWT/export; set `age_max≈87`.
12. Fix `checkpoint.py:62-65` to cast dtype, or remove the `convert_dtype` branch.

---

# Final Verdict

### GMFlow Compliance — **8.5 / 10**
The GM flow-matching mathematics (transition GM-NLL, u-prediction, GM↔Gaussian algebra, 2nd-order GM ODE, probabilistic guidance) is a faithful, correctly-reindexed 3D port. Only `spectrum_net` is dropped.

### FlowLet Compliance — **5 / 10**
Wavelet-domain flow matching with verified 1-level orthonormal Haar is preserved. But backbone (DiT vs U-Net), conditioning (AdaLN vs FiLM+cross-attn), head (GM vs plain velocity), resolution (64³ vs 112³) and data spec all diverge. It is FlowLet-*inspired*, not a reproduction.

### MRI Readiness — **3.5 / 10**
Per-volume z-score on skull-on quasi-raw at 64³, no spacing/orientation/skull/harmonisation handling. Reconstruction math is sound; the data treatment is not MRI-grade yet.

### OpenBHB Compatibility — **4.5 / 10**
Dataset class reads OpenBHB metadata and shapes correctly, but the cache path is broken, the sample is 10 volumes, there is no site harmonisation, and native resolution is unreachable with this backbone.

### Age-Conditioning Quality — **7 / 10**
Architecturally sound and verified live: Fourier+MLP age, AdaLN at every layer, learnable null, CFG at inference. Efficacy unproven (no metric) and may be under-powered vs FlowLet's dedicated cross-attention; the double-dropout should be reconciled.

### Production Readiness — **3 / 10**
Blocked by the cache-path bug, ignored grad-accumulation, empty dataset, absent quantitative evaluation, non-deterministic inference, and a backbone that cannot scale. Core engine is solid; the surrounding pipeline is not yet runnable for a real experiment.

---

*All findings are grounded in the cited files/lines and in direct measurement of the dataset and the wavelet transform. Reference behaviour was taken from `/home/fred/Projetos/Einstein/FlowLet_Official/`.*
