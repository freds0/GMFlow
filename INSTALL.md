 python -m pip install --no-build-isolation \
    git+https://github.com/Lakonik/mmgeneration.git@500a93e474638c87dcaf8fad94cfbe1ef29acd1f


  conda create -n gmflow-legacy python=3.9 -y
  conda activate gmflow-legacy

  python -m pip install -U "pip<27" "setuptools<70" wheel
  python -m pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

  python -m pip install openmim
  python -m pip install "mmcv-full==1.7.2" \
    -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html

  Then install the rest:

  python -m pip install git+https://github.com/Lakonik/mmgeneration.git@500a93e474638c87dcaf8fad94cfbe1ef29acd1f
  python -m pip install git+https://github.com/huggingface/diffusers.git@57084dacc5275d7212513b24837b60a28e55603d
  python -m pip install transformers torchsde tensorboard wandb bitsandbytes yapf==0.40.1 pandas matplotlib






  If it still tries to build a wheel and fails, clone/install editable:

  mkdir -p /tmp/openmmlab-src
  cd /tmp/openmmlab-src

  git clone https://github.com/Lakonik/mmgeneration.git
  cd mmgeneration
  git checkout 500a93e474638c87dcaf8fad94cfbe1ef29acd1f

  python -m pip install --no-build-isolation -e .
  cd ~/Projetos/Einstein/GMFlow

  Then install the remaining deps:

  python -m pip install --no-build-isolation \
    git+https://github.com/huggingface/diffusers.git@57084dacc5275d7212513b24837b60a28e55603d

  python -m pip install transformers torchsde tensorboard wandb bitsandbytes yapf==0.40.1 pandas matplotlib

  One more important correction: you currently have numpy==2.0.2. Old OpenMMLab stacks often break with NumPy 2.x. Pin NumPy before testing:

  python -m pip install "numpy<2"

  Then verify:

  python - <<'PY'
  import torch, mmcv, mmgen
  print("torch", torch.__version__, torch.version.cuda)
  print("mmcv", mmcv.__version__)
  print("mmgen", mmgen.__version__)
  from mmcv.runner import get_dist_info
  from mmgen.models import build_model
  print("OpenMMLab legacy stack OK")
  PY

  Then:

  bash train.sh
