set -e
Y=/media/gsk/disk4t/yixing
mkdir -p $Y/{cache,logs,src,data}
cat > $Y/env.sh <<'EOT'
export Y=/media/gsk/disk4t/yixing
export XDG_CACHE_HOME=$Y/cache
export PIP_CACHE_DIR=$Y/cache/pip
export HF_HOME=$Y/cache/hf
export TMPDIR=$Y/cache/tmp
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
mkdir -p $TMPDIR
source $Y/miniforge3/etc/profile.d/conda.sh 2>/dev/null || true
EOT
# redirect omniverse dirs in home to big disk
for d in .local/share/ov .cache/ov .nvidia-omniverse .nv; do
  if [ ! -e ~/$d ]; then mkdir -p $Y/home_links/$d; mkdir -p ~/$(dirname $d); ln -s $Y/home_links/$d ~/$d; fi
done
cd $Y
if [ ! -d miniforge3 ]; then
  curl -sSL -o miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash miniforge.sh -b -p $Y/miniforge3
fi
source $Y/env.sh
conda create -y -p $Y/envs/isaac41 python=3.10
conda activate $Y/envs/isaac41
pip install --upgrade pip
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install isaacsim==4.1.0.0 isaacsim-extscache-physics==4.1.0.0 isaacsim-extscache-kit==4.1.0.0 isaacsim-extscache-kit-sdk==4.1.0.0 --extra-index-url https://pypi.nvidia.com
pip install "boto3==1.34.68" "botocore==1.34.68" "s3transfer<0.11"
echo SETUP1_DONE
