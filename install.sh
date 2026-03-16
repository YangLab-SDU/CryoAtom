#!/bin/bash
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR" || exit 1
if [ ! -f "./CryoAtom2/checkpoint/CryoNet.pth" ] || [ ! -f "./CryoAtom2/checkpoint/RUNet.pth" ]; then
  echo "Downloading the required weight files for CryoAtom:"
  wget https://yanglab.qd.sdu.edu.cn/CryoAtom/download/checkpoints_v2.0.zip --no-check-certificate
  unzip checkpoints_v2.0.zip
  rm -f checkpoints_v2.0.zip
  if [ ! -f "./CryoAtom2/checkpoint/CryoNet.pth" ] || [ ! -f "./CryoAtom2/checkpoint/RUNet.pth" ]; then
      echo "Please manually download the weight file from https://yanglab.qd.sdu.edu.cn/CryoAtom/download/checkpoints_v2.0.zip, and download it to the checkpoint folder within the CryoAtom2 directory."
      exit 1
  fi
fi
echo "Detected weight files exist"
is_env_installed=$(conda info --envs | grep CryoAtom2 -c)
if [[ "${is_env_installed}" == "0" ]];then
  echo "Deploying conda environment..."
  cuda_version=$(nvcc --version | grep -oP 'V\K[0-9]+\.[0-9]+')
  cuda_major_version=$(echo $cuda_version | cut -d. -f1)
  if [ "$cuda_major_version" -lt 11 ]; then
      echo "Detected CUDA version is below 11"
      conda env create -f linux.yml
  else
      echo "Detected CUDA version is above 11"
      conda env create -f linux.yml
  fi
else
  echo "Detected an existing CryoAtom2 environment, exiting installation";
  exit 1;
fi

if [[ `command -v activate` ]]
then
  source `which activate` CryoAtom2
else
  conda activate CryoAtom2
fi
  
# Check to make sure CryoAtom2 is activated
if [[ "${CONDA_DEFAULT_ENV}" != "CryoAtom2" ]]
then
  echo "Could not run conda activate CryoAtom2, please check the errors";
  exit 1;
fi

python_exc="${CONDA_PREFIX}/bin/python"

$python_exc setup.py install

cd "$SCRIPT_DIR/CryoAtom2/RUNet" || exit 1
getp_dir=$($python_exc -c "import CryoAtom2,os; print(os.path.join(os.path.dirname(CryoAtom2.__file__),'utils','getp'))")
chmod +x $getp_dir
echo "done!"
