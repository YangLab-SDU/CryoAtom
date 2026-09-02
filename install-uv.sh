#!/bin/bash
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR" || exit 1

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Please install it first with:" >&2
  echo "  pip install uv" >&2
  exit 1
fi

if [ ! -f "./CryoAtom2/checkpoint/CryoNet.pth" ] || [ ! -f "./CryoAtom2/checkpoint/RUNet.pth" ]; then
  echo "Downloading the required weight files for CryoAtom:"
  wget https://yanglab.qd.sdu.edu.cn/CryoAtom/download/checkpoints_v2.1.zip --no-check-certificate
  unzip checkpoints_v2.1.zip
  rm -f checkpoints_v2.1.zip
  if [ ! -f "./CryoAtom2/checkpoint/CryoNet.pth" ] || [ ! -f "./CryoAtom2/checkpoint/RUNet.pth" ]; then
      echo "Please manually download the weight file from https://yanglab.qd.sdu.edu.cn/CryoAtom/download/checkpoints_v2.1.zip, and download it to the checkpoint folder within the CryoAtom2 directory."
      exit 1
  fi
fi
echo "Detected weight files exist"

if [ -d ".venv" ]; then
  echo "Detected an existing .venv environment, exiting installation";
  exit 1;
fi

echo "Deploying the uv environment and syncing pinned dependencies (uv sync --locked)..."
uv sync --locked || {
    echo "uv sync --locked failed, please check the errors above";
    exit 1;
}

# Check to make sure the uv environment is activated
source .venv/bin/activate
if [[ "${VIRTUAL_ENV}" != "${SCRIPT_DIR}/.venv" ]]
then
  echo "Could not activate the uv environment, please check the errors";
  exit 1;
fi

python_exc="${VIRTUAL_ENV}/bin/python"

TORCH_HUB_DIR=$("$python_exc" -c "import torch; print(torch.hub.get_dir())")
MODEL_RNALLM_PATH="$TORCH_HUB_DIR/checkpoints/RNA-FM_pretrained.pth"
MODEL_RNALLM_URL="https://yanglab.qd.sdu.edu.cn/CryoAtom/download/RNA-FM_pretrained.pth"
MODEL_RNALLM_SIZE=$(stat -c%s "$MODEL_RNALLM_PATH" 2>/dev/null || echo 0)

if [ ! -s "$MODEL_RNALLM_PATH" ] || [ "$MODEL_RNALLM_SIZE" -ne 1194424423 ]; then
    mkdir -p "$(dirname "$MODEL_RNALLM_PATH")"
    TMP_RNALLM_PATH="${MODEL_RNALLM_PATH}.tmp"
    rm -f "$TMP_RNALLM_PATH"

    wget --no-check-certificate --tries=3 --timeout=60 \
    -O "$TMP_RNALLM_PATH" "$MODEL_RNALLM_URL"

    [ "$(stat -c%s "$TMP_RNALLM_PATH" 2>/dev/null || echo 0)" -eq 1194424423 ] || {
        echo "RNA-FM download failed or corrupted"
        rm -f "$TMP_RNALLM_PATH"
        exit 1
    }

    mv "$TMP_RNALLM_PATH" "$MODEL_RNALLM_PATH"
fi

cd "$SCRIPT_DIR/CryoAtom2/RUNet" || exit 1
getp_dir=$("$python_exc" -c "import CryoAtom2,os; print(os.path.join(os.path.dirname(CryoAtom2.__file__),'utils','getp'))")
chmod +x "$getp_dir"
echo "done!"