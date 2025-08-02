#!/bin/bash
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR" || exit 1

if [[ `command -v activate` ]]
then
  source `which activate` CryoAtom
else
  conda activate CryoAtom
fi
  
# Check to make sure CryoAtom is activated
if [[ "${CONDA_DEFAULT_ENV}" != "CryoAtom" ]]
then
  echo "Could not run conda activate CryoAtom, please check the errors";
  exit 1;
fi
pip uninstall CryoAtom
python_exc="${CONDA_PREFIX}/bin/python"

$python_exc setup.py install
echo "done!"
