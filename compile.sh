#!/bin/bash
# Compile getp on your local machine, required g++ >= 4.8.5
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ver=$(g++ -dumpfullversion 2>/dev/null || g++ -dumpversion)

IFS=. read -r major minor patch <<< "$ver"
patch=${patch:-0}

req_major=4
req_minor=8
req_patch=5

if   (( major > req_major )) \
  || (( major == req_major && minor > req_minor )) \
  || (( major == req_major && minor == req_minor && patch >= req_patch )); then
    echo "OK: g++ >= 4.8.5 (current: $ver)"
else
    echo "ERROR: g++ < 4.8.5 (current: $ver)"
    exit 1
fi

echo "Compiling getp"
cd "$SCRIPT_DIR/CryoAtom2/src/getp" || exit 1
  make clean
  make -j4 all

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
getp_dir=$($python_exc -c "import CryoAtom2,os; print(os.path.join(os.path.dirname(CryoAtom2.__file__),'utils','getp'))")
cp getp $getp_dir
chmod +x $getp_dir
echo "done"


