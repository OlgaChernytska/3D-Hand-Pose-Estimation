#/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo export JUPYTER_PATH=${DIR}:$JUPYTER_PATH >> $HOME/.bashrc