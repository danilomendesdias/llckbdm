!/bin/bash
echo Building...
# export LLCKBDM_LIB_PATH=llckbdm/
# export PYTHONPATH=$LLCKBDM_LIB_PATH

# echo LLCKBDM_LIB_PATH: $LLCKBDM_LIB_PATH
# echo Pythonpath: $PYTHONPATH

rm -Rf build && mkdir build && cd build && cmake .. && make install && cd ..

# if [ "${OSTYPE}" = "linux-gnu" ]; then
#     echo Linux detected, adding $LLCKBDM_LIB_PATH to LD_LIBRARY_PATH.
#     export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LLCKBDM_LIB_PATH
# elif [ "${OSTYPE}" = "darwin"* ]; then
#     echo OSX detected, adding $LLCKBDM_LIB_PATH to DYLD_LIBRARY_PATH.
#     export DYLD_LIBRARY_PATH=$DYDL_LIBRARY_PATH:$LLCKBDM_LIB_PATH
# fi