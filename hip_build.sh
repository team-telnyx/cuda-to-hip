cd build
cmake -DCMAKE_MODULE_PATH=/opt/rocm-5.3.0/hip/cmake -DFAISS_ENABLE_PYTHON=OFF -DCMAKE_CXX_COMPILER=/opt/rocm-5.3.0/bin/hipcc -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
cd ..

