if(EXISTS "/home/hadi.sharifi/ws/faiss/faiss/gpu/test/TestGpuIndexIVFPQ[1]_tests.cmake")
  include("/home/hadi.sharifi/ws/faiss/faiss/gpu/test/TestGpuIndexIVFPQ[1]_tests.cmake")
else()
  add_test(TestGpuIndexIVFPQ_NOT_BUILT TestGpuIndexIVFPQ_NOT_BUILT)
endif()