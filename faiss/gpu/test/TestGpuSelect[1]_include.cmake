if(EXISTS "/home/hadi.sharifi/ws/faiss/faiss/gpu/test/TestGpuSelect[1]_tests.cmake")
  include("/home/hadi.sharifi/ws/faiss/faiss/gpu/test/TestGpuSelect[1]_tests.cmake")
else()
  add_test(TestGpuSelect_NOT_BUILT TestGpuSelect_NOT_BUILT)
endif()