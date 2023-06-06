# Install script for directory: /home/hadi.sharifi/ws/faiss/faiss

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/hadi.sharifi/ws/faiss/faiss/libfaiss.a")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/AutoTune.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/Clustering.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/IVFlib.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/Index.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/Index2Layer.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/IndexAdditiveQuantizer.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/IndexBinary.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/IndexBinaryFlat.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/IndexBinaryFromFloat.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/IndexBinaryHNSW.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/IndexBinaryHash.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/IndexBinaryIVF.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/IndexFlat.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/IndexFlatCodes.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/IndexHNSW.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/IndexIDMap.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/IndexIVF.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/IndexIVFAdditiveQuantizer.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/IndexIVFFlat.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/IndexIVFPQ.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/IndexIVFFastScan.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/IndexIVFAdditiveQuantizerFastScan.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/IndexIVFPQFastScan.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/IndexIVFPQR.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/IndexIVFSpectralHash.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/IndexLSH.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/IndexLattice.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/IndexNNDescent.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/IndexNSG.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/IndexPQ.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/IndexFastScan.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/IndexAdditiveQuantizerFastScan.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/IndexPQFastScan.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/IndexPreTransform.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/IndexRefine.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/IndexReplicas.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/IndexRowwiseMinMax.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/IndexScalarQuantizer.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/IndexShards.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/IndexShardsIVF.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/MatrixStats.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/MetaIndexes.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/MetricType.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/VectorTransform.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/clone_index.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/index_factory.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/index_io.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/impl/AdditiveQuantizer.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/impl/AuxIndexStructures.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/impl/IDSelector.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/impl/DistanceComputer.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/impl/FaissAssert.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/impl/FaissException.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/impl/HNSW.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/impl/LocalSearchQuantizer.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/impl/ProductAdditiveQuantizer.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/impl/LookupTableScaler.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/impl/NNDescent.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/impl/NSG.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/impl/PolysemousTraining.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/impl/ProductQuantizer-inl.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/impl/ProductQuantizer.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/impl/Quantizer.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/impl/ResidualQuantizer.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/impl/ResultHandler.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/impl/ScalarQuantizer.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/impl/ThreadedIndex-inl.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/impl/ThreadedIndex.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/impl/io.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/impl/io_macros.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/impl/kmeans1d.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/impl/lattice_Zn.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/impl/platform_macros.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/impl/pq4_fast_scan.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/impl/simd_result_handlers.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl/code_distance" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/impl/code_distance/code_distance.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl/code_distance" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/impl/code_distance/code_distance-generic.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl/code_distance" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/impl/code_distance/code_distance-avx2.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/invlists" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/invlists/BlockInvertedLists.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/invlists" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/invlists/DirectMap.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/invlists" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/invlists/InvertedLists.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/invlists" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/invlists/InvertedListsIOHook.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/utils/AlignedTable.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/utils/Heap.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/utils/WorkerThread.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/utils/distances.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/utils/extra_distances-inl.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/utils/extra_distances.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/utils/fp16-fp16c.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/utils/fp16-inl.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/utils/fp16.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/utils/hamming-inl.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/utils/hamming.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/utils/ordered_key_value.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/utils/partitioning.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/utils/quantize_lut.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/utils/random.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/utils/simdlib.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/utils/simdlib_avx2.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/utils/simdlib_emulated.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/utils/simdlib_neon.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/utils/utils.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils/distances_fused" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/utils/distances_fused/avx512.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils/distances_fused" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/utils/distances_fused/distances_fused.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils/distances_fused" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/utils/distances_fused/simdlib_based.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils/approx_topk" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/utils/approx_topk/approx_topk.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils/approx_topk" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/utils/approx_topk/avx2-inl.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils/approx_topk" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/utils/approx_topk/generic.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils/approx_topk" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/utils/approx_topk/mode.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils/approx_topk_hamming" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/utils/approx_topk_hamming/approx_topk_hamming.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils/transpose" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/utils/transpose/transpose-avx2-inl.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils/hamming_distance" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/utils/hamming_distance/common.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils/hamming_distance" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/utils/hamming_distance/generic-inl.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils/hamming_distance" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/utils/hamming_distance/hamdis-inl.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils/hamming_distance" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/utils/hamming_distance/neon-inl.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils/hamming_distance" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/utils/hamming_distance/avx2-inl.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/invlists" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/invlists/OnDiskInvertedLists.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/faiss" TYPE FILE FILES
    "/home/hadi.sharifi/ws/faiss/cmake/faiss-config.cmake"
    "/home/hadi.sharifi/ws/faiss/cmake/faiss-config-version.cmake"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/faiss/faiss-targets.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/faiss/faiss-targets.cmake"
         "/home/hadi.sharifi/ws/faiss/faiss/CMakeFiles/Export/5de5d61945e1ac7c8fc01cce1738f759/faiss-targets.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/faiss/faiss-targets-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/faiss/faiss-targets.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/faiss" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/CMakeFiles/Export/5de5d61945e1ac7c8fc01cce1738f759/faiss-targets.cmake")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^()$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/faiss" TYPE FILE FILES "/home/hadi.sharifi/ws/faiss/faiss/CMakeFiles/Export/5de5d61945e1ac7c8fc01cce1738f759/faiss-targets-noconfig.cmake")
  endif()
endif()

