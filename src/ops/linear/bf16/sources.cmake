target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/bf16_dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n14336_k5120.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n5120_k6144.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n256_k5120.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n320_k10240.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n10240_k320.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n4_k10240.cu"
)
