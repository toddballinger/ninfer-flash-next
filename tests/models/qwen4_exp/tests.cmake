ninfer_add_test(ninfer_qwen4_exp_config_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_config.cpp"
  LIBRARIES ninfer_model_loading)

ninfer_add_test(ninfer_qwen4_exp_bindings_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_bindings.cpp"
  LIBRARIES ninfer_model_loading)
