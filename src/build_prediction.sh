#build_prediction
bazel build -c opt --copt -DMESA_EGL_NO_X11_HEADERS app:prediction_gpu