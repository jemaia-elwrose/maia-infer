#!/bin/sh
# Ultralytics depends on `opencv-python`, the desktop build linked against Qt/X11
# (libxcb, libGL, ...), which server and container images don't have:
#   ImportError: libxcb.so.1: cannot open shared object file
# Both builds install into the same `cv2` folder, so replace them with the
# headless build only, at the same version pip resolved.
set -e
VERSION=$(python -c "import importlib.metadata as m; print(m.version('opencv-python-headless'))" 2>/dev/null \
       || python -c "import importlib.metadata as m; print(m.version('opencv-python'))")
pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless >/dev/null 2>&1 || true
pip install --no-deps --force-reinstall "opencv-python-headless==${VERSION}"
python -c "import cv2; print('opencv-python-headless', cv2.__version__)"
