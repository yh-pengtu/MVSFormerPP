

set -ex



test $(python -c "import cffi;print(cffi.__version__)") == "1.16.0"
exit 0
