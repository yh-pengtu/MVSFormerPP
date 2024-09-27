

set -ex



python --version
test $(python -c "import sys;print('.'.join(str(i) for i in sys.version_info[:3]))") == "3.9.18"
exit 0
