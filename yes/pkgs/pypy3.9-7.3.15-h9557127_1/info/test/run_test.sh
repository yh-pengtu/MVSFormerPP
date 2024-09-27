

set -ex



pypy3 --help
pypy3 -c "import platform; print(platform._sys_version())"
test -f $PREFIX/lib/pypy3.9/lib2to3/Grammar3.9*.pickle
test -f $PREFIX/lib/pypy3.9/lib2to3/PatternGrammar3.9*.pickle
pypy3 -c "from zoneinfo import ZoneInfo; from datetime import datetime; dt = datetime(2020, 10, 31, 12, tzinfo=ZoneInfo('America/Los_Angeles')); print(dt.tzname())"
pypy3 -m test.test_ssl
exit 0
