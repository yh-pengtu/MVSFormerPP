



test $(python -c "import cffi;print(cffi.__version__)") == "1.16.0"
IF %ERRORLEVEL% NEQ 0 exit /B 1
exit /B 0
