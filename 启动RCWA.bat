@echo off
setlocal DisableDelayedExpansion
rem Optional: remove "rem " below and select your existing company Python.
rem set "MARKRCWA_PYTHON=C:\Company Python\python.exe"
rem The path must name python.exe, with no extra arguments or embedded quotes.

if defined MARKRCWA_PYTHON goto explicit_python

python -c "import sys" >nul 2>&1
if not errorlevel 1 goto path_python
py -3 -c "import sys" >nul 2>&1
if not errorlevel 1 goto py_launcher
echo No working Python was found on PATH or through py -3.
echo Set MARKRCWA_PYTHON in this launcher to an existing python.exe.
echo This launcher does not download, install, or request administrator rights.
set "RCWA_EXIT_CODE=1"
goto finish

:explicit_python
if not exist "%MARKRCWA_PYTHON%" (
  echo The configured MARKRCWA_PYTHON executable does not exist.
  echo Check the full python.exe path in this launcher.
  set "RCWA_EXIT_CODE=1"
  goto finish
)
"%MARKRCWA_PYTHON%" -B -X utf8 "%~dp0run_rcwa.py" %*
goto capture_result

:path_python
python -B -X utf8 "%~dp0run_rcwa.py" %*
goto capture_result

:py_launcher
py -3 -B -X utf8 "%~dp0run_rcwa.py" %*

:capture_result
set "RCWA_EXIT_CODE=%ERRORLEVEL%"

:finish
if /i "%~1"=="--check-env" exit /b %RCWA_EXIT_CODE%
if not "%RCWA_EXIT_CODE%"=="0" pause
exit /b %RCWA_EXIT_CODE%
