@echo off
chcp 65001 >nul
setlocal DisableDelayedExpansion
call "%~dp0启动RCWA.bat" --check-env
set "RCWA_EXIT_CODE=%ERRORLEVEL%"
echo.
pause
exit /b %RCWA_EXIT_CODE%
