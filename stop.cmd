@echo off
setlocal
chcp 65001 >nul
title DevFlow AI - stop
set "DEVFLOW_LAUNCHER_ROOT=%~dp0"
echo 正在停止 DevFlow AI 并自动备份，请等待完成...
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; try { Import-Module (Join-Path $env:DEVFLOW_LAUNCHER_ROOT 'scripts\internal\DevFlowLauncher.psm1') -Force; Stop-DevFlow; exit 0 } catch { Write-Host ('停止或备份失败：' + $_.Exception.Message) -ForegroundColor Red; exit 1 }"
set "DEVFLOW_STOP_EXIT=%errorlevel%"
if not "%DEVFLOW_STOP_EXIT%"=="0" (
  echo.
  echo 请查看上方错误。数据卷和上传文件不会被删除；修复后可再次双击 stop。
  pause
)
exit /b %DEVFLOW_STOP_EXIT%
