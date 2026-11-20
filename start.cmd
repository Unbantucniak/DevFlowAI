@echo off
setlocal
chcp 65001 >nul
title DevFlow AI
set "DEVFLOW_LAUNCHER_ROOT=%~dp0"
echo 正在启动 DevFlow AI 完整项目，请等待服务和显卡模型就绪...
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; try { Import-Module (Join-Path $env:DEVFLOW_LAUNCHER_ROOT 'scripts\internal\DevFlowLauncher.psm1') -Force; Start-DevFlow -OpenBrowser; exit 0 } catch { Write-Host ('启动失败：' + $_.Exception.Message) -ForegroundColor Red; exit 1 }"
set "DEVFLOW_LAUNCH_EXIT=%errorlevel%"
if not "%DEVFLOW_LAUNCH_EXIT%"=="0" (
  echo.
  echo 请查看上方错误。日志位于项目目录中的 .data\logs。
  pause
)
exit /b %DEVFLOW_LAUNCH_EXIT%
