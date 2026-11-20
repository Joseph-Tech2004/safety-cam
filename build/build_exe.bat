@echo off
rem Builds the program folder dist\FactorySafetyMonitor with the light build
rem environment (ONNX Runtime + DirectML, no PyTorch).
cd /d "%~dp0"
build_env\Scripts\python.exe -m PyInstaller factory_safety_monitor.spec --noconfirm
pause
