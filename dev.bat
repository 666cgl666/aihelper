@echo off
setlocal ENABLEEXTENSIONS ENABLEDELAYEDEXPANSION

REM Simple dev runner for AIHelper (Windows cmd)
REM Usage: dev.bat [port]

set PORT=%1
if "%PORT%"=="" set PORT=9020

set SCRIPT_DIR=%~dp0
set VENV_PY="%SCRIPT_DIR%\.venv\Scripts\python.exe"
set UVICORN_ARGS=-m uvicorn app.server:app --reload --host 127.0.0.1 --port %PORT%

echo [AIHelper] Starting FastAPI on http://127.0.0.1:%PORT% (Ctrl+C to stop)
if exist %VENV_PY% (
  %VENV_PY% %UVICORN_ARGS%
) else (
  echo [AIHelper] .venv not found. Falling back to system Python.
  python %UVICORN_ARGS%
)

endlocal

