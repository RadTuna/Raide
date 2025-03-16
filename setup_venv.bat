@echo off

REM 사용자에게 프로젝트 이름 입력 받기
set /p PROJECT_NAME="Enter your project name: "

REM 가상 환경 디렉토리 이름 설정
set VENV_DIR=./.venv

REM 파이썬 가상 환경 생성
echo Creating Python virtual environment...
python -m venv %VENV_DIR%

REM 가상 환경 경로 확인
set VENV_PATH=%VENV_DIR%/Scripts/python.exe

REM src 폴더 생성
set SRC_DIR=src
if not exist %SRC_DIR% mkdir %SRC_DIR%

REM main.py 파일 생성 및 내용 작성
set MAIN_FILE=%SRC_DIR%/main.py
echo print("Hello, World!") > %MAIN_FILE%

REM .code-workspace 파일 생성
set WORKSPACE_FILE=./%PROJECT_NAME%.code-workspace

echo { > %WORKSPACE_FILE%
echo    "folders": [ >> %WORKSPACE_FILE%
echo        { >> %WORKSPACE_FILE%
echo            "path": "." >> %WORKSPACE_FILE%
echo        } >> %WORKSPACE_FILE%
echo    ], >> %WORKSPACE_FILE%
echo    "settings": { >> %WORKSPACE_FILE%
echo        "python.pythonPath": "%VENV_PATH%", >> %WORKSPACE_FILE%
echo        "python.venvPath": "%VENV_DIR%" >> %WORKSPACE_FILE%
echo    } >> %WORKSPACE_FILE%
echo } >> %WORKSPACE_FILE%

REM Visual Studio Code에서 워크스페이스 열기
echo Opening VSCode with the workspace...
code %WORKSPACE_FILE%

echo Done.
pause