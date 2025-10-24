@echo off
echo Finding Python installation...

REM Try common Python installation paths
if exist "C:\Python313\python.exe" (
    echo Found Python at C:\Python313\python.exe
    C:\Python313\python.exe -m pip install -r requirements-simple.txt
    goto :done
)

if exist "C:\Python312\python.exe" (
    echo Found Python at C:\Python312\python.exe
    C:\Python312\python.exe -m pip install -r requirements-simple.txt
    goto :done
)

if exist "%USERPROFILE%\AppData\Local\Programs\Python\Python313\python.exe" (
    echo Found Python at %USERPROFILE%\AppData\Local\Programs\Python\Python313\python.exe
    "%USERPROFILE%\AppData\Local\Programs\Python\Python313\python.exe" -m pip install -r requirements-simple.txt
    goto :done
)

if exist "%USERPROFILE%\AppData\Local\Programs\Python\Python312\python.exe" (
    echo Found Python at %USERPROFILE%\AppData\Local\Programs\Python\Python312\python.exe
    "%USERPROFILE%\AppData\Local\Programs\Python\Python312\python.exe" -m pip install -r requirements-simple.txt
    goto :done
)

echo Python executable not found in common locations.
echo Please install Python 3.9+ from https://python.org
pause
goto :eof

:done
echo Dependencies installed successfully!
pause
