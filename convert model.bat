@echo off
call _internal\setenv.bat

"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py" convert ^
    --input-dir "%WORKSPACE%\data_dst" ^
    --output-dir "%WORKSPACE%\data_dst\merged" ^
    --aligned-dir "%WORKSPACE%\data_dst\aligned" ^
    --model-dir "%WORKSPACE%\model" ^
    --model SAE
    
pause