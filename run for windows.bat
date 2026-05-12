@echo off
call cls
REM Activate the virtual environment
call conda activate ./venv

REM Run the Python script
python Detect.py

REM Optional: deactivate the virtual environment (not strictly necessary)
conda deactivate

pause