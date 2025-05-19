@echo off
REM Run scripts in order

REM python scSMD_dl_train.py 
REM python scSMD_dl_fine_tune.py
python scSMD_dl_test_result_distribution.py
python scSMD_dl_graph.py
python scSMD_fl_train.py
python scSMD_fl_fine_tune.py
python scSMD_fl_test_result_distribution.py
python scSMD_fl_graph.py

echo All scripts executed successfully.
pause
