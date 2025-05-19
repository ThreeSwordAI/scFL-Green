@echo off
REM Run scripts in order

python scCDCG_dl_train.py 
python scCDCG_dl_fine_tune.py
python scCDCG_dl_test_result_distribution.py
python scCDCG_dl_graph.py
python scCDCG_fl_train.py
python scCDCG_fl_fine_tune.py
python scCDCG_fl_test_result_distribution.py
python scCDCG_fl_graph.py

echo All scripts executed successfully.
pause
