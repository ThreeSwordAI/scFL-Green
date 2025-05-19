@echo off
REM Run scripts in order

python scDLC_dl_train.py 
python scDLC_dl_fine_tune.py
python scDLC_dl_test_result_distribution.py
python scDLC_dl_graph.py
python scDLC_fl_train.py
python scDLC_fl_fine_tune.py
python scDLC_fl_test_result_distribution.py
python scDLC_fl_graph.py

echo All scripts executed successfully.
pause
