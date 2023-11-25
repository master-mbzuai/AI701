cd ..
xterm -e "python main.py --model_name original --experiment_name adaptive_lr0.0001_epochs_200 --epochs 50 --lr 0.0001; exit"

python main.py --model_name original_pre --experiment_name new_mapping_better --epochs 100 --lr 0.0001