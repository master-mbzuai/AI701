args=("50" "75")
geometry_settings=("80x24+0+0" "80x24+400+0" "80x24+0+400" "80x24+400+400" "80x24+0+800" "80x24+400+800")

cd ..
for i in "${!args[@]}"; do
  geometry=${geometry_settings[$i % ${#geometry_settings[@]}]}
  xterm -geometry $geometry -e "python main.py --d ${args[$i]} --model_name adaptive --epochs 30 --experiment_name adaptive_lr0.0001_epochs_50_sched_bigger_images --lr 0.0001; exit" &  
done