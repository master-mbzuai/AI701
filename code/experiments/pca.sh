#args=("25" "50" "75" "90")
args=("200" "250" "300" "344")
geometry_settings=("80x24+0+0" "80x24+400+0" "80x24+0+400" "80x24+400+400" "80x24+0+800" "80x24+400+800")

cd ..
for i in "${!args[@]}"; do
  geometry=${geometry_settings[$i % ${#geometry_settings[@]}]}
  xterm -geometry $geometry -e "python main.py --d ${args[$i]} --model_name original_pre_pca --epochs 100 --experiment_name pca --lr 0.0001; exit" &  
done

xterm -e "python main.py --d 0 --model_name original_pre_pca --epochs 100 --experiment_name pca --lr 0.0001; exit" &  
