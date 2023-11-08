args=("10" "20" "30")
geometry_settings=("80x24+0+0" "80x24+400+0" "80x24+0+400" "80x24+400+400" "80x24+0+800" "80x24+400+800")

cd ..
for i in "${!args[@]}"; do
  geometry=${geometry_settings[$i % ${#geometry_settings[@]}]}
  xterm -geometry $geometry -e "python adaptive_classifier.py --d ${args[$i]}; exit" &  
done