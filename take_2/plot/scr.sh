cat training_output.txt | grep "Avg\sQ" | sed "s/Avg Q value: /(/g" | sed "s/Avg testing reward: //g" | sed "s/Avg duration: //g" | sed "s/ /,/g" | sed -e "s/^\(.*\)$/\1),/g"