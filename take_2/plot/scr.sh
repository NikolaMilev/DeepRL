FILE1=$1
cat $FILE1 | grep --text "Avg\sQ" | sed "s/Avg Q value: /(/g" | sed "s/Avg testing reward: //g" | sed "s/Avg duration: //g" | sed "s/ /,/g" | sed -e "s/^\(.*\)$/\1),/g"