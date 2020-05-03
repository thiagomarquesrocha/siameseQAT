#!/bin/bash
#for i in "eclipse" "netbeans" "openoffice"
#do
#	export epochs=1000 base=$i 
#	echo "Executing deepQL_trainable"
#	ipython deepQL_trainable.py
#done

for i in "eclipse" "netbeans" "openoffice"
do
        export epochs=1000 base=$i
        echo "Executing deepQL_no_trainable"
        ipython deepQL_no_trainable.py
done

