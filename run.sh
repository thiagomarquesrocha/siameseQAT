#!/bin/bash
for i in "eclipse" "netbeans"
do
	export epochs=1000 base=$i 
	echo "Executing deepQL_trainable"
	ipython deepQL_trainable.py
done
