#!/bin/bash

# DeepQL trainable
#for i in "eclipse" "netbeans" "openoffice"
#do
#	export epochs=1000 base=$i 
#	echo "Executing deepQL_trainable"
#	ipython deepQL_trainable.py
#done

# DeepQL no trainable
#for i in "eclipse" "netbeans" "openoffice"
#do
#        export epochs=1000 base=$i
#        echo "Executing deepQL_no_trainable"
#        ipython deepQL_no_trainable.py
#done

# DMS_QL
for i in "eclipse" "netbeans" "openoffice"
do
        export epochs=1000 base=$i
        echo "Executing DMS_QL"
        ipython baseline_dms_QL.py
done

# DWEN_QL
for i in "eclipse" "netbeans" "openoffice"
do
        export epochs=1000 base=$i
        echo "Executing DWEN_QL"
        ipython baseline_dwen_QL.py
done

