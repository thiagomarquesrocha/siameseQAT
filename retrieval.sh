#!/bin/bash

# DeepQL_topics
#for i in "eclipse" "netbeans" "openoffice"
#do
#        export epochs=1000 base=$i
#        echo "Executing deepQL_topics"
#        ipython deepQL_topics.py
#done

# DeepCOREL_gausian
#for i in "eclipse" "netbeans" "openoffice"
#do
#        export epochs=1000 base=$i
#        echo "Executing deepCOREL_gausian"
#        ipython deepCOREL_gausian.py
#done

# DeepCOREL_cosine
#for i in "eclipse" "netbeans" "openoffice"
#do
#	export epochs=1000 base=$i
#	echo "Executing deepCOREL_cosine"
#	ipython deepCOREL_cosine.py
#done

# DeepQL no trainable
#for i in "eclipse" "netbeans" "openoffice"
#do
#        export epochs=1000 base=$i
#        echo "Executing deepQL_no_trainable"
#        ipython deepQL_no_trainable.py
#done

# DeepTL
for i in "eclipse" "netbeans" "openoffice"
do
        export epochs=1000 base=$i
        echo "Executing deepTL"
        ipython deepTL.py
done

# DeepQL trainable
#for i in "eclipse" "netbeans" "openoffice"
#do
#        export epochs=1000 base=$i
#        echo "Executing deepQL_trainable"
#        ipython deepQL_trainable.py
#done


# DMS_QL
#for i in "eclipse" "netbeans" "openoffice"
#do
#        export epochs=1000 base=$i
#        echo "Executing DMS_QL"
#        ipython baseline_dms_QL.py
#done

# DWEN_QL
#for i in "eclipse" "netbeans" "openoffice"
#do
#        export epochs=1000 base=$i
#        echo "Executing DWEN_QL"
#        ipython baseline_dwen_QL.py
#done

