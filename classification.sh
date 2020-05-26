#!/bin/bash

# DeepQL topics
#for i in "eclipse" "netbeans" "openoffice"
#do
#        export epochs=1000 base=$i method=deepQL_topics
#        echo "Executing deepQL_topics"
#        ipython classification_deep_QL_TL_topics.py
#done

# DeepTL topics
for i in "eclipse" "netbeans" "openoffice"
do
        export epochs=1000 base=$i method=deepTL_topics
        echo "Executing deepTL_topics"
        ipython classification_deep_QL_TL_topics.py
done


# DeepQL no trainable
#for i in "eclipse" "netbeans" "openoffice"
#do
#       export epochs=1000 base=$i method=deepQL_no_trainable
#       echo "Executing deepQL_no_trainable"
#       ipython classification_deep_QL_AND_TL.py
#done

# DeepTL
#for i in "eclipse" "netbeans" "openoffice"
#do
#        export epochs=1000 base=$i method=deepTL
#        echo "Executing deepTL"
#        ipython classification_deep_QL_AND_TL.py
#done


# DMS_QL
#for i in "eclipse" "netbeans" "openoffice"
#do
#       export epochs=1000 base=$i method=DMS_QL
#       echo "Executing DMS_QL"
#       ipython classification_baseline_dms.py
#done

# DWEN_QL
#for i in "eclipse" "netbeans" "openoffice"
#do
#       export epochs=1000 base=$i method=DWEN_QL
#       echo "Executing DWEN_QL"
#       ipython classification_baseline_dwen.py
#done

# DeepQL trainable
#for i in "eclipse" "netbeans" "openoffice"
#do
#        export epochs=1000 base=$i method=deepQL_trainable
#        echo "Executing deepQL_trainable"
#        ipython classification_deep_QL_AND_TL.py
#done
