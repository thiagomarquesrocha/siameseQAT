#!/bin/bash

# DeepQL_topics
for i in "eclipse" "netbeans" "openoffice"
do
        export epochs=1000 base=$i
        echo "Executing deepQL_topics"
        ipython deepQL_topics.py
done

# DeepQL no trainable
for i in "eclipse" "netbeans" "openoffice"
do
        export epochs=1000 base=$i
        echo "Executing deepQL_no_trainable"
        ipython deepQL_no_trainable.py
done

# DeepTL topics
for i in "eclipse" "netbeans" "openoffice"
do
        export epochs=1000 base=$i
        echo "Executing deepTL_topics"
        ipython deepTL_topics.py
done

# DeepTL
for i in "eclipse" "netbeans" "openoffice"
do
        export epochs=1000 base=$i
        echo "Executing deepTL"
        ipython deepTL.py
done

# DeepQL trainable
for i in "eclipse" "netbeans" "openoffice"
do
        export epochs=1000 base=$i
        echo "Executing deepQL_trainable"
        ipython deepQL_trainable.py
done


# DMS
for i in "eclipse" "netbeans" "openoffice"
do
        export epochs=1000 base=$i
        echo "Executing DMS"
        ipython baseline_dms.py
done

# DWEN
for i in "eclipse" "netbeans" "openoffice"
do
        export epochs=1000 base=$i
        echo "Executing DWEN"
        ipython baseline_dwen.py
done

