# DeepQL topics
for i in "netbeans" "openoffice"
do
        export epochs=1000 base=$i method=deepQL_topics
        echo "Executing deepQL_topics"
        ipython cluster_evaluation.py
done

# DeepQL no trainable
for i in "netbeans" "openoffice"
do
        export epochs=1000 base=$i method=deepQL_no_trainable
        echo "Executing deepQL_no_trainable"
        ipython cluster_evaluation.py
done

# DMS
for i in "netbeans" "openoffice"
do
        export epochs=1000 base=$i method=baseline
        echo "Executing DMS"
        ipython cluster_evaluation.py
done


# DeepTL
for i in "eclipse" "netbeans" "openoffice"
do
        export epochs=1000 base=$i method=deepTL
        echo "Executing deepTL"
        ipython cluster_evaluation.py
done

# DWEN
for i in "eclipse" "netbeans" "openoffice"
do
        export epochs=1000 base=$i method=baseline_dwen
        echo "Executing DWEN"
        ipython cluster_evaluation.py
done


# DeepQL trainable
for i in "eclipse" "netbeans" "openoffice"
do
        export epochs=1000 base=$i method=deepQL_trainable
        echo "Executing deepQL_trainable"
        ipython cluster_evaluation.py
done



