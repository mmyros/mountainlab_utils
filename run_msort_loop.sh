export PATH="/home/m/anaconda2/bin:$PATH"
#source activate gen
#source deactivate
#PATH=`echo $PATH | sed -e 's/:\/home\/m\/anaconda2\/bin$//'`
export PATH=/home/m/git/mountainlab/bin:$PATH
echo "">/home/m/ssd/res/oe/msort/datasets.txt
exper_name=$1 #e.g. hp14
cd /home/m/data/oe/maze/$exper_name #hp14/
FILES=*
COUNTER=1 
for name in $FILES
do
    echo   "Processing file # " $COUNTER $name " in exper_name " $exper_name
    bash ~/Dropbox/spikesort/run_msort_prep.sh $name $exper_name $COUNTER
    (( COUNTER++ )) #untested yet
done
bash ~/Dropbox/spikesort/run_msort_clust.sh $name $exper_name
