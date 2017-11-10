exper_name=$1
export PATH="/home/m/anaconda2/bin:$PATH"
for idataset in {2..43}
do
    cd /home/m/ssd/res/oe/$exper_name/output/ms3mn--ds$idataset/
    python ~/Dropbox/spikesort/mlab_parse.py
    cd /home/m/ssd/res/oe/$exper_name/
    kron-view results ms3mn ds$idataset
done
