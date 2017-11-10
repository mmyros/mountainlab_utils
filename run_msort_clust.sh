exper_name=$1
export PATH=/home/m/git/mountainlab/bin:$PATH

### Create pipeline #### 
# w/o curation:
# echo 'ms2mn ms2_002.pipeline --whiten=true --detect_sign=0 --multineighborhood=false --adjacency_radius=500 --mask_out_artifacts=true' >/home/m/ssd/res/oe/$exper_name/pipelines.txt
cd /home/m/ssd/res/oe/$exper_name
# With curation:

echo \
    'ms2mn ms2_002.pipeline --whiten=true --detect_sign=0 --multineighborhood=false --adjacency_radius=500 --mask_out_artifacts=true --curation=~/git/mountainlab/examples/003_kron_mountainsort/curation.script \
    ms3mn ms3.pipeline --whiten=true --detect_sign=0 --compute_metrics=true --adjacency_radius=5- --mask_out_artifacts=true --curation=~/git/mountainlab/examples/003_kron_mountainsort/curation.script' \
    >/home/m/ssd/res/oe/$exper_name/pipelines.txt


### Remove old files, create new directories
rm -r /home/m/ssd/temp/mountainlab
#rm -r  /home/m/ssd/res/oe/$exper_name/output 

### Use Mountainsort ###
#nohup kron-run ms2mn ds2,ds3,ds4,ds5,ds6,ds7,ds8,ds9,ds10,ds11,ds12,ds13,ds14,ds15,ds16,ds17,ds18,ds19,ds20,ds21,ds22,ds23,ds24,ds25,ds26,ds27,ds28,ds29,ds30,ds31,ds32,ds33,ds34,ds35,ds36,ds37,ds38,ds39,ds40,ds41,ds42,ds43 >log_$exper_name.txt & multitail log_$exper_name.txt
nohup kron-run ms3mn ds2,ds3,ds4,ds5,ds6,ds7,ds8,ds9,ds10,ds11,ds12,ds13,ds14,ds15,ds16,ds17,ds18,ds19,ds20,ds21,ds22,ds23,ds24,ds25,ds26,ds27,ds28,ds29,ds30,ds31,ds32,ds33,ds34,ds35,ds36,ds37,ds38,ds39,ds40,ds41,ds42,ds43 >log_$exper_name.txt & multitail log_$exper_name.txt

for idataset in {2...43}
do
    cd /home/m/ssd/res/oe/$exper_name/output/ms3mn--ds$idataset/
    export PATH="/home/m/anaconda2/bin:$PATH"
    python ~/Dropbox/spikesort/mlab_parse.py
done

    
