export PATH="/home/m/anaconda2/bin:$PATH"
#source activate gen
#source deactivate
#PATH=`echo $PATH | sed -e 's/:\/home\/m\/anaconda2\/bin$//'`
export PATH=/home/m/git/mountainlab/bin:$PATH
name=149063045915
############## Script starts #########################################
### Convert kwd to mda, subtract reference: ###
python2.7 ~/Dropbox/python/mlpy/mdaio.py '/home/m/ssd/data/oe/maze/149063045915/2017-03-27_12-00-59/experiment1_114.raw.kwd' /home/m/ssd/res/oe/BIGFILES/$name.mda '1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32' 1
#python2.7 ~/Dropbox/python/mlpy/mdaio.py '/home/m/ssd/data/oe/maze/149497139278/2017-05-16_17-49-52/experiment1_114.raw.kwd' '/home/m/ssd/res/oe/BIGFILES/9278.mda' '1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64' 0

### Use Mountainsort ###
echo 'ds1 '$name>/home/m/ssd/res/oe/msort/datasets.txt
mkdir /home/m/ssd/res/oe/msort/$name
prv-create /home/m/ssd/res/oe/BIGFILES/$name.mda /home/m/ssd/res/oe/msort/$name/raw.mda.prv
# geom: TODO generalize type of geom; maybe mask dead channels
cp ~/Dropbox/spikesort/mlab/geom_tetr8.csv /home/m/ssd/res/oe/msort/$name/geom.csv

echo 'ms2mn ms2_002.pipeline --whiten=true --detect_sign=0 --multineighborhood=true --adjacency_radius=5 --mask_out_artifacts=true' >/home/m/ssd/res/oe/msort/pipelines.txt
echo '{"samplerate":30000}'>/home/m/ssd/res/oe/msort/$name/params.json
rm -r /home/m/ssd/temp/mountainlab
rm -r  /home/m/ssd/res/oe/msort/output

cd /home/m/ssd/res/oe/msort/
nohup kron-run ms2mn ds1>log_$name.txt & multitail log_$name.txt
kron-view results ms2mn ds1

############## Script starts #########################################
# next, run kron-view
