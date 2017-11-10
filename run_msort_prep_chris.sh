# 1st input: file ID to convert and add to datasets.txt
# 2nd input: clear datasets.txt? 1 for yes
export PATH="~/anaconda2/bin:$PATH"
#export PATH=/home/m/git/mountainlab/bin:$PATH
#name=149063045915
name=$1
which python
echo   "Processing file " $name
############## Script starts #########################################
### Convert kwd to mda, subtract reference: ###
fname_in=~/max/data/oe/maze/$name
#fname_in=$name
python2.7 ~/max/git/mountainlab_utils/mdaio.py '~/max/data/oe/maze/hp14/'$name'/*/experiment*raw.kwd' ~/max/BIGFILES/$name.mda '1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32' 1 31
#5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,23,24,25
#python2.7 ~/Dropbox/python/mlpy/mdaio.py '/home/m/ssd/data/oe/maze/149497139278/2017-05-16_17-49-52/experiment1_114.raw.kwd' '/home/m/ssd/res/oe/BIGFILES/9278.mda' '1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64' 0

### Use Mountainsort ###
echo 'ds'$3' '$name>>~/max/res/msort/datasets.txt
mkdir ~/max/res/msort/$name
prv-create ~/max/BIGFILES/$name.mda ~/max/res/msort/$name/raw.mda.prv
# geom: TODO generalize type of geom
cp ~/max/git/mountainlab_utils/geom_tetr8.csv ~/max/res/msort/$name/geom.csv
echo '{"samplerate":30000}'>~/max/res/msort/$name/params.json
