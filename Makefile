# 0- Set to python path
python=~/anaconda3/bin/python3.7

# 1- Set path to input time series
TimeSeries=./LCCB_ascii.txt

# 2- Set output file name
OutFileName=Tree_6_5_1_tile_4M_Te_10_Rc_10_Re_5
SCAMP_OutFileName=Tree_6_5_1_tile_4M_Te_10

# 3- set the level capacity for the tree
Levels= 6 5 1

# 4- set tile size
## Train
tile= 4000000
## Test
t_tile= 5000000

# 5- set initial and final segments' for training and testing
## Train 
It= 0
Ft= 29
## Test
Ie= 25
Fe= 25

# 6-  set the number of the leaf nodes for the targte tree
leaves = 30

# 7- set the training and retraining epochs and number of retraining per new segment
Tepoch = 10
Repoch = 5
Rcount = 10

# 8- set the desired threshold for a high correlation (e.g. 0.8 for LCCB dataset)
threshold = 0.8


# 9- set the path to the target tree
#5-5-1 (5M)
trainedStructure = Info/Tree_6_5_1_tile_4M_Te_10_Rc_10_Re_5/JSONs/Tree_6_5_1__start_2020-08-02-10\:52\:19.551789.json 


# 10- set the path to actual MP file
actualMPfile = <Set to actual_mp.csv path if available to avoid running SCAMP> 

Train:
	$(python) run_SCAMP.py -w 100 -Ts $(TimeSeries) -tile $(tile) -iSeg $(It) -fSeg $(Ft) -out $(SCAMP_OutFileName)
	$(python) TreeStruct.py -T -I -L $(Levels) -w 100 -Ts $(TimeSeries) -tile $(tile) -Te $(Tepoch) -R -Rc $(Rcount) -Re $(Repoch) -out $(OutFileName) -sout $(SCAMP_OutFileName) -iSeg $(It) -fSeg $(Ft)

Predict:
	$(python) run_SCAMP.py -w 100 -Ts $(TimeSeries) -tile $(tile) -test_tile $(t_tile) -iSeg $(It) -fSeg $(Ft) -itSeg $(Ie) -ftSeg $(Fe) -test -out $(SCAMP_OutFileName)
	$(python) TreeEval.py -Es $(TimeSeries) -th $(threshold) -w 100 -o $(OutFileName)  -so $(SCAMP_OutFileName) -bw $(t_tile) -iSeg $(Ie) -fSeg $(Fe) -nleaf $(leaves) -mp 1 -s $(trainedStructure)
	#$(python) TreeEval.py -Es $(TimeSeries) -th $(threshold) -w 100 -o $(OutFileName) -bw $(tile) -iSeg $(Ie) -fSeg $(Fe) -nleaf $(leaves) -mp 2 -actuals $(actualMPfile) -s $(trainedStructure)
