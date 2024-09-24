# Training Tree Structure

## Train without Retrain
	* < make Train >
## Train with Retrain
	* < make TrainRetrain>

# Evaluating a trained Tree Structure
	* < make Predict>


### Important Files/Directories
	* Ascii Input File
	* Trained Trees "<tree_struct Path>/Info/<Structure_Name>/JSONs/<JSON_File>"

### Parameters inside Makefile
	* python
	* TimeSeries
	* OutFileName
	* Levels
	* tile
	* It
	* Ft
	* Tepoch 
	* Repoch
	* Rcount
	* threshold 
	* Ie
	* Fe 
	* leaves
	* trainedStructure


## Note
	* Training/Prediction is being done in background and all logs are redirected to a nohup file. After each training/prediction make sure to delete that file since it grows and uses a lot of memory.
