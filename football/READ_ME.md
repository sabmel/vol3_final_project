# Using these modules

To use these modules, you will need the proper dataset loaded onto your computer, and you will need to know its full path name. You can use the following code to accomplish this:

### Read in the super long data set
import kagglehub

### Download latest version
path = kagglehub.dataset_download("maxhorowitz/nflplaybyplay2009to2016")

print("Path to dataset files:", path)


