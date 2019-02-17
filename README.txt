The code to recreate all results in this assignment can be found at 

https://github.com/jeffrobots/gatech_ml.git

Cloning the latest version of this repository will also include the data needed to run the software.
Because of the mechanism to link the data folder, this will only work on Linux or a machine with a Unix OS.


To run the code for Assignment 1, use the following steps from the root of the project.

`cd assignment1`
`python ./run_experiment.py --all`

This will take approximately 8 hours to run in total, as some of the default cross validation parameter sweeps are very extensive.

The output results will be placed in assignment1/outputs.



In order to recreate the dataset analysis plots, perform the following steps from the git project root.

`python ./analyze_dataset.py --rain`
`python ./analyze_dataset.py --skyserver`

The resulting plots will be placed in the project root directory.


