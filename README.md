# HPE
Heterogeneous Predictors Ensembling for Quantitative Toxicity Prediction


## System Setting:

Our OS is Ubuntu 18.04.3 LTS. We will build a system or a virtual envoirnment using conda for our HPE model development. Please follow the procedure below in the link to set up conda envoirnment.

https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart

Copy the file `lib.sh` in your working directory where you have activated conda envoirnment. 
Run the following command. 
`bash lib.sh `        
This will install all the required libraries. Pleas select "Yes" when asked during libraries installation process.

### FCPC:
Create a folder with a name FCPC in your local machine. Download the following files from our GitHub FCPC folder into your local FCPC folder.


 1. `FCPC.py`    ## This is the code file to train and test the model.
 2. `train.smi`  ## This contains SMILES strings for train set.
 3. `train_output.csv`  ## This contains Activity column and SMILES strings column for train set.  
 4. `test.smi`  ## This contains SMILES strings for test. set
 5. `test_output.csv`  ## This contains Activity column and SMILES strings column for test set.
        
`cd` into FCPC local directory while keeping the conda envoirnment activated. Now run the following command which will train the FCPC predictor on given train data.

`python FCPC.py`

After the training and testing is done, a file with a name `test_preds_FCPC.xlsx` will be generated which contains output for the test data using FCPC predictor. It also generates two other files with names `train_output.csv` and `test_output.csv` which contains the 2D physio-chemical features used in training FCPC predictor.


### FCPCe:
Create a folder with a name FCPCe in your local machine. Download the following files from our GitHub FCPCe folder into your local FCPC folder.


 1. `FCPCe.py`    ## This is the code file to train and test the model.
 2. `train.smi`  ## This contains SMILES strings for train set.
 3. `train_output.csv`  ## This contains Activity column and SMILES strings column for train set.  
 4. `test.smi`  ## This contains SMILES strings for test. set
 5. `test_output.csv`  ## This contains Activity column and SMILES strings column for test set.
        
`cd` into FCPC local directory while keeping the conda envoirnment activated. Now run the following command which will train the FCPC predictor on given train data.

`python FCPC.py`

After the training and testing is done, a file with a name `test_preds_FCPC.xlsx` will be generated which contains output for the test data using FCPC predictor. It also generates two other files with names `train_output.csv` and `test_output.csv` which contains the 2D physio-chemical features used in training FCPC predictor.
