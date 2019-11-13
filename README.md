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
Create a folder with a name FCPC. Add your train.smi and test.smi files to FCPC folder. Both .smi files only contain the SMILES strings. MOreover, add train_out.csv and test_output.csv files to FCPC as well. Sample smi and csv files are already uploaded in the folder FCPC for reference. You may replace these files with your own files. PLease keep the format (column names and order ) same. FCPC folder contains following 
