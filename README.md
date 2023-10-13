# DGMM2024_comptage_cellule
code for DGMM2024: Counting melanocytes with trainable $h$-maxima and connected components counting layers
A version more clean is coming.

## Download dataset
Download the dataset from https://cloud.minesparis.psl.eu/index.php/s/c50xFQFENFZ6I5h

## Install morpholayers
0.Install morpholayer: 
```
cd DGMM2024_comptage_cellule; git clone https://github.com/Jacobiano/morpholayers.git
```

## Test using pretrained model

After changing --DATA_DIR in test.sh to the dir containing database_melanocytes_trp1 :

```bash test.sh```  

This will load the pretrained model weight and using the preprocessed inputs in ./DGMM2024_comptage_cellule/best_h_dataset255/input_np (preprocessed using operation opening-closing with structural element size=3) 

After testing finished, in the directory ```./DGMM2024_comptage_cellule/visualize_test_only_hmaxima``` you can find the groud truth and detected data samples.

## Train and save the model

After changing --DATA_DIR in train.sh to the dir containing database_melanocytes_trp1 :

```bash train.sh```  

This will train the CNN using preprocessed inputs from set1 in ./DGMM2024_comptage_cellule/, the best model weight with lowest validation error will be saved.

