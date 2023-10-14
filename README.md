# DGMM2024_comptage_cellule
code for DGMM2024: Counting melanocytes with trainable $h$-maxima and connected components counting layers
A version more clean is coming.

## 0.Download dataset
Download the dataset from https://cloud.minesparis.psl.eu/index.php/s/c50xFQFENFZ6I5h

## 1.Install morpholayers
Install morpholayer: 
```
cd DGMM2024_comptage_cellule; git clone https://github.com/Jacobiano/morpholayers.git
```

## 2.Generate preprocessed numpy datasets using opening-closing with structural element size=3 (This step can be skipped, as this Github directory already contains the pre-generated preprocessed numpy dataset):

After changing --DATA_DIR in generate_preprocessed_numpy_dataset.sh to the dir containing database_melanocytes_trp1 :
```
bash generate_preprocessed_numpy_dataset.sh
```
This will create a directory named ./DGMM2024_comptage_cellule/best_h_dataset255, which contains the preprocessed numpy files and best h parameter ground truth.

## 3.Test using pretrained model

After changing --DATA_DIR in test.sh to the dir containing database_melanocytes_trp1 :

```bash test.sh```  

This will load the pretrained model weight and using the preprocessed inputs. 

After testing finished, in the directory ```./DGMM2024_comptage_cellule/visualize_test_only_hmaxima``` you can find the groud truth and detected data samples.

## Train from scratch and save the model

After changing --DATA_DIR in train.sh to the dir containing database_melanocytes_trp1 :

```bash train.sh```  

This will train the CNN using preprocessed inputs from set1 in ./DGMM2024_comptage_cellule/, the best model weight with lowest validation error will be saved for each epoch.

    If you find this code useful in your research, please consider citing:

    @inproceedings{VelascoBMVC2022,
    Author = {Velasco-Forero, S. and Rhim, A. and Angulo, J.},
    Title = {Fixed Point Layers for Geodesic Morphological Operations},
    Booktitle  = {British Machine Vision Conference (BMVC)},
    Year = {2022}
    }


    @article{VelascoSIAM2022,
    author = {Velasco-Forero, Santiago and Pag\`{e}s, R. and Angulo, Jesus},
    title = {Learnable Empirical Mode Decomposition based on Mathematical Morphology},
    journal = {SIAM Journal on Imaging Sciences},
    volume = {15},
    number = {1},
    pages = {23-44},
    year = {2022},
    }