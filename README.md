# DGMM2024_comptage_cellule
code for DGMM2024: Counting melanocytes with trainable $h$-maxima and connected components counting layers
A version more clean is coming.

For now this script is for test only:

1.change the variable DATA_DIR in main.py to the dir containing database_melanocytes_trp1
2.change the variable ROOT_PATH in main.py to the current root dir ("./DGMM2024_comptage_cellule")
3.For test: python main.py
this will load the pretrained model weight and using the preprocessed inputs in ./DGMM2024_comptage_cellule/best_h_dataset255/input_np (preprocessed using operation opening-closing with structural element size=3)
