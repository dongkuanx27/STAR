# STAR

This is a reference implementation for Spatio-Temporal Attentive RNN for Node Classification in Temporal Attributed Graphs (IJCAI'19).

Please feel free to contact Dongkuan Xu (dux19@psu.edu) if you have any question.

# Notes
1. The datasets we use are included in the folder 'Data'

2. The program that can run on the DBLP3 dataset is included in the folder 'Code'

3. When running the main_DBLP3.py, you should 
		3.1 change the data path (Lines 366 & 367)
		3.2 change 'num_classes' (Line 383) according to the dataset
		3.3 change the saving path (Line 541)

4. In Line 541, the results include
		4.1 Alpha: the temporal attention values of the test set
		4.2 Beta: the patial attention values of neighbors of the test set
		4.3 X_test_idx: the idx of the test set
		4.4 y_test: the label of the test set
		4.5 samples_idx: the idx of neighbors of the test set (k-hop neighbors)

5. In order to get a good results, you should well tune
		5.1 lr: learning rate
		5.2 training_iters: training epochs
		5.3 lambda_l2_reg: penalty for parameters
		5.4 lambda_reg_att: penalty for multiple temporal attention units

# Citing
Please consider citing the following paper if you find STAR useful for your research:

@inproceedings{xu2019sp,
  
  title={Spatio-Temporal Attentive RNN for Node Classification in Temporal Attributed Graphs},
  
  author={Xu, Dongkuan and Cheng, Wei and Luo, Dongsheng and Liu, Xiao and Zhang, Xiang},
  
  booktitle={Proceedings of the 29th International Joint Conference on Artificial Intelligence},
  
  year={2019}
}
