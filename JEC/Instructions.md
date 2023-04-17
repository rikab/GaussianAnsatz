<!-- Packages -->

- Tensorflow 
- Keras
- Energyflow
- matplotlib 
- numpy
- scipy.special
- sklearn.feature_selection
- scipy.stats

For namespace reasons, do NOT have the utils package. 

<!-- Download Data -->

1. Open "download_dataset.py"
2. Edit the cache_dir parameter to wherever you would like to store the dataset. Keep in mind "/datasets" will be appended to whatever you say
3. Make sure the amount parameter to 1.00, NOT 1 (int). This will download the entire dataset (If a float, it will download amount fraction, if an int, it will download that # of files)
4. Run download_dataset.py

<!-- Config Network Parameters -->

1. Open "JEC/config.py"
2. Make sure the parameters "cache_dir" and "amount" match the download location and amount
3. By default, the parameter "n" is 1e6. This should probably be increased to 1e7 or even 1e8 to reflect the much larger dataset.
4. Note that if "use_distributed_gpu" is True, then "clipnorm" must be None. 
5. The rest of the hyperparameters can be left as is. All 4 networks will share this global list.
6. Other hyperparameters can be adjust as needed, but the defaults should be fine.


<!-- Training -->

1. There are 4 networks to train. Each can be run with "JEC/DNN.py", "JEC/EFN.py", "JEC/PFN.py", "JEC.PFN_PID.py". All four files should (hopefully) work out of the box if the above setup is correct
2. Each training has 2 stages: Pre-training, and training, both of which are set to verbose. The only purpose of pre-training is to initialize good parameters to prevent exploding losses later - if it is taking too long, the pre_train_batch_size can be increases.
3. It is important to monitor the "MI" variable during training. If it becomes NaN or a large negative number, the training has failed.
4. If the training has failed, one can attempt to reroll the random seed by training again. If this fails, one can change the hyperparameters in the config to make training less aggressive (reduce learning rate, increase regularizers, etc). I have chosen the hyperparameters to minimize this risk, and this is also less likely to occur with larger n.
4. The training will then repeat "re-train" number of times, with a lower learning rate and higher d-regularization. This is part of the training schedule to aid in convergence. The networks will really run for (epochs) * (re-train + 1) epochs.
6. A plot of the training MI is printed to the model save directory. 

<!-- Plots -->

1. Run "JEC/plots.py". This should work out of the box.
2. Plots will appear in "JEC/Plots".