


param_dict = {"epochs" : 200,
            "dict" : "JEC/Models/",
            "re-train" : 2, # Load previous save, train again, then save over
            "use_distributed_gpu" : True,
            "pre_train_epochs" : 1,
            "batch_size" : 4096,
            "pre_train_batch_size" : 4096,
            "Phi_sizes" : (100, 100, 128),
            "F_sizes" : (100, 100, 100),
            "DNN_sizes" : (64, 64, 64),
            "learning_rate" : 1e-4,
            "clipnorm" : None,
            "l2_reg" : 1e-6,
            "d_l1_reg" : 1e-3,
            "d_multiplier" : 0.5,
            }

dataset_dict = {"n" : 1000000,
                "cache_dir" : "data",
                "amount" : 4,
                "momentum_scale" : 1000,
                "pad" : 150,
                "pt_lower" : 500,
                "pt_upper" : 1000,
                "eta" : 2.4,
                "quality" : 2,
                }